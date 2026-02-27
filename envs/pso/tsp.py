import torch
import torch.distributions as dist
from problems import TSPProblem

from .base import BaseEnvPSOProblem


class TSPEnvVectorEdge(BaseEnvPSOProblem):
    """
    Each particle is represented as a continuous vector of size n_cities.
    Values in the vector are in range [0, 1], representing the priority of visiting each city.
    Tours are decoded by gradually selecting the city via distribution formed by softmax over unvisited cities' priorities.
    """

    def __init__(
        self,
        n_particles: int,
        problem: TSPProblem,
        device="cpu",
        eval_n_starts=1,
        init_n_heuristic=0,
        **kwargs,
    ):
        self.n_cities = problem.n_cities
        self.k_sparse = problem.k_sparse
        self.dim = self.n_cities * self.k_sparse
        self.eval_n_starts = eval_n_starts
        self.init_n_heuristic = init_n_heuristic
        assert (
            self.init_n_heuristic <= self.n_particles
        ), "Cannot initialize more heuristic particles than total particles."
        super().__init__(n_particles, problem, device=device)

    def initialize_population(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pbest = torch.zeros(
            (self.n_particles, self.dim), dtype=torch.float, device=self.device
        )
        gbest = torch.zeros((self.dim,), dtype=torch.float, device=self.device)
        velocity = torch.zeros(
            (self.n_particles, self.dim), dtype=torch.float, device=self.device
        )
        population = []
        for i in range(self.init_n_heuristic):
            start_city = i % self.n_cities
            nn_tour = self._nearest_neighbor_tour(start_city)
            edge_weights = self._tour_to_edge_weights(nn_tour)
            # Small noise to break symmetry
            particle = edge_weights + torch.randn_like(edge_weights) * 0.3
            population.append(particle)
        for _ in range(self.n_particles - self.init_n_heuristic):
            particle = torch.randn(self.dim, dtype=torch.float, device=self.device)
            population.append(particle)
        return torch.stack(population), velocity, pbest, gbest

    def _nearest_neighbor_tour(self, start_city: int = 0) -> torch.Tensor:
        """Greedy nearest-neighbour heuristic: always visit the closest unvisited city."""
        dist_matrix = self.problem.distance_matrix  # (n_cities, n_cities)
        visited = set([start_city])
        tour = [start_city]
        current = start_city
        for _ in range(self.n_cities - 1):
            dists = dist_matrix[current].clone()
            for v in visited:
                dists[v] = float("inf")
            next_city = dists.argmin().item()
            visited.add(next_city)
            tour.append(next_city)
            current = next_city
        return torch.tensor(tour, device=self.device)

    def _tour_to_edge_weights(self, tour: torch.Tensor) -> torch.Tensor:
        """Convert a tour (permutation) to an edge-weight vector.
        Edges used in the tour get weight 3.0; others stay at 0.0."""
        weights = torch.zeros(self.dim, device=self.device)
        edge_index = self.problem.pyg_data.edge_index  # (2, n_edges)
        u_nodes = tour
        v_nodes = torch.roll(tour, -1)
        # Fully-vectorised: broadcast (n_edges, 1) vs (1, n_cities)
        u_match = edge_index[0].unsqueeze(1) == u_nodes.unsqueeze(0)
        v_match = edge_index[1].unsqueeze(1) == v_nodes.unsqueeze(0)
        is_tour_edge = (u_match & v_match).any(dim=1)  # (n_edges,)
        weights[is_tour_edge] = 3.0
        return weights

    def step(self, wc1c2: torch.Tensor, temperature: float, using_random: bool = True):
        w = wc1c2[..., 0]  # shape: (n_particles, dim)
        c1 = wc1c2[..., 1]  # shape: (n_particles, dim)
        c2 = wc1c2[..., 2]  # shape: (n_particles, dim)
        if using_random:
            r1 = torch.rand((self.n_particles, 1), device=self.device)
            r2 = torch.rand((self.n_particles, 1), device=self.device)
            c1 = c1 * r1
            c2 = c2 * r2

        self.velocity = (
            w * self.velocity
            + c1 * (self.pbest - self.population)
            + c2 * (self.gbest.unsqueeze(0) - self.population)
        )
        # Clamp velocity to prevent divergence
        self.velocity = torch.clamp(self.velocity, -4.0, 4.0)
        self.population = self.population + self.velocity

        # Get stochastic cost for training signal
        _, costs_stochastic = self.decode_solutions(
            stochastic=True, temperature=temperature
        )

        # Get deterministic cost for metadata update
        _, costs = self.decode_solutions_eval()
        self.update_metadata(costs)

        return (
            (self.population, self.velocity, self.pbest, self.gbest, self.problem),
            -costs_stochastic,
            None,
            None,
            {},
        )

    def decode_solutions(
        self,
        stochastic: bool = True,
        start: torch.Tensor = None,
        temperature: float = 1.0,
    ):
        """
        Decode particle positions into TSP tours.
        Args:
            stochastic: if True, sample from Categorical distribution (for training);
                        if False, use greedy argmax (for evaluation).
            start: optional tensor of shape (n_particles,) specifying start cities.
                   If None, a random start city is sampled per particle.
            temperature: temperature for stochastic sampling (higher = more random).
        """
        # convert population edge weights into a (n_particles, n_cities, n_cities) matrix for decoding
        mat = torch.full(
            (self.n_particles, self.n_cities, self.n_cities),
            float("-inf"),
            device=self.population.device,
        )
        mat[
            :, self.problem.pyg_data.edge_index[0], self.problem.pyg_data.edge_index[1]
        ] = self.population

        # Sample start cities if not provided
        if start is None:
            start = torch.randint(
                low=0,
                high=self.n_cities,
                size=(self.n_particles,),
                device=self.population.device,
            )

        # mask to keep track of visited cities: 1 = unvisited, 0 = visited. Initially all cities are unvisited except the start city.
        mask = torch.ones(
            (self.n_particles, self.n_cities), device=self.population.device
        )
        mask[torch.arange(self.n_particles), start] = 0

        # iteratively build tours by selecting the next city based on the current city's edge weights, masked by unvisited cities.
        paths_list = [start]
        prev = start
        for i in range(self.n_cities - 1):
            cur_mat = mat[
                torch.arange(self.n_particles), prev, :
            ]  # shape: (n_particles, n_cities)
            masked_mat = cur_mat.masked_fill(
                mask == 0, float("-inf")
            )  # shape: (n_particles, n_cities)
            # check if any row is all -inf (should not happen)
            all_inf_mask = (masked_mat == float("-inf")).all(dim=1)
            if all_inf_mask.any():
                # if happend, replace with uniform distribution on unvisited cities
                masked_mat[all_inf_mask] = 0.0  # uniform logits
                masked_mat = masked_mat.masked_fill(mask == 0, float("-inf"))
            if stochastic:
                # sample from distribution with temperature
                actions = dist.Categorical(
                    logits=masked_mat / temperature
                ).sample()  # shape: (n_particles,)
            else:
                # greedy argmax
                actions = torch.argmax(masked_mat, dim=1)  # shape: (n_particles,)
            paths_list.append(actions)
            mask[torch.arange(self.n_particles), actions] = 0
            prev = actions

        paths = torch.stack(paths_list)  # shape: (n_cities, n_particles)
        paths = paths.T  # shape: (n_particles, n_cities)

        # sanity check: each tour should visit all cities exactly once
        for tour in paths:
            assert (
                len(set(tour.tolist())) == self.n_cities
            ), f"Invalid tour decoded: \n{tour.tolist()}"

        # evaluate costs of decoded tours
        costs = self.problem.evaluate(paths)
        return paths, costs

    def decode_solutions_eval(self):
        """
        Multi-start greedy decode: try multiple start cities per particle,
        keep the tour with the lowest cost.  This gives a robust, deterministic
        measure of how good each particle's position is.

        Args:
            n_starts: number of start cities to try per particle.
                      Defaults to n_cities (full sweep).
        Returns:
            best_paths: (n_particles, n_cities)
            best_costs: (n_particles,)
        """
        if self.eval_n_starts is None:
            n_starts = self.n_cities
        else:
            n_starts = self.eval_n_starts
        starts = torch.rand(
            self.n_particles, self.n_cities, device=self.device
        ).argsort(dim=1)[:, :n_starts]
        best_paths = None
        best_costs = torch.full((self.n_particles,), float("inf"), device=self.device)
        for s in starts.T:
            paths, costs = self.decode_solutions(stochastic=False, start=s)
            improved = costs < best_costs
            if best_paths is None:
                best_paths = paths
                best_costs = costs
            else:
                best_paths[improved] = paths[improved]
                best_costs[improved] = costs[improved]
        return best_paths, best_costs

    def update_metadata(self, costs: torch.Tensor):
        self.population = self.population.detach().clone()
        self.velocity = self.velocity.detach().clone()
        self.pbest = self.pbest.detach().clone()
        self.gbest = self.gbest.detach().clone()
        better_pbest_mask = costs < self.val_pbest
        self.val_pbest[better_pbest_mask] = costs[better_pbest_mask]
        self.pbest[better_pbest_mask] = (
            self.population[better_pbest_mask].detach().clone()
        )

        # Update global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost < self.val_gbest:
            self.val_gbest = min_cost
            self.gbest = self.population[min_idx].detach().clone()
        return costs
