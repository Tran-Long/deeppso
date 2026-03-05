import torch
import torch.distributions as dist

from utils import timeit

from ..problems import TSPProblem,TSPBatchProblem
from .base import BaseEnvPSOBatchProblem
from torch_geometric.data import Batch

class TSPEnvVectorEdgeBatch(BaseEnvPSOBatchProblem):
    """
    Each particle is represented as a continuous vector of size n_cities.
    Values in the vector are in range [0, 1], representing the priority of visiting each city.
    Tours are decoded by gradually selecting the city via distribution formed by softmax over unvisited cities' priorities.
    """

    def __init__(
        self,
        n_particles: int,
        problem: TSPBatchProblem,
        device="cpu",
        eval_n_starts=1,
        init_n_heuristic=0,
        **kwargs,
    ):
        self.n_cities = problem.n_cities
        self.k_sparse = problem.k_sparse
        self.dim = self.n_cities * self.k_sparse
        self.eval_n_starts = eval_n_starts
        assert (
            init_n_heuristic <= n_particles
        ), "Cannot initialize more heuristic particles than total particles."
        self.init_n_heuristic = init_n_heuristic
        super().__init__(n_particles, problem, device=device)

    def initialize_population(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pbest = torch.zeros(
            (self.batch_size, self.n_particles, self.dim),
            dtype=torch.float,
            device=self.device,
        )
        gbest = torch.zeros(
            (self.batch_size, self.dim), dtype=torch.float, device=self.device
        )
        velocity = torch.zeros(
            (self.batch_size, self.n_particles, self.dim),
            dtype=torch.float,
            device=self.device,
        )
        if self.init_n_heuristic is None:
            self.init_n_heuristic = 0
        if (
            isinstance(self.init_n_heuristic, float)
            and 0 < self.init_n_heuristic <= 1.0
        ):
            self.init_n_heuristic = int(self.init_n_heuristic * self.n_particles)
        elif not isinstance(self.init_n_heuristic, int) or self.init_n_heuristic < 0:
            raise ValueError(
                "init_n_heuristic must be a non-negative int or a float in (0, 1]."
            )

        heuristic_population = []
        for i in range(self.init_n_heuristic):
            for problem_idx in range(self.batch_size):
                start_city = i % self.n_cities
                nn_tour = self._nearest_neighbor_tour(problem_idx, start_city)
                edge_weights = self._tour_to_edge_weights(problem_idx, nn_tour)
                # Small noise to break symmetry
                particle = edge_weights + torch.randn_like(edge_weights) * 0.3
                heuristic_population.append(particle)
        heuristic_population = torch.stack(heuristic_population).reshape(
            self.batch_size, self.init_n_heuristic, self.dim
        )
        if self.init_n_heuristic < self.n_particles:
            random_population = torch.rand(
                (self.batch_size, self.n_particles - self.init_n_heuristic, self.dim),
                device=self.device,
            )
            population = torch.cat([heuristic_population, random_population], dim=1)
        else:
            population = heuristic_population
        return population, velocity, pbest, gbest

    def _nearest_neighbor_tour(
        self, problem_idx: int, start_city: int = 0
    ) -> torch.Tensor:
        """Greedy nearest-neighbour heuristic: always visit the closest unvisited city."""
        dist_matrix = self.problem.distance_matrix[problem_idx]  # (n_cities, n_cities)
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

    def _tour_to_edge_weights(
        self, problem_idx: int, tour: torch.Tensor
    ) -> torch.Tensor:
        """Convert a tour (sequence of city indices) into edge weights for the population representation."""
        edge_index = self.problem.pyg_data.edge_index  # (2, num_edges)
        # Create a mapping from (u, v) to edge index
        edge_to_idx = {
            (u.item(), v.item()): idx for idx, (u, v) in enumerate(edge_index.t())
        }
        weights = torch.zeros(self.dim, device=self.device)
        for i in range(len(tour)):
            u = tour[i].item()
            v = tour[(i + 1) % len(tour)].item()  # Wrap around to form a cycle
            edge_idx = edge_to_idx.get((u, v))
            if edge_idx is not None:
                weights[edge_idx] = 1.0  # Set weight for edges in the tour
            else:
                raise ValueError(f"Edge ({u}, {v}) not found in problem graph.")
        return weights

    def step(
        self,
        wc1c2: torch.Tensor,
        temperature: float = 1.0,
        using_random: bool = True,
        return_stochastic_cost: bool = True,
        **kwargs,
    ):
        w = wc1c2[..., 0]  # shape: (batch_size, n_particles, dim)
        c1 = wc1c2[..., 1]  # shape: (batch_size, n_particles, dim)
        c2 = wc1c2[..., 2]  # shape: (batch_size, n_particles, dim)
        if using_random:
            r1 = torch.rand((self.batch_size, self.n_particles, 1), device=self.device)
            r2 = torch.rand((self.batch_size, self.n_particles, 1), device=self.device)
            c1 = c1 * r1
            c2 = c2 * r2

        self.velocity = (
            w * self.velocity
            + c1 * (self.pbest - self.population)
            + c2 * (self.gbest.unsqueeze(1) - self.population)
        )
        # Clamp velocity to prevent divergence
        self.velocity = torch.clamp(self.velocity, -4.0, 4.0)
        self.population = self.population + self.velocity

        # Get deterministic cost for metadata update
        _, costs = self.decode_solutions_eval()
        self.update_metadata(costs)

        # Get stochastic cost for training signal
        if return_stochastic_cost:
            _, return_cost = self.decode_solutions(
                stochastic=True, temperature=temperature
            )
        else:
            return_cost = (
                costs  # Use deterministic cost if not returning stochastic cost
            )

        return (
            (self.population, self.velocity, self.pbest, self.gbest, self.problem),
            -return_cost,  # Negate cost to make it a reward (maximize is better)
            None,
            None,
            {
                "population_costs": costs.detach().cpu().numpy().tolist(),
                "gbest_cost": self.val_gbest.cpu().numpy().tolist(),
            },
        )

    def step_train(self, wc1c2, temperature=1.0, using_random=True):
        return self.step(
            wc1c2,
            return_stochastic_cost=True,
            temperature=temperature,
            using_random=using_random,
        )

    def step_eval(self, wc1c2: torch.Tensor, using_random: bool = True):
        return self.step(wc1c2, return_stochastic_cost=False, using_random=using_random)

    def decode_solutions(
        self,
        stochastic: bool = True,
        start: torch.Tensor = None,
        temperature: float = 1.0,
    ):
        """
        Decode particle positions into TSP tours (batched version).
        Args:
            stochastic: if True, sample from Categorical distribution (for training);
                        if False, use greedy argmax (for evaluation).
            start: optional tensor of shape (batch_size, n_particles) specifying start cities.
                   If None, a random start city is sampled per particle.
            temperature: temperature for stochastic sampling (higher = more random).
        Returns:
            paths: (batch_size, n_particles, n_cities)
            costs: (batch_size, n_particles)
        """
        # Convert population edge weights into a (batch_size * n_particles, n_cities, n_cities) matrix for decoding
        mat = torch.full(
            (self.batch_size * self.n_particles, self.n_cities, self.n_cities),
            float("-inf"),
            device=self.population.device,
        )
        # Use first problem's edge_index (assumes all problems have same structure)
        edge_index = self.problem.pyg_data.edge_index
        mat[:, edge_index[0], edge_index[1]] = self.population.view(-1, self.dim)

        # Sample start cities if not provided
        if start is None:
            start = torch.randint(
                low=0,
                high=self.n_cities,
                size=(self.batch_size * self.n_particles,),
                device=self.population.device,
            )
        else:
            start = start.view(-1)  # Flatten to (batch_size * n_particles,)
            assert start.shape == (self.batch_size * self.n_particles,)
        # Mask to keep track of visited cities: 1 = unvisited, 0 = visited
        mask = torch.ones(
            (self.batch_size * self.n_particles, self.n_cities),
            device=self.population.device,
        )
        mask[torch.arange(self.batch_size * self.n_particles), start] = 0

        # Iteratively build tours
        paths_list = [start]
        prev = start  # (batch_size * n_particles,)
        for i in range(self.n_cities - 1):
            # Gather edge weights from current city: mat[b, p, prev[b,p], :]
            cur_mat = mat[
                torch.arange(self.batch_size * self.n_particles), prev, :
            ]  # (batch_size * n_particles, n_cities)
            masked_mat = cur_mat.masked_fill(mask == 0, float("-inf"))

            # Check if any position is all -inf (shouldn't happen with valid sparse graph)
            all_inf_mask = (masked_mat == float("-inf")).all(
                dim=-1
            )  # (batch_size * n_particles,)
            if all_inf_mask.any():
                # Replace with uniform distribution on unvisited cities
                masked_mat[all_inf_mask] = 0.0
                masked_mat = masked_mat.masked_fill(mask == 0, float("-inf"))

            if stochastic:
                # Sample from distribution with temperature
                logits_flat = (masked_mat / temperature).reshape(-1, self.n_cities)
                actions_flat = dist.Categorical(logits=logits_flat).sample()
                actions = actions_flat.reshape(self.batch_size, self.n_particles)
            else:
                # Greedy argmax
                actions = torch.argmax(masked_mat, dim=-1)  # (batch_size * n_particles,)

            paths_list.append(actions)
            mask[torch.arange(self.batch_size * self.n_particles), actions] = 0
            prev = actions

        paths = torch.stack(paths_list, dim=-1)  # (batch_size * n_particles, n_cities)
        
        # sanity check: each tour should visit all cities exactly once
        for tour in paths:
            assert (
                len(set(tour.tolist())) == self.n_cities
            ), f"Invalid tour decoded: \n{tour.tolist()}"

        # Evaluate costs for each problem in batch
        paths = paths.view(self.batch_size, self.n_particles, self.n_cities)
        costs = self.problem.evaluate(paths)  # (batch_size, n_particles)
        return paths, costs

    def decode_solutions_eval(self):
        """
        Multi-start greedy decode: try multiple start cities per particle,
        keep the tour with the lowest cost (batched version).

        Vectorized implementation: process all starts in parallel by expanding
        the particle dimension.

        Returns:
            best_paths: (batch_size, n_particles, n_cities)
            best_costs: (batch_size, n_particles)
        """
        if self.eval_n_starts is None:
            n_starts = self.n_cities
        else:
            if isinstance(self.eval_n_starts, int):
                n_starts = min(self.eval_n_starts, self.n_cities)
            elif (
                isinstance(self.eval_n_starts, float) and 0 < self.eval_n_starts <= 1.0
            ):
                n_starts = max(1, int(self.eval_n_starts * self.n_cities))
            else:
                raise ValueError(
                    "eval_n_starts must be None, an int, or a float in (0, 1]."
                )

        # Generate random starts: (batch_size, n_particles, n_starts)
        starts = torch.rand(
            self.batch_size, self.n_particles, self.n_cities, device=self.device
        ).argsort(dim=-1)[..., :n_starts]

        # Save original state
        original_population = self.population
        original_n_particles = self.n_particles

        # Expand population: (batch_size, n_particles, dim) -> (batch_size, n_particles * n_starts, dim)
        expanded_population = original_population.unsqueeze(2).expand(
            -1, -1, n_starts, -1
        )
        expanded_population = expanded_population.reshape(self.batch_size, -1, self.dim)


        # Temporarily set expanded state for decode_solutions
        self.population = expanded_population
        self.n_particles = original_n_particles * n_starts

        paths, costs = self.decode_solutions(stochastic=False, start=starts)
        # Restore original state
        self.population = original_population
        self.n_particles = original_n_particles

        # Reshape results: (batch_size, n_particles * n_starts) -> (batch_size, n_particles, n_starts)
        costs_reshaped = costs.reshape(self.batch_size, original_n_particles, n_starts)
        paths_reshaped = paths.reshape(
            self.batch_size, original_n_particles, n_starts, self.n_cities
        )

        # Find the best start for each particle
        best_indices = costs_reshaped.argmin(dim=-1)  # (batch_size, n_particles)
        batch_idx = torch.arange(self.batch_size, device=self.device)[:, None]
        particle_idx = torch.arange(original_n_particles, device=self.device)[None, :]
        best_costs = costs_reshaped[batch_idx, particle_idx, best_indices] # (batch_size, n_particles)
        best_paths = paths_reshaped[batch_idx, particle_idx, best_indices] # (batch_size, n_particles, n_cities)

        return best_paths, best_costs
