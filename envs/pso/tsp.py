import torch
import torch.distributions as dist

from utils import timeit

from ..problems import TSPBatchProblem
from .base import BaseEnvPSOBatchProblem


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
        reward_mode="greedy",
        train_n_starts=1,
        eval_n_starts=1,
        init_n_heuristic=0,
        patience=5,
        auto_reset=True,
        use_local_search=False,
        **kwargs,
    ):
        super().__init__(n_particles, problem, use_local_search, auto_reset, patience)
        self.n_cities = problem.n_cities
        self.k_sparse = problem.k_sparse
        self.dim = self.n_cities * self.k_sparse
        self.reward_mode = reward_mode

        if init_n_heuristic is None:
            self.init_n_heuristic = 0
        if isinstance(init_n_heuristic, float) and 0 < init_n_heuristic <= 1.0:
            self.init_n_heuristic = int(init_n_heuristic * self.n_particles)
        elif (
            isinstance(init_n_heuristic, int)
            and init_n_heuristic >= 0
            and init_n_heuristic <= self.n_particles
        ):
            self.init_n_heuristic = init_n_heuristic
        else:
            raise ValueError(
                "init_n_heuristic must be a non-negative int less than or equal to n_particles or a float in (0, 1]."
            )

        if train_n_starts is None:
            self.train_n_starts = 1
        else:
            if isinstance(train_n_starts, int):
                self.train_n_starts = min(train_n_starts, self.n_cities)
            elif isinstance(train_n_starts, float) and 0 < train_n_starts <= 1.0:
                self.train_n_starts = max(1, int(train_n_starts * self.n_cities))
            else:
                raise ValueError(
                    "train_n_starts must be None, an int, or a float in (0, 1]."
                )

        if eval_n_starts is None:
            self.eval_n_starts = self.n_cities
        else:
            if isinstance(eval_n_starts, int):
                self.eval_n_starts = min(eval_n_starts, self.n_cities)
            elif isinstance(eval_n_starts, float) and 0 < eval_n_starts <= 1.0:
                self.eval_n_starts = max(1, int(eval_n_starts * self.n_cities))
            else:
                raise ValueError(
                    "eval_n_starts must be None, an int, or a float in (0, 1]."
                )

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

        heuristic_population = []
        for i in range(self.init_n_heuristic):
            for problem_idx in range(self.batch_size):
                start_city = i % self.n_cities
                nn_tour = self._nearest_neighbor_tour(problem_idx, start_city)
                edge_weights = self._tour_to_edge_weights(problem_idx, nn_tour)
                # Small noise to break symmetry
                particle = edge_weights + torch.randn_like(edge_weights) * 0.3
                heuristic_population.append(particle)
        if len(heuristic_population) == 0:
            population = torch.rand((self.batch_size, self.n_particles, self.dim))
        elif len(heuristic_population) == self.n_particles:
            population = torch.stack(heuristic_population).reshape(
                self.batch_size, self.n_particles, self.dim
            )
        else:
            random_population = torch.rand(
                (self.batch_size, self.n_particles - self.init_n_heuristic, self.dim)
            )
            population = torch.cat(
                [torch.stack(heuristic_population), random_population], dim=1
            )
        return population.to(self.device), velocity, pbest, gbest

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
        return torch.tensor(tour)

    def _tour_to_edge_weights(
        self, problem_idx: int, tour: torch.Tensor
    ) -> torch.Tensor:
        """Convert a tour (sequence of city indices) into edge weights for the population representation."""
        edge_index = self.problem.pyg_data.edge_index  # (2, num_edges)
        # Create a mapping from (u, v) to edge index
        edge_to_idx = {
            (u.item(), v.item()): idx for idx, (u, v) in enumerate(edge_index.t())
        }
        weights = torch.zeros(self.dim)
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
        using_random: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        wc1c2 = wc1c2.to(self.device)
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
        _, costs, _, cost_ls, avg_costs = self.decode_solutions_eval()
        delta_val_pbest, delta_val_gbest = self.update_metadata(costs, cost_ls)
        
        # Update patience counter for early stopping
        improved = (delta_val_gbest > 0)  # (batch_size, )
        self.cnt_patience = torch.where(improved, torch.zeros_like(self.cnt_patience, device=self.device), self.cnt_patience + 1)
        done = self.cnt_patience >= self.patience

        if self.reward_mode == "stochastic":
            _, stochastic_costs = self.decode_solutions(stochastic=True, temperature=temperature)
            reward = -stochastic_costs
        elif self.reward_mode == "greedy":
            used_costs = cost_ls if self.use_local_search else avg_costs
            reward = -used_costs  # (B, P)
        elif self.reward_mode == "pbest":
            reward = -self.val_pbest  # (B, P)
        elif self.reward_mode == "gbest":
            reward = -self.val_gbest  # (B,)
        elif self.reward_mode == "delta_pg":
            reward = delta_val_pbest + 0.5 * delta_val_gbest.unsqueeze(-1)  # (B, P)
        elif self.reward_mode == "delta_g":
            reward = torch.clamp(delta_val_gbest, min=0.0)  # (B,)
        elif self.reward_mode == "delta_g_raw":
            reward = delta_val_gbest  # (B,)
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")

        if self.auto_reset:
            # Automatically reset done instances to keep training stable
            # No need to do this for evaluation, as we want to see the final performance after patience runs out
            self._auto_reset_done(done)

        return (
            (self.population, self.velocity, self.pbest, self.gbest, self.problem),
            reward,
            done,
            None,
            {
                "population_costs": costs.cpu()
                .numpy()
                .tolist(),  # List of lists: (batch_size, n_particles)
                "population_costs_ls": cost_ls.cpu()
                .numpy()
                .tolist(),  # List of lists: (batch_size, n_particles)
                "gbest_cost": self.val_gbest.cpu()
                .numpy()
                .tolist(),  # List: (batch_size,)
            },
        )

    def step_train(self, wc1c2, temperature=1.0, using_random=True):
        return self.step(
            wc1c2,
            using_random=using_random,
            temperature=temperature,
            stochastic=True,
        )

    def step_eval(self, wc1c2: torch.Tensor, using_random: bool = True):
        return self.step(wc1c2, using_random=using_random)

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
        device = self.population.device
        B = self.batch_size
        P = self.n_particles
        N = self.n_cities
        k = self.k_sparse
        BP = B * P

        # --- Sparse lookup tables: avoids O(B*P*N²) dense mat ---
        # edge_index: (2, B*N*k), edges ordered by src within each batch block
        # (src layout: [0,0,...0, 1,1,...,1, ..., N-1,...,N-1] with k repeats)
        edge_index = self.problem.pyg_data.edge_index
        edge_index_reshaped = edge_index.view(2, B, N * k)
        batch_offsets = torch.arange(B, device=device) * N
        local_dst = edge_index_reshaped[1] - batch_offsets[:, None]  # (B, N*k)

        # dst_table[b, i, j] = j-th neighbor destination of node i in batch b
        dst_table = local_dst.view(B, N, k)  # (B, N, k)

        # wt_flat[bp, i, :] = k edge weights from node i (zero-copy view of population)
        # population layout (B, P, N*k) -> (BP, N, k): bp = b*P + p
        wt_flat = self.population.view(BP, N, k)  # (BP, N, k)

        # b_indices[bp] = batch index for element bp in the BP dimension (bp // P)
        b_indices = torch.arange(B, device=device).repeat_interleave(P)  # (BP,)
        bp_arange = torch.arange(BP, device=device)  # (BP,)

        # Sample start cities if not provided
        if start is None:
            start = torch.randint(0, N, size=(BP,), device=device)
        else:
            start = start.to(device).flatten()
            assert start.shape == (BP,)

        # Mask to keep track of visited cities: 1 = unvisited, 0 = visited
        mask = torch.ones(BP, N, device=device)
        mask[bp_arange, start] = 0

        # Iteratively build tours
        paths_list = [start]
        prev = start  # (BP,)
        for _ in range(N - 1):
            # Sparse edge weights and destinations from current node
            cur_weights = wt_flat[bp_arange, prev, :]  # (BP, k)
            cur_dst = dst_table[b_indices, prev, :]  # (BP, k)

            # Scatter sparse weights into a dense (BP, N) logit vector
            logits = torch.full((BP, N), float("-inf"), device=device)
            logits.scatter_(1, cur_dst, cur_weights)

            # Mask visited cities
            masked_logits = logits.masked_fill(mask == 0, float("-inf"))

            # Fallback: if a row is all -inf (sparse graph misses a transition),
            # use uniform distribution over remaining unvisited cities
            all_inf_mask = (masked_logits == float("-inf")).all(dim=-1)  # (BP,)
            if all_inf_mask.any():
                fallback = torch.zeros(all_inf_mask.sum(), N, device=device)
                fallback.masked_fill_(mask[all_inf_mask] == 0, float("-inf"))
                masked_logits[all_inf_mask] = fallback

            if stochastic:
                # Sample from distribution with temperature
                actions = dist.Categorical(
                    logits=masked_logits / temperature
                ).sample()  # (BP,)
            else:
                # Greedy argmax
                actions = masked_logits.argmax(dim=-1)  # (BP,)

            paths_list.append(actions)
            mask[bp_arange, actions] = 0
            prev = actions

        paths = torch.stack(paths_list, dim=-1)  # (BP, N)

        # # sanity check: each tour should visit all cities exactly once
        # for tour in paths:
        #     assert (
        #         len(set(tour.tolist())) == self.n_cities
        #     ), f"Invalid tour decoded: \n{tour.tolist()}"

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
        n_starts = self.eval_n_starts
        # Generate random starts: (batch_size, n_particles, n_starts)
        starts = torch.rand(self.batch_size, self.n_particles, self.n_cities).argsort(
            dim=-1
        )[..., :n_starts]

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

        # Find the mean cost for each particle across its starts (for metadata update)
        mean_costs = costs_reshaped.mean(dim=-1)  # (batch_size, n_particles)

        # Find the best start for each particle
        best_indices = costs_reshaped.argmin(dim=-1)  # (batch_size, n_particles)
        batch_idx = torch.arange(self.batch_size)[:, None]
        particle_idx = torch.arange(original_n_particles)[None, :]
        best_costs = costs_reshaped[
            batch_idx, particle_idx, best_indices
        ]  # (batch_size, n_particles)
        best_paths = paths_reshaped[
            batch_idx, particle_idx, best_indices
        ]  # (batch_size, n_particles, n_cities)

        ### Local search on best paths
        best_paths_ls = self.problem.local_search(best_paths)
        best_costs_ls = self.problem.evaluate(best_paths_ls)
        return best_paths, best_costs, best_paths_ls, best_costs_ls, mean_costs
