import numpy as np
import torch
import torch.distributions as dist

from problems import BaseProblem, TSPProblem


class BaseParticlePopulationModel:
    def __init__(self, n_particles, problem: BaseProblem, device="cpu"):
        self.n_particles = n_particles
        self.problem = problem
        self.device = device

        self.population: torch.Tensor = self.initialize_population()
        initial_solutions = self.decode_solutions(return_log_probs=False)
        initial_costs = problem.evaluate(initial_solutions)
        self.update_metadata(initial_costs)

    def initialize_population(self) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode_solutions(self, return_log_probs=False):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, wc1c2: torch.Tensor):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def update_metadata(self, costs: torch.Tensor):
        raise NotImplementedError("This method should be overridden by subclasses.")

class TSPVectorEdgePP(BaseParticlePopulationModel):
    """
    Each particle is represented as a continuous vector of size n_cities.
    Values in the vector are in range [0, 1], representing the priority of visiting each city.
    Tours are decoded by gradually selecting the city via distribution formed by softmax over unvisited cities' priorities.
    """

    def __init__(self, n_particles: int, problem: TSPProblem, device="cpu"):
        self.n_cities = problem.n_cities
        self.k_sparse = problem.k_sparse
        self.dim = self.n_cities * self.k_sparse
        super().__init__(n_particles, problem, device=device)
        

    def initialize_population(self) -> torch.Tensor:
        self.val_pbest = torch.full((self.n_particles,), float("inf"), device=self.device)
        self.val_gbest = torch.tensor(float("inf"), device=self.device)
        self.pbest = torch.zeros(
            (self.n_particles, self.dim), dtype=torch.float, device=self.device
        )
        self.gbest = torch.zeros((self.dim,), dtype=torch.float, device=self.device)
        # self.velocity = torch.rand(
        #     (self.n_particles, self.dim), dtype=torch.float, device=self.device
        # ) * 0.1  - 0.05  # Random values in [-0.05, 0.05]
        self.velocity = torch.zeros(
            (self.n_particles, self.dim), dtype=torch.float, device=self.device
        )
        population = []
        for _ in range(self.n_particles):
            # particle = (
            #     torch.randn(self.dim, dtype=torch.float, device=self.device) * 2 - 1
            # )  # Random values in [-1, 1]
            particle = torch.ones(self.dim, dtype=torch.float, device=self.device) + torch.randn(self.dim, dtype=torch.float, device=self.device) * 0.1
            population.append(particle)
        return torch.stack(population)

    def step(self, wc1c2: torch.Tensor, using_random: bool = False):
        # wc1c2: shape (n_particles, 3) -> (w, c1, c2), tensor containing model gradients
        w = wc1c2[:, 0].unsqueeze(1)  # shape: (n_particles, 1)
        c1 = wc1c2[:, 1].unsqueeze(1)  # shape: (n_particles, 1)
        c2 = wc1c2[:, 2].unsqueeze(1)  # shape: (n_particles, 1)
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
        self.population = self.population + self.velocity

    def decode_solutions(self, return_log_probs=False):
        mat = torch.full((self.n_particles, self.n_cities, self.n_cities), float('-inf'), device=self.population.device)
        mat[:, self.problem.pyg_data.edge_index[0], self.problem.pyg_data.edge_index[1]] = self.population
        start = torch.randint(low=0, high=self.n_cities, size=(self.n_particles,), device=self.population.device)
        mask = torch.ones((self.n_particles, self.n_cities), device=self.population.device)
        mask[torch.arange(self.n_particles), start] = 0
        paths_list = [start]; log_probs_list = []
        prev = start
        for i in range(self.n_cities - 1):
            cur_mat = mat[torch.arange(self.n_particles), prev, :]  # shape: (n_particles, n_cities)
            masked_mat = cur_mat.masked_fill(mask == 0, float('-inf'))  # shape: (n_particles, n_cities)
            # if any row is all -inf (should not happen), replace with uniform distribution
            all_inf_mask = (masked_mat == float('-inf')).all(dim=1)
            if all_inf_mask.any():
                masked_mat[all_inf_mask] = 0.0  # uniform logits
            d = dist.Categorical(logits=masked_mat)
            actions = d.sample()  # shape: (n_particles,)
            paths_list.append(actions)
            log_probs = d.log_prob(actions)  #shape: (n_particles,)
            log_probs_list.append(log_probs)
            mask[torch.arange(self.n_particles), actions] = 0
            prev = actions
        
        paths = torch.stack(paths_list)  # shape: (n_cities, n_particles)
        paths = paths.T  # shape: (n_particles, n_cities)
        # for tour in paths:
        #     assert len(set(tour.tolist())) == self.n_cities, f"Invalid tour decoded: \n{tour.tolist()}"
        if return_log_probs:
            log_probs = torch.stack(log_probs_list)  # shape: (n_cities, n_particles)
            return paths, log_probs
        return paths

    def decode_solutions_eval(self):
        # population: shape (n_particles, n_cities)
        paths = torch.argsort(self.population, dim=1)  # shape: (n_particles, n_cities)
        return paths

    def update_metadata(self, costs: torch.Tensor):
        self.population = self.population.detach().clone()
        self.velocity = self.velocity.detach().clone()
        self.pbest = self.pbest.detach().clone()
        self.gbest = self.gbest.detach().clone()
        better_pbest_mask = costs < self.val_pbest
        self.val_pbest[better_pbest_mask] = costs[better_pbest_mask]
        self.pbest[better_pbest_mask] = self.population[better_pbest_mask].detach().clone()

        # Update global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost < self.val_gbest:
            self.val_gbest = min_cost
            self.gbest = self.population[min_idx].detach().clone()
        return costs
