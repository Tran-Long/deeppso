import numpy as np
import torch
from problems import BaseProblem, TSPProblem
import torch.distributions as dist

class BaseParticleModel:
    def __init__(self, n_particles, problem: BaseProblem, device='cpu'):
        self.n_particles = n_particles
        self.problem = problem
        self.device = device

        self.val_pbest = torch.full((n_particles,), float('inf'), device=self.device)
        self.val_gbest = torch.tensor(float('inf'), device=self.device)

        self.pbest = None
        self.gbest = None
        self.population: torch.Tensor = self.initialize_population()

    def initialize_population(self) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode_solutions(self, return_log_probs=False):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def step(self, wc1c2: torch.Tensor):
        raise NotImplementedError("This method should be overridden by subclasses.")

class TSPParticleVector(BaseParticleModel):
    """
    Each particle is represented as a continuous vector of size n_cities.
    Values in the vector are in range [0, 1], representing the priority of visiting each city.
    Tours are decoded by gradually selecting the city via distribution formed by softmax over unvisited cities' priorities.
    """
    def __init__(self, n_particles: int, problem: TSPProblem, device='cpu'):
        self.n_cities = problem.n_cities
        super().__init__(n_particles, problem, device=device)
        self.pbest = torch.zeros((n_particles, self.n_cities), dtype=torch.float, device=device)
        self.gbest = torch.zeros((self.n_cities,), dtype=torch.float, device=device)
        self.device = device

    def initialize_population(self) -> torch.Tensor:
        population = []
        for _ in range(self.n_particles):
            # particle = torch.ones(self.n_cities, dtype=torch.float, device=self.device) / self.n_cities
            particle = torch.rand(self.n_cities, dtype=torch.float, device=self.device)
            population.append(particle)
        return torch.stack(population)

    def step(self, wc1c2: torch.Tensor, inplace: bool = True):
        # wc1c2: shape (n_particles, 3) -> (w, c1, c2), tensor containing model gradients
        w = wc1c2[:, 0].unsqueeze(1)  # shape: (n_particles, 1)
        c1 = wc1c2[:, 1].unsqueeze(1)  # shape: (n_particles, 1)
        c2 = wc1c2[:, 2].unsqueeze(1)  # shape: (n_particles, 1)
        new_population = w * self.population + \
                                c1 * self.pbest + \
                                c2 * self.gbest.unsqueeze(0)
        new_population = new_population / new_population.sum(dim=1, keepdim=True)  # Normalize to [0,1]
        if inplace:
            self.population = new_population
        else:
            return new_population

    def decode_solutions(self, return_log_probs=False):
        # population: shape (n_particles, n_cities)
        n_particles = self.population.shape[0]
        mask = torch.ones((n_particles, self.n_cities), device=self.population.device)
        paths_list = []
        log_probs_list = []
        for i in range(self.n_cities):
            masked_pop = self.population.masked_fill(mask == 0, 0)
            masked_pop = masked_pop / masked_pop.sum(dim=1, keepdim=True)  # Re-normalize
            d = dist.Categorical(probs=masked_pop)
            actions = d.sample()  # shape: (n_particles,)
            paths_list.append(actions)
            log_probs = d.log_prob(actions)  # shape: (n_particles,)
            log_probs_list.append(log_probs)
            mask[torch.arange(n_particles), actions] = 0
        assert len(set(paths_list)) == self.n_cities, "All cities must be visited."
        paths = torch.stack(paths_list)  # shape: (n_cities, n_particles)
        paths = paths.T  # shape: (n_particles, n_cities)
        
        if return_log_probs:
            log_probs = torch.stack(log_probs_list)  # shape: (n_cities, n_particles)
            return paths, log_probs
        return paths
    
    def decode_solutions_eval(self):
        # population: shape (n_particles, n_cities)
        paths = torch.argsort(self.population, dim=1)  # shape: (n_particles, n_cities)
        return paths

    def update_metadata(self, costs: torch.Tensor):
        # Update personal best
        better_pbest_mask = costs < self.val_pbest
        self.val_pbest[better_pbest_mask] = costs[better_pbest_mask]
        self.pbest[better_pbest_mask] = self.population[better_pbest_mask]

        # Update global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost < self.val_gbest:
            self.val_gbest = min_cost
            self.gbest = self.population[min_idx]
        return costs