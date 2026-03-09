import torch

from ..problems import BaseProblem

class BaseEnvPSOBatchProblem:
    def __init__(self, n_particles, batch_problem: BaseProblem, **kwargs):
        self.n_particles = n_particles
        self.problem = batch_problem
        self.batch_size = self.problem.batch_size
        self.initialized = False
        self.device = "cpu" # default device, will be updated in training/validation/test loops

    def to(self, device):
        self.device = device
        self.problem = self.problem.to(device)
        if self.initialized:
            self.population = self.population.to(device)
            self.velocity = self.velocity.to(device)
            self.pbest = self.pbest.to(device)
            self.gbest = self.gbest.to(device)
            self.val_pbest = self.val_pbest.to(device)
            self.val_gbest = self.val_gbest.to(device)
        return self

    def initialize_population(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Must return the PSO properties

        Raises:
            NotImplementedError: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - population: shape (n_particles, problem-specific solution representation)
                - velocity: shape (n_particles, problem-specific solution representation)
                - pbest: shape (n_particles, problem-specific solution representation)
                - gbest: shape (problem-specific solution representation,)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode_solutions(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode_solutions_eval(self):
        """Default: single decode + evaluate. Subclasses should override."""
        solutions, costs = self.decode_solutions()
        return solutions, costs

    def reset(self, **kwargs):
        """
        Compatible to the OpenAI Gym interface

        Initialize the PSO population, evaluate and return the initial observation.
        
        Args:
            device: the torch device to initialize tensors on
            **kwargs: any additional info needed for initialization, problem-specific, ...

        Returns:
            initial_observations (tuple[Tensor, Tensor, Tensor, Tensor, TSPProblem]): (population, velocity, pbest, gbest, problem)
            info (dict): optional dict for extra info
        """

        self.population, self.velocity, self.pbest, self.gbest = (
            self.initialize_population(**kwargs)
        )
        self.initialized = True
        self.val_pbest = torch.full(
            (self.batch_size, self.n_particles), float("inf"), device=self.device
        )
        self.val_gbest = torch.full(
            (self.batch_size,), float("inf"), device=self.device
        )
        _, initial_costs = self.decode_solutions_eval()
        self.update_metadata(initial_costs)
        return (
            self.population,
            self.velocity,
            self.pbest,
            self.gbest,
            self.problem,
        ), {}

    def step_train(self, wc1c2: torch.Tensor, **kwargs) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, BaseProblem],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        dict,
    ]:
        """
        Compatible to the OpenAI Gym interface

        Args:
            wc1c2 (Tensor): "actions" - shape (n_particles, dim, 3) — per-edge hyperparameters
            **kwargs: any additional info needed for stepping, e.g., pre-computed problem embedding
        Returns:
            observations (tuple[Tensor, Tensor, Tensor, Tensor, TSPProblem]): (population, velocity, pbest, gbest, problem) — the new state after stepping with wc1c2
            reward (Tensor): evaluation value (cost, **maximize is better**) for each particle after the step, shape (n_particles,)
            terminated (Tensor): False for all particles (PSO doesn't have terminal states), shape (n_particles,) or maybe None
            truncated (Tensor): False for all particles (PSO doesn't have truncated states), shape (n_particles,) or maybe None
            info (dict): optional dict for extra info
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step_eval(self, wc1c2: torch.Tensor, **kwargs) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, BaseProblem],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        dict,
    ]:
        """Default: same stepping for training and evaluation. Subclasses can override for different behavior."""
        return self.step_train(wc1c2, **kwargs)

    def update_metadata(self, costs: torch.Tensor):
        # cost shape (batch_size, n_particles), each row represents the costs of the n_particles in each problem instance
        
        self.population = self.population.detach().clone() # shape (batch_size, n_particles, dim)
        self.velocity = self.velocity.detach().clone() # shape (batch_size, n_particles, dim)
        self.pbest = self.pbest.detach().clone() # shape (batch_size, n_particles, dim)
        self.gbest = self.gbest.detach().clone() # shape (batch_size, dim)
        better_pbest_mask = costs < self.val_pbest # shape (batch_size, n_particles)
        self.val_pbest[better_pbest_mask] = costs[better_pbest_mask]

        self.pbest[better_pbest_mask] = self.population[better_pbest_mask].detach().clone() # shape (batch_size, n_particles, dim))

        # Update global best
        min_cost, min_idx = torch.min(costs, dim=1) # shape (batch_size,), shape (batch_size,)
        better_gbest_mask = min_cost < self.val_gbest # shape (batch_size,)
        self.val_gbest[better_gbest_mask] = min_cost[better_gbest_mask]
        self.gbest[better_gbest_mask] = self.population[better_gbest_mask, min_idx[better_gbest_mask]].detach().clone() # shape (batch_size, dim)
        return costs

    def evaluate(self, solutions: torch.Tensor):
        """Evaluate the solutions for all problem instances in the batch.

        Args:
            solutions: shape (batch_size, n_particles, problem-specific solution representation)
        Returns:
            costs: shape (batch_size, n_particles)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")