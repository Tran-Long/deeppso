import torch

from ..problems import BaseProblem


class BaseEnvPSOProblem:
    def __init__(self, n_particles, problem: BaseProblem, device="cpu", **kwargs):
        self.n_particles = n_particles
        self.problem = problem
        self.device = device

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

        Returns:
            initial_observations (tuple[Tensor, Tensor, Tensor, Tensor, TSPProblem]): (population, velocity, pbest, gbest, problem)
            info (dict): optional dict for extra info
        """

        self.population, self.velocity, self.pbest, self.gbest = (
            self.initialize_population(**kwargs)
        )
        self.val_pbest = torch.full(
            (self.n_particles,), float("inf"), device=self.device
        )
        self.val_gbest = torch.tensor(float("inf"), device=self.device)
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
