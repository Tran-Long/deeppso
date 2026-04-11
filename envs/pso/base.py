from enum import Enum

import torch

from ..problems import BaseProblem


class RewardMode(str, Enum):
    GREEDY = "greedy"
    STOCHASTIC = "stochastic"
    PBEST = "pbest"
    GBEST = "gbest"
    DELTA_GBEST = "delta_gbest"  # clipped to 0
    DELTA_GBEST_RAW = "delta_gbest_raw"  # not clipped, can be negative
    DELTA_PBEST = "delta_pbest"  # clipped to 0
    DELTA_PBEST_RAW = "delta_pbest_raw"  # not clipped, can be negative
    DELTA_PG = "delta_pg"  # weighted delta_gbest + delta_pbest


class BaseEnv:
    def __init__(
        self,
        n_particles,
        batch_problem: BaseProblem,
        use_local_search: bool = False,
        auto_reset: bool = True,
        patience: int = 5,
        do_normalize: bool = False,
        **kwargs
    ):
        self.n_particles = n_particles
        self.problem = batch_problem
        self.batch_size = self.problem.batch_size
        self.initialized = False
        self.use_local_search = use_local_search
        self.auto_reset = auto_reset  # whether to automatically reset done instances in the batch, should be True for training, False for validation/test
        self.patience = patience
        self.device = (
            "cpu"  # default device, will be updated in training/validation/test loops
        )
        self.cnt_patience = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )  # counter for how many consecutive steps without improvement for each instance in the batch
        self.do_normalize = do_normalize

    def to(self, device):
        self.device = device
        self.problem = self.problem.to(device)
        self.cnt_patience = self.cnt_patience.to(device)
        if self.initialized:
            self.population = self.population.to(device)
            self.velocity = self.velocity.to(device)
            self.pbest = self.pbest.to(device)
            self.gbest = self.gbest.to(device)
            self.val_pbest = self.val_pbest.to(device)
            self.val_gbest = self.val_gbest.to(device)
            self.val_gbest_ls = self.val_gbest_ls.to(device)
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
        # costs shape (batch_size, n_particles)
        solutions, costs = self.decode_solutions()
        solutions_ls = self.problem.local_search(solutions)
        costs_ls = self.problem.evaluate(solutions_ls)
        mean_costs = costs  # avg cost of each particle, by default set to equal to the best cost.
        return solutions, costs, solutions_ls, costs_ls, mean_costs

    def normalize(self):
        self.population = self.population / self.population.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) # normalize to [-1, 1]

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
        self.cnt_patience = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.val_pbest = torch.full(
            (self.batch_size, self.n_particles), float("inf"), device=self.device
        )
        self.val_gbest = torch.full(
            (self.batch_size,), float("inf"), device=self.device
        )
        self.val_gbest_ls = torch.full(
            (self.batch_size,), float("inf"), device=self.device
        )
        _, initial_costs, _, initial_costs_ls, _ = self.decode_solutions_eval()
        self.update_metadata(initial_costs, initial_costs_ls)
        return (
            self.population,
            self.velocity,
            self.pbest,
            self.gbest,
            self.problem,
        ), {}

    def _auto_reset_done(self, done_mask: torch.Tensor, **kwargs):
        """Reset only the instances where done_mask is True."""
        if not done_mask.any():
            return  # no instance is done, no need to reset
        new_population, new_velocity, new_pbest, new_gbest = self.initialize_population(
            **kwargs
        )
        self.population[done_mask] = new_population[done_mask]
        self.velocity[done_mask] = new_velocity[done_mask]
        self.pbest[done_mask] = new_pbest[done_mask]
        self.gbest[done_mask] = new_gbest[done_mask]
        self.val_pbest[done_mask] = float("inf")
        self.val_gbest[done_mask] = float("inf")
        self.val_gbest_ls[done_mask] = float("inf")

        _, costs, _, costs_ls, _ = self.decode_solutions_eval()
        self.update_metadata(costs, costs_ls)
        self.cnt_patience[done_mask] = (
            0  # reset patience counter for the reset instances
        )

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
            terminated (Tensor): True for particles that have reached a terminal state, shape (n_particles,)
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

    def update_metadata(
        self, costs: torch.Tensor, costs_ls: torch.Tensor | None = None
    ):
        # cost shape (batch_size, n_particles), each row represents the costs of the n_particles in each problem instance
        # return delta_gbest and delta_pbest for potential use in reward shaping, if needed
        # positive delta means improvement (cost reduction), negative delta means deterioration (cost increase)

        used_costs = (
            costs_ls if self.use_local_search and costs_ls is not None else costs
        )
        if self.do_normalize:
            self.normalize()

        self.population = (
            self.population.detach().clone()
        )  # shape (batch_size, n_particles, dim)
        self.velocity = (
            self.velocity.detach().clone()
        )  # shape (batch_size, n_particles, dim)
        self.pbest = self.pbest.detach().clone()  # shape (batch_size, n_particles, dim)
        self.gbest = self.gbest.detach().clone()  # shape (batch_size, dim)
        better_pbest_mask = (
            used_costs < self.val_pbest
        )  # shape (batch_size, n_particles)
        old_val_pbest = self.val_pbest.clone()
        self.val_pbest[better_pbest_mask] = used_costs[better_pbest_mask]
        delta_val_pbest = (
            old_val_pbest - self.val_pbest
        )  # shape (batch_size, n_particles)
        self.pbest[better_pbest_mask] = (
            self.population[better_pbest_mask].detach().clone()
        )  # shape (batch_size, n_particles, dim)

        # Update global best
        min_cost, min_idx = torch.min(
            used_costs, dim=1
        )  # shape (batch_size,), shape (batch_size,)
        better_gbest_mask = min_cost < self.val_gbest  # shape (batch_size,)
        old_val_gbest = self.val_gbest.clone()
        self.val_gbest[better_gbest_mask] = min_cost[better_gbest_mask]
        delta_val_gbest = old_val_gbest - self.val_gbest  # shape (batch_size,)
        self.gbest[better_gbest_mask] = (
            self.population[better_gbest_mask, min_idx[better_gbest_mask]]
            .detach()
            .clone()
        )  # shape (batch_size, dim)

        # Update local search global best
        min_cost_ls, min_idx_ls = torch.min(costs_ls, dim=1)
        better_gbest_ls_mask = min_cost_ls < self.val_gbest_ls
        self.val_gbest_ls[better_gbest_ls_mask] = min_cost_ls[better_gbest_ls_mask]
        return delta_val_pbest, delta_val_gbest

    @classmethod
    def batching_observations(
        cls,
        populations: list | torch.Tensor,  # list of shape (B, P, D)
        velocitys: list | torch.Tensor,  # list of shape (B, P, D)
        pbests: list | torch.Tensor,  # list of shape (B, P, D)
        gbests: list | torch.Tensor,  # list of shape (B, D)
        problems: list[BaseProblem] = None,
        problem_embeddings: list | torch.Tensor | None = None,
        problem_cls: type[BaseProblem] = None,
    ):
        """Batching the PSO state for input to the agent. Default: just return the raw state tensors. Subclasses can override for different behavior."""

        batched_population = (
            torch.concat(populations, dim=0)
            if isinstance(populations, list)
            else populations
        )
        batched_velocity = (
            torch.concat(velocitys, dim=0) if isinstance(velocitys, list) else velocitys
        )
        batched_pbest = (
            torch.concat(pbests, dim=0) if isinstance(pbests, list) else pbests
        )
        batched_gbest = (
            torch.concat(gbests, dim=0) if isinstance(gbests, list) else gbests
        )

        if problem_embeddings is not None:
            problem_embeddings = (
                torch.concat(problem_embeddings, dim=0)
                if isinstance(problem_embeddings, list)
                else problem_embeddings
            )
            return (
                batched_population,
                batched_velocity,
                batched_pbest,
                batched_gbest,
                None,
                problem_embeddings,
            )
        else:
            batched_problem = (
                problem_cls.batch_instances(*problems)
                if problem_cls is not None
                else None
            )
            return (
                batched_population,
                batched_velocity,
                batched_pbest,
                batched_gbest,
                batched_problem,
                problem_embeddings,
            )
            batched_problem = (
                problem_cls.batch_instances(*problems)
                if problem_cls is not None
                else None
            )
            return (
                batched_population,
                batched_velocity,
                batched_pbest,
                batched_gbest,
                batched_problem,
                problem_embeddings,
            )
