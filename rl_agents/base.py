import torch
import torch.nn as nn
from envs import BaseProblem

class BaseAgent(nn.Module):
    def get_action(self, obs) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Get action based on observation.

        Args:
            obs (tuple[Tensor, Tensor, Tensor, Tensor, BaseProblem, Tensor]): (population, velocity, pbest, gbest, problem, problem_embedding)

        Returns:
            wc1c2 (Tensor): shape based on problem-specific solution representation, e.g. (n_particles, dim, 3) for TSP
            log_probs (Tensor | None): log probabilities of the actions, for policy gradient methods
            entropy (Tensor | None): entropy of the action distribution, for policy gradient methods
            
        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError("Subclasses must implement get_action method")
    
    def get_problem_embedding(self, problem: BaseProblem) -> torch.Tensor:
        """For fixed problem embeddings, compute once at the start of each episode and reuse across all iterations."""
        raise NotImplementedError("Subclasses must implement get_problem_embedding method")