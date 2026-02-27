import torch
from torch_geometric.data import Batch

from envs.problems import TSPProblem

from .base import BaseAgent
from .nets import TSPActorNet


class TSPAgent(BaseAgent):
    def __init__(
        self,
        emb_dim=32,
        act_fn="silu",
    ):
        super().__init__()
        self.actor = TSPActorNet(emb_dim=emb_dim, act_fn=act_fn)

    def get_problem_embedding(self, tsp_problem: TSPProblem):
        """Use to cache the TSP embedding since it doesn't change during PSO iterations."""

        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations, so there is no need
        # to run the GNN on every iteration — this is ~n_iter speedup for the GNN.
        # Pre-compute TSP graph embedding once per problem instance.
        # Computed WITH gradients so the GNN receives gradient signal.
        # Its graph is freed in the single backward call below, not per-iteration.

        pyg_data = tsp_problem.pyg_data
        tsp_embedding = self.actor.tsp_emb(
            pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr
        )  # (dim, emb_dim)
        return tsp_embedding

    def get_problem_embedding_batch(self, problems: list[TSPProblem]) -> torch.Tensor:
        """Batch GNN embedding for multiple problems using PyG Batch.

        All problems must have the same n_cities and k_sparse.
        Returns:
            embeddings: (B, dim, emb_dim)
        """
        device = next(self.actor.parameters()).device
        pyg_data_list = [p.pyg_data.to(device) for p in problems]
        batch = Batch.from_data_list(pyg_data_list)
        all_edge_emb = self.actor.tsp_emb(
            batch.x, batch.edge_index, batch.edge_attr
        )  # (total_edges, emb_dim)
        B = len(problems)
        dim = problems[0].n_cities * problems[0].k_sparse
        return all_edge_emb.view(B, dim, -1)  # (B, dim, emb_dim)

    def get_action(self, obs):
        pos, vel, pbest, gbest, problem, problem_embedding = obs
        wc1c2_mu, wc1c2_sigma = self.actor(
            pos, vel, pbest, gbest, problem, problem_embedding
        )
        wc1c2_dist = torch.distributions.Normal(wc1c2_mu, wc1c2_sigma)
        wc1c2 = wc1c2_dist.sample()
        log_probs = wc1c2_dist.log_prob(wc1c2)
        pl_entropy = wc1c2_dist.entropy()

        # Sum log probs over the 3 hyperparams, mean for each particle to keep scale stable across problem sizes.
        # Shape: (n_particles,)
        log_probs = log_probs.sum(dim=-1).mean(dim=-1)
        # Entropy bonus (mean over dim to keep scale stable)
        entropy = pl_entropy.sum(dim=-1).mean()
        return wc1c2, log_probs, entropy

    def get_action_batch(
        self,
        batch_pos: torch.Tensor,
        batch_vel: torch.Tensor,
        batch_pbest: torch.Tensor,
        batch_gbest: torch.Tensor,
        k_sparse: int,
        batch_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Batched action sampling for parallel validation (no log_probs / entropy).

        Args:
            batch_pos, batch_vel, batch_pbest: (B, n_particles, dim)
            batch_gbest: (B, dim)
            k_sparse: int (same for all envs)
            batch_embedding: (B, dim, emb_dim)
        Returns:
            wc1c2: (B, n_particles, dim, 3)
        """
        wc1c2_mu, wc1c2_sigma = self.actor.forward_batch(
            batch_pos,
            batch_vel,
            batch_pbest,
            batch_gbest,
            k_sparse,
            batch_embedding,
        )
        wc1c2_dist = torch.distributions.Normal(wc1c2_mu, wc1c2_sigma)
        return wc1c2_dist.sample()
