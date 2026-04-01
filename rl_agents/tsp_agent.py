import torch
from torch_geometric.data import Batch

from envs.problems import TSPBatchProblem

from .base import BaseAgent
from .nets import TSPActorNet, TSPCriticNet, TSPCriticNetPerParticle

class TSPAgent(BaseAgent):
    def __init__(
        self,
        emb_dim=32,
        act_fn="silu",
        n_gnn_feats=2,
        use_k_sparse=True,
        constraint_w_mean=(0.4, 0.9),
        constraint_c_mean=(1.0, 3.0),
        constraint_w_std=(0.1, 0.2),
        constraint_c_std=(0.5, 1.0),
    ):
        super().__init__()
        self.actor = TSPActorNet(emb_dim=emb_dim, act_fn=act_fn, n_gnn_feats=n_gnn_feats, use_k_sparse=use_k_sparse, constraint_w_mean=constraint_w_mean, constraint_c_mean=constraint_c_mean, constraint_w_std=constraint_w_std, constraint_c_std=constraint_c_std)

    def get_problem_embedding(self, tsp_problem: TSPBatchProblem):
        """Use to cache the TSP embedding since it doesn't change during PSO iterations."""

        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations, so there is no need
        # to run the GNN on every iteration — this is ~n_iter speedup for the GNN.
        # Pre-compute TSP graph embedding once per problem instance.
        # Computed WITH gradients so the GNN receives gradient signal.
        # Its graph is freed in the single backward call below, not per-iteration.

        return self.actor.get_problem_embedding(tsp_problem)

    def get_action(self, obs):
        pos, vel, pbest, gbest, problem, problem_embedding = obs
        wc1c2_mu, wc1c2_sigma = self.actor(
            pos, vel, pbest, gbest, problem, problem_embedding
        ) # Shape (batch_size, n_particles, dim, 3)
        wc1c2_dist = torch.distributions.Normal(wc1c2_mu, wc1c2_sigma)
        wc1c2 = wc1c2_dist.sample()
        log_probs = wc1c2_dist.log_prob(wc1c2)
        pl_entropy = wc1c2_dist.entropy()

        # Sum log probs over the 3 hyperparams, mean for each particle to keep scale stable across problem sizes.
        log_probs = log_probs.sum(dim=-1).mean(dim=-1)  # (batch_size, n_particles)
        pl_entropy = pl_entropy.sum(dim=-1).mean(dim=-1)  # (batch_size, n_particles)
        return wc1c2, log_probs, pl_entropy
    

class TSPACAgent(TSPAgent):
    def __init__(
        self,
        emb_dim=32,
        act_fn="silu",
        n_gnn_feats=2,
        use_k_sparse=True,
        constraint_w_mean=(0.4, 0.9),
        constraint_c_mean=(1.0, 3.0),
        constraint_w_std=(0.1, 0.2),
        constraint_c_std=(0.5, 1.0),
    ):
        super().__init__(emb_dim=emb_dim, act_fn=act_fn, n_gnn_feats=n_gnn_feats, use_k_sparse=use_k_sparse, constraint_w_mean=constraint_w_mean, constraint_c_mean=constraint_c_mean, constraint_w_std=constraint_w_std, constraint_c_std=constraint_c_std)
        # self.critic = TSPCriticNet(emb_dim=emb_dim, act_fn=act_fn)
        self.critic = TSPCriticNetPerParticle(emb_dim=emb_dim, act_fn=act_fn)
    
    def get_action_and_value(self, obs):
        pos, vel, pbest, gbest, problem, _ = obs
        action = self.get_action(obs)
        value = self.critic(pos, vel, pbest, gbest, problem)
        return action, value

    def get_value(self, obs):
        pos, vel, pbest, gbest, problem, _ = obs
        value = self.critic(pos, vel, pbest, gbest, problem)
        return value