import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import TSPEnvVectorEdge, TSPProblem

from .nets import *


class TSPActorNet(nn.Module):
    """
    Per-edge hyperparameter prediction for PSO-TSP.
    Architecture:
        1. TSPEmbGNN  → edge_embs: (dim, emb_dim)   — per-edge graph identity
        2. SwarmEncoder → particle_ctx: (n_particles, emb_dim), gbest_emb: (1, emb_dim)
        3. CrossAttention: each particle query attends to [edge_embs, gbest_emb, swarm_mean]
           → particle_ctx: (n_particles, emb_dim)  — particle "strategy"
        4. Per-edge MLP: for each (particle, edge) pair, concatenate
           [pos_e, vel_e, pbest_e, gbest_e,  edge_emb_e,  particle_ctx_p]
           → shared MLP → mu(w,c1,c2), sigma(w,c1,c2)
        Output: (n_particles, dim, 3)
    """

    def __init__(
        self,
        emb_dim=32,
        act_fn="silu",
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # --- Encoders ---
        self.swarm_emb = SwarmEncoder(emb_dim=emb_dim, act_fn=act_fn)
        self.tsp_emb = TSPEmbGNN(emb_dim=emb_dim, depth=2, act_fn=act_fn)

        # --- Cross-attention: particle queries attend to (edge_embs + gbest + swarm_mean) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=4, batch_first=True
        )
        self.layer_norm_attn = nn.LayerNorm(emb_dim)

        # --- Type embeddings so attention can distinguish token roles ---
        self.edge_type_emb = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.gbest_type_emb = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.swarm_type_emb = nn.Parameter(torch.randn(1, emb_dim) * 0.02)

        # --- Per-edge MLP heads ---
        # Input: [pos_e, vel_e, pbest_e, gbest_e] (4) + edge_emb (emb_dim) + particle_ctx (emb_dim)
        edge_input_dim = 4 + 2 * emb_dim
        self.edge_head_mu = MLP(
            units_list=[edge_input_dim, emb_dim, emb_dim, 3],
            act_fn=act_fn,
        )
        self.edge_head_sigma = MLP(
            units_list=[edge_input_dim, emb_dim, emb_dim, 3],
            act_fn=act_fn,
        )
        self.sigmoid = nn.Sigmoid()
        self.device = next(self.parameters()).device

    def get_problem_embedding(self, tsp_problem: TSPProblem):
        """Use to cache the TSP embedding since it doesn't change during PSO iterations."""

        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations, so there is no need
        # to run the GNN on every iteration — this is ~n_iter speedup for the GNN.
        # Pre-compute TSP graph embedding once per problem instance.
        # Computed WITH gradients so the GNN receives gradient signal.
        # Its graph is freed in the single backward call below, not per-iteration.

        pyg_data = tsp_problem.pyg_data.to(self.device)
        tsp_embedding = self.tsp_emb(
            pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr
        )  # (dim, emb_dim)
        return tsp_embedding

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        pbest: torch.Tensor,
        gbest: torch.Tensor,
        problem: TSPProblem,
        problem_embedding: torch.Tensor = None,
    ):
        """
        Args:
            pos, vel, pbest: (n_particles, dim)
            gbest: (dim,)
            problem: TSPProblem object (used when problem_embedding is None)
            problem_embedding: pre-computed (dim, emb_dim) — full per-edge embeddings
        Returns:
            wc1c2_mu:    (n_particles, dim, 3)
            wc1c2_sigma: (n_particles, dim, 3)
        """
        n_particles, dim = pos.shape

        # --- 1. Particle & gbest embeddings (global summaries) ---
        particle_embeddings, gbest_embedding = self.swarm_emb(
            pos, vel, pbest, gbest, k_sparse=problem.k_sparse
        )  # (n_particles, D), (1, D)

        # --- 2. Per-edge TSP embedding ---
        if problem_embedding is None:
            tsp_embedding = self.get_problem_embedding(problem)

        # --- 3. Cross-attention ---
        # Context: [edge_embs(dim, D), gbest(1, D), swarm_mean(1, D)]
        swarm_mean = particle_embeddings.mean(dim=0, keepdim=True)  # (1, D)
        context = torch.cat(
            [
                tsp_embedding + self.edge_type_emb,  # (dim, D)
                gbest_embedding + self.gbest_type_emb,  # (1, D)
                swarm_mean + self.swarm_type_emb,  # (1, D)
            ],
            dim=0,
        )  # (dim+2, D)

        attn_output, _ = self.cross_attention(
            query=particle_embeddings.unsqueeze(0),  # (1, n_particles, D)
            key=context.unsqueeze(0),  # (1, dim+2, D)
            value=context.unsqueeze(0),  # (1, dim+2, D)
        )  # (1, n_particles, D)
        attn_output = attn_output.squeeze(0)  # (n_particles, D)
        particle_ctx = self.layer_norm_attn(
            attn_output + particle_embeddings
        )  # (n_particles, D)

        # --- 4. Per-edge feature assembly ---
        # Local PSO scalars per edge: (n_particles, dim, 4)
        local_feats = torch.stack(
            [pos, vel, pbest, gbest.unsqueeze(0).expand_as(pos)], dim=-1
        )
        # (n_particles, dim, 4)

        # Broadcast embeddings to every edge:
        # tsp_embedding: (dim, D) → (n_particles, dim, D)
        edge_emb_expanded = tsp_embedding.unsqueeze(0).expand(n_particles, -1, -1)
        # particle_ctx: (n_particles, D) → (n_particles, dim, D)
        ctx_expanded = particle_ctx.unsqueeze(1).expand(-1, dim, -1)

        # Concatenate: (n_particles, dim, 4 + 2*D)
        edge_input = torch.cat([local_feats, edge_emb_expanded, ctx_expanded], dim=-1)

        # --- 5. Shared MLP → per-edge (w, c1, c2) ---
        raw_mu = self.edge_head_mu(edge_input)  # (n_particles, dim, 3)
        raw_sigma = self.edge_head_sigma(edge_input)  # (n_particles, dim, 3)

        # Bounded outputs
        w_mu = 0.4 + 0.5 * self.sigmoid(raw_mu[..., 0:1])  # [0.4, 0.9]
        c1_mu = 1.0 + 2.0 * self.sigmoid(raw_mu[..., 1:2])  # [1.0, 3.0]
        c2_mu = 1.0 + 2.0 * self.sigmoid(raw_mu[..., 2:3])  # [1.0, 3.0]
        w_sig = 0.1 + 0.3 * self.sigmoid(raw_sigma[..., 0:1])  # [0.1, 0.4]
        c1_sig = 0.2 + 0.8 * self.sigmoid(raw_sigma[..., 1:2])  # [0.2, 1.0]
        c2_sig = 0.2 + 0.8 * self.sigmoid(raw_sigma[..., 2:3])  # [0.2, 1.0]

        wc1c2_mu = torch.cat([w_mu, c1_mu, c2_mu], dim=-1)  # (n_particles, dim, 3)
        wc1c2_sigma = torch.cat(
            [w_sig, c1_sig, c2_sig], dim=-1
        )  # (n_particles, dim, 3)
        return wc1c2_mu, wc1c2_sigma
