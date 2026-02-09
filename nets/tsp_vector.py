from torch_geometric.data import Data

from .model import *


class HyperparamTSPModel(nn.Module):
    def __init__(self, n_particles, emb_dim=32, act_fn="silu", softmax_temperature=1.0):
        super().__init__()
        self.n_particles = n_particles
        self.softmax_temperature = softmax_temperature
        self.swarm_emb = SwarmEncoder(emb_dim=emb_dim, act_fn=act_fn)
        self.tsp_emb = TSPEmbGNN(emb_dim=emb_dim, depth=2, act_fn=act_fn)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=4, batch_first=True
        )
        self.fc_norm = nn.Linear(emb_dim, emb_dim)
        self.fc_head = MLP(
            units_list=[emb_dim] * 1 + [3],
            act_fn=act_fn,
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pos, vel, pbest, gbest, tsp_pyg: Data) -> torch.Tensor:
        """
        Args:
            pos, vel, pbest: torch tensors each with shape (n_particles, problem_dim)
            gbest: torch tensor with shape (problem_dim,)
            tsp_pyg: PyG Data object for TSP problem
        Returns:
            wc1c2: torch tensor with shape (n_particles, 3), logits for (w, c1, c2)
        """
        particle_embeddings, gbest_embedding = self.swarm_emb(
            pos, vel, pbest, gbest
        )  # shape: (n_particles, emb_dim) & (1, emb_dim)
        # print(
        #     f"shape: particle_embeddings: {particle_embeddings.shape}, gbest_embedding: {gbest_embedding.shape}"
        # )
        # print(f"particle_embeddings std: {torch.abs(particle_embeddings).std(dim=0)}")
        tsp_embedding = self.tsp_emb(
            tsp_pyg.x, tsp_pyg.edge_index, tsp_pyg.edge_attr
        )  # shape: (n_cities x k_sparse, emb_dim)
        tsp_embedding = tsp_embedding.mean(dim=0, keepdim=True)  # shape: (1, emb_dim)
        context = torch.cat(
            [tsp_embedding, gbest_embedding], dim=0
        )  # shape: (2, emb_dim)
        # print(f"gbest std: {gbest_embedding.std().item():.6e}")
        # print(f"tsp_embedding std: {tsp_embedding.std().item():.6e}")
        # print(
        #     f"context: min={context.min().item():.4f}, max={context.max().item():.4f}, std={context.std().item():.6e}"
        # )
        attn_output, attn_weights = self.cross_attention(
            query=particle_embeddings.unsqueeze(0),
            key=context.unsqueeze(0),
            value=context.unsqueeze(0),
        )  # shape: (1, n_particles, emb_dim)
        # print(
        #     f"attn_weights shape: {attn_weights.shape}, values: {attn_weights[0, :3, :]}"
        # )  # First 3 particles' attention
        attn_output = attn_output.squeeze(0)  # shape: (n_particles, emb_dim)
        # print(
        #     f"attn_output shape: {attn_output.shape}, min={attn_output.min().item():.6e}, max={attn_output.max().item():.6e}"
        # )
        attn_output = self.layer_norm(attn_output + particle_embeddings)  # Normalize before FC
        wc1c2 = self.fc_norm(attn_output)  # shape: (n_particles, 3)
        wc1c2 = self.layer_norm(wc1c2 + attn_output)  # Normalize before softmax
        # print(f"wc1c2 std: {torch.abs(wc1c2).std(dim=0)}")
        # w  = 0.5 + 0.4 * torch.sigmoid(wc1c2[..., 0:1])  # Range [0.5, 0.9]
        # c1 = 1.0 + 2.0 * torch.sigmoid(wc1c2[..., 1:2])  # Range [1.0, 3.0]
        # c2 = 1.0 + 2.0 * torch.sigmoid(wc1c2[..., 2:3])  # Range [1.0, 3.0]
        # return torch.cat([w, c1, c2], dim=-1)

        # Apply temperature scaling to prevent softmax saturation
        wc1c2 = self.fc_head(wc1c2)  # shape: (n_particles, 3)
        out = self.softmax(wc1c2 / self.softmax_temperature)
        # print(f"wc1c2 softmaxed std: {torch.abs(out).std(dim=0)}")
        return out
        # return self.softmax(wc1c2 / self.softmax_temperature)
