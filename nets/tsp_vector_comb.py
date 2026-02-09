from copy import deepcopy

import torch
import torch_geometric.nn as gnn
from torch import nn
from torch.nn import functional as F

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x.mean(dim=0)

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act_fn='silu'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act_fn = getattr(F, act_fn)
    def forward(self, x, last=False):
        x = self.conv(x)
        x = self.bn(x)
        if not last:
            x = self.act_fn(x)
        return x

class TSPVectorCNN1DModel(nn.Module):
    def __init__(self, n_particles=128, emb_dim=32, act_fn='silu'):
        super().__init__()
        self.n_particles = n_particles
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.conv1 = Conv1dBlock(n_particles, n_particles//2, kernel_size=3, padding=1, act_fn=act_fn)
        self.conv2 = Conv1dBlock(n_particles//2, n_particles*2, kernel_size=3, padding=1, act_fn=act_fn)
        self.conv3 = Conv1dBlock(n_particles*2, n_particles, kernel_size=3, padding=1, act_fn=act_fn)
        self.avg_global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1, emb_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: torch tensor with shape (n_particles, n_cities)
        Returns:
            logits: torch tensor with shape (n_particles, 3), logits for (w, c1, c2)
        '''
        x = x.view(1, x.size(0), x.size(1))  # shape: (1, n_particles, n_cities)
        o = self.avgpool(self.conv1(x))
        o = self.avgpool(self.conv2(o))
        o = self.conv3(o)
        o = self.avg_global_pool(o)
        return self.fc(o).squeeze()

class TSPVectorPopGraph(nn.Module):
    def __init__(self, embed_dim=32, n_gcns=3, act_fn='silu', k_sparse=None, loop=False):
        super().__init__()
        self.stem = Conv1dBlock(1, embed_dim, kernel_size=1, padding=0, act_fn=act_fn)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gcns = nn.ModuleList([gnn.GCNConv(embed_dim, embed_dim) for _ in range(n_gcns)])
        self.out_gcn = gnn.GCNConv(embed_dim, 3)
        self.act_fn = getattr(F, act_fn)
        self.k_sparse = k_sparse
        self.loop = loop
    
    def forward(self, pop):
        x = pop.unsqueeze(1)  # (n_particles, 1, n_cities)
        x = self.stem(x)     # (n_particles, embed_dim, n_cities)
        x = self.pool(x).squeeze(-1)  # (n_particles, embed_dim)

        n_particles = x.size(0)
        edge_index = gnn.knn_graph(x, k=self.k_sparse if self.k_sparse is not None else n_particles, loop=self.loop)
        
        for gcn in self.gcns:
            x = gcn(x, edge_index)
            x = self.act_fn(x)
        
        x = self.out_gcn(x, edge_index)  # (n_particles, 3)
        x = F.softmax(x, dim=-1)
        return x



class TSPVectorGraphCombined(nn.Module):
    def __init__(self, n_particles=50, embed_dim=32, act_fn='silu'):
        super().__init__()
        self.prob_emb = EmbNet(units=embed_dim)
        self.pop_emb = TSPVectorCNN1DModel(n_particles=n_particles, emb_dim=embed_dim)
        self.act_fn = getattr(F, act_fn)
        self.combined_lin = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, pyg, pop):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        prob_embed = self.prob_emb(x, edge_index, edge_attr)
        pop_embed = self.pop_emb(pop)
        combined = pop_embed + prob_embed
        return self.combined_lin(combined)
