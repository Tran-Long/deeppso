import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act_fn='silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_fn = getattr(F, act_fn)
    def forward(self, x, last=False):
        x = self.conv(x)
        x = self.bn(x)
        if not last:
            x = self.act_fn(x)
        return x

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn='silu', end_act_fn=None):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.end_act_fn = getattr(F, end_act_fn) if end_act_fn is not None else None
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                if self.end_act_fn is not None:
                    x = self.end_act_fn(x)
        return x

class TSPEmbGNN(nn.Module):
    def __init__(self, emb_dim=32, depth=12, feats=2, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.emb_dim = emb_dim
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.emb_dim)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.emb_dim) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.emb_dim)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.emb_dim) for i in range(self.depth)])
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
        return w

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

class ParticleVectorCNN1DModel(nn.Module):
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

class ParticleVectorStem(nn.Module):
    def __init__(self, emb_dim=32, input_dim=3, n_layers=3, act_fn='silu'):
        super().__init__()
        self.stem = MLP(
            units_list=[input_dim] + [emb_dim] * n_layers,
            act_fn=act_fn,
            end_act_fn=None
        )
    
    def forward(self, *args) -> torch.Tensor:
        '''
        Args:
            pos, vel, pbest: torch tensors each with shape (n_particles, dim)
        Returns:
            embeddings: torch tensor with shape (n_particles, emb_dim)
        '''
        x = torch.stack(args, dim=-1)  # shape: (n_particles, dim, 3)
        x = self.stem(x)  # shape: (n_particles, dim, emb_dim)
        x = x.mean(dim=1)  # shape: (n_particles, emb_dim)
        return x
    
class SwarmEncoder(nn.Module):
    def __init__(self, emb_dim=32, n_layers=2, act_fn='silu'):
        super().__init__()
        # self.shared_stem = ParticleVectorStem(emb_dim=emb_dim, n_layers=n_layers, act_fn=act_fn) # Your city-agnostic stem
        self.pop_stem = ParticleVectorStem(emb_dim=emb_dim, n_layers=n_layers, act_fn=act_fn) # Your city-agnostic stem
        self.gbest_stem = ParticleVectorStem(emb_dim=emb_dim, input_dim=1, n_layers=1, act_fn=act_fn) # Your city-agnostic stem
        
        # Two unique vectors to tell the network who is who
        # self.particle_type_embed = nn.Parameter(torch.randn(1, emb_dim))
        # self.gbest_type_embed = nn.Parameter(torch.randn(1, emb_dim))

    def forward(self, pos, vel, pbest, gbest):        
        # h_particles = self.shared_stem(pos, vel, pbest) # (n_particles, D)
        # h_gbest = self.shared_stem(
        #     gbest.unsqueeze(0),
        #     torch.zeros_like(gbest).unsqueeze(0), 
        #     gbest.unsqueeze(0)
        # ) # (1, D)
        
        # # Add role-specific "Identity"
        # h_particles = h_particles + self.particle_type_embed
        # h_gbest = h_gbest + self.gbest_type_embed
        h_particles = self.pop_stem(pos, vel, pbest) # (n_particles, D)
        h_gbest = self.gbest_stem(
            gbest.unsqueeze(0),
        ) # (1, D)

        return h_particles, h_gbest