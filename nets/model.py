import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TSPVectorCNNModel(nn.Module):
    def __init__(self, hidden_channels=16, depth=3):
        super().__init__()
        self.init_block = ConvBlock(1, hidden_channels, kernel_size=3, padding="same")
        self.conv_modules = nn.ModuleList()
        for i in range(depth):
            in_channels = hidden_channels if i == 0 else hidden_channels + (i - 1) * 2
            out_channels = hidden_channels + i * 2
            self.conv_modules.append(ConvBlock(in_channels, out_channels, kernel_size=3, padding="same"))
        self.lin = nn.Linear(1, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: torch tensor with shape (n_particles, n_cities)
        Returns:
            logits: torch tensor with shape (n_particles, 3), logits for (w, c1, c2)
        '''
        x = x.view(1, 1, x.size(0), x.size(1))  
        o = self.init_block(x)
        for i, module in enumerate(self.conv_modules):
            o = module(o, last=(i == len(self.conv_modules) - 1))
        o = o.mean(dim=[1, 3], keepdim=True)  # shape: (1, 1, n_particles, 1)
        o = self.lin(o).squeeze()  # shape: (n_particles, 3)
        return self.softmax(o)