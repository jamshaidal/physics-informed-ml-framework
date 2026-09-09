"""
Neural Network Architectures for Physics-Informed Neural Networks (PINNs)
Implements standard MLPs, modified MLP with Fourier feature embeddings,
and Sinusoidal Representation Networks (SIREN) for high-frequency physics problems.
"""

import torch
import torch.nn as nn
import numpy as np

class PINNNetwork(nn.Module):
    """
    Standard Fully-Connected Architecture for PINNs with Xavier/Glorot initialization
    and smooth activation functions (Tanh, GELU, Sin) suitable for higher-order derivatives.
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, activation="tanh"):
        super(PINNNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if activation.lower() == "tanh":
            act_fn = nn.Tanh
        elif activation.lower() == "gelu":
            act_fn = nn.GELU
        elif activation.lower() == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.Tanh
            
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        return self.net(x)

class FourierFeaturePINN(nn.Module):
    """
    PINN with Random Fourier Features to overcome spectral bias for high-frequency PDEs.
    Maps x -> [cos(2*pi*B*x), sin(2*pi*B*x)]
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, num_frequencies=32, scale=2.0):
        super(FourierFeaturePINN, self).__init__()
        self.B = nn.Parameter(torch.randn(in_dim, num_frequencies) * scale, requires_grad=False)
        fourier_dim = num_frequencies * 2
        
        layers = []
        layers.append(nn.Linear(fourier_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Fourier projection
        proj = 2.0 * np.pi * torch.matmul(x, self.B)
        features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.net(features)
