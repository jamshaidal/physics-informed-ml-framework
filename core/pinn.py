"""
Core Physics-Informed Neural Network (PINN) Framework
Provides automatic differentiation wrappers and multi-objective loss computation
supporting two-stage optimization (Adam + L-BFGS).
"""

import torch
import torch.nn as nn
import numpy as np

def compute_gradient(y, x, create_graph=True):
    """Computes dy/dx using PyTorch autograd."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True
    )[0]

class BasePINNSolver:
    """
    Abstract base solver for Physics-Informed Neural Networks.
    Subclasses define specific physical residual functions (PDE operators).
    """
    def __init__(self, network, device="cpu"):
        self.network = network.to(device)
        self.device = device
        
    def predict(self, x_coords):
        """Forward evaluation without gradient tracking."""
        self.network.eval()
        with torch.no_grad():
            if not isinstance(x_coords, torch.Tensor):
                x_coords = torch.tensor(x_coords, dtype=torch.float32)
            return self.network(x_coords.to(self.device)).cpu().numpy()
            
    def pde_residual(self, x, t):
        """Must be implemented by subclasses to return the PDE residual tensor."""
        raise NotImplementedError("Subclasses must implement pde_residual(x, t)")
        
    def train_adam(self, train_loader, epochs=2000, lr=1e-3, lambda_pde=1.0, lambda_bc=1.0):
        """Stage 1: Adam optimizer for broad parameter search."""
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_dict = self.compute_total_loss(batch, lambda_pde, lambda_bc)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                history.append(total_loss)
                print(f"[Adam Epoch {epoch+1}/{epochs}] Total Loss: {total_loss:.5e}")
                
        return history
        
    def compute_total_loss(self, batch, lambda_pde=1.0, lambda_bc=1.0):
        """Aggregates data, PDE residual, and boundary condition losses."""
        raise NotImplementedError("Subclasses must implement compute_total_loss")
