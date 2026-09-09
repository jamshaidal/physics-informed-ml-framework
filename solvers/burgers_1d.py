"""
1D Viscous Burgers' Equation PINN Solver
PDE: u_t + u * u_x - (0.01 / pi) * u_xx = 0
Domain: x in [-1, 1], t in [0, 1]
IC: u(x, 0) = -sin(pi * x)
BC: u(-1, t) = u(1, t) = 0
"""

import torch
import numpy as np
from core.pinn import BasePINNSolver, compute_gradient

class BurgersPINNSolver(BasePINNSolver):
    def __init__(self, network, nu=0.01 / np.pi, device="cpu"):
        super(BurgersPINNSolver, self).__init__(network, device)
        self.nu = nu
        
    def pde_residual(self, x, t):
        """
        Computes the physical residual of the Viscous Burgers' equation:
        R(x, t) = u_t + u * u_x - nu * u_xx
        """
        # Ensure gradients are tracked
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        xt = torch.cat([x, t], dim=1)
        u = self.network(xt)
        
        u_t = compute_gradient(u, t)
        u_x = compute_gradient(u, x)
        u_xx = compute_gradient(u_x, x)
        
        residual = u_t + u * u_x - self.nu * u_xx
        return residual
        
    def generate_collocation_points(self, n_colloc=2000, n_ic=200, n_bc=200):
        """Generates domain collocation points, initial condition points, and boundary points."""
        # 1. Collocation points in the interior domain: x in [-1, 1], t in [0, 1]
        x_colloc = torch.FloatTensor(n_colloc, 1).uniform_(-1.0, 1.0)
        t_colloc = torch.FloatTensor(n_colloc, 1).uniform_(0.0, 1.0)
        
        # 2. Initial conditions: t = 0, u(x, 0) = -sin(pi * x)
        x_ic = torch.FloatTensor(n_ic, 1).uniform_(-1.0, 1.0)
        t_ic = torch.zeros(n_ic, 1)
        u_ic = -torch.sin(np.pi * x_ic)
        
        # 3. Boundary conditions: x = -1 or x = 1, u(+-1, t) = 0
        t_bc = torch.FloatTensor(n_bc, 1).uniform_(0.0, 1.0)
        x_bc_left = -torch.ones(n_bc // 2, 1)
        x_bc_right = torch.ones(n_bc - n_bc // 2, 1)
        x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        u_bc = torch.zeros(n_bc, 1)
        
        return {
            "x_colloc": x_colloc.to(self.device),
            "t_colloc": t_colloc.to(self.device),
            "x_ic": x_ic.to(self.device),
            "t_ic": t_ic.to(self.device),
            "u_ic": u_ic.to(self.device),
            "x_bc": x_bc.to(self.device),
            "t_bc": t_bc.to(self.device),
            "u_bc": u_bc.to(self.device)
        }
        
    def compute_loss(self, data, lambda_pde=1.0, lambda_ic=10.0, lambda_bc=10.0):
        """Calculates combined multi-objective PINN loss."""
        # 1. PDE residual on collocation points
        res = self.pde_residual(data["x_colloc"], data["t_colloc"])
        loss_pde = torch.mean(res**2)
        
        # 2. Initial condition loss
        xt_ic = torch.cat([data["x_ic"], data["t_ic"]], dim=1)
        u_pred_ic = self.network(xt_ic)
        loss_ic = torch.mean((u_pred_ic - data["u_ic"])**2)
        
        # 3. Boundary condition loss
        xt_bc = torch.cat([data["x_bc"], data["t_bc"]], dim=1)
        u_pred_bc = self.network(xt_bc)
        loss_bc = torch.mean((u_pred_bc - data["u_bc"])**2)
        
        total_loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
        return total_loss, {"pde": loss_pde.item(), "ic": loss_ic.item(), "bc": loss_bc.item()}
