"""
Damped Harmonic Oscillator PINN Solver
ODE: m * x_tt + mu * x_t + k * x = 0
Domain: t in [0, 10]
IC: x(0) = 1.0, x_t(0) = 0.0
Includes exact analytical solution for rigorous L2 error benchmarking.
"""

import torch
import numpy as np
from core.pinn import BasePINNSolver, compute_gradient

class HarmonicOscillatorPINN(BasePINNSolver):
    def __init__(self, network, m=1.0, mu=0.5, k=4.0, device="cpu"):
        super(HarmonicOscillatorPINN, self).__init__(network, device)
        self.m = m
        self.mu = mu
        self.k = k
        
        # Analytical properties (Underdamped case: mu^2 < 4*m*k)
        self.delta = self.mu / (2.0 * self.m)
        self.omega = np.sqrt(self.k / self.m - self.delta**2)
        
    def analytical_solution(self, t):
        """Exact analytical solution: x(t) = exp(-delta * t) * (cos(omega * t) + (delta/omega) * sin(omega * t))"""
        return np.exp(-self.delta * t) * (np.cos(self.omega * t) + (self.delta / self.omega) * np.sin(self.omega * t))
        
    def ode_residual(self, t):
        t.requires_grad_(True)
        x = self.network(t)
        x_t = compute_gradient(x, t)
        x_tt = compute_gradient(x_t, t)
        
        residual = self.m * x_tt + self.mu * x_t + self.k * x
        return residual
        
    def generate_points(self, n_colloc=500):
        t_colloc = torch.FloatTensor(n_colloc, 1).uniform_(0.0, 10.0)
        t_ic = torch.zeros(1, 1)
        x_ic = torch.tensor([[1.0]], dtype=torch.float32)
        return {
            "t_colloc": t_colloc.to(self.device),
            "t_ic": t_ic.to(self.device),
            "x_ic": x_ic.to(self.device)
        }
        
    def compute_loss(self, data, lambda_ode=1.0, lambda_ic=20.0):
        # 1. ODE residual
        res = self.ode_residual(data["t_colloc"])
        loss_ode = torch.mean(res**2)
        
        # 2. IC position loss: x(0) = 1
        t_ic = data["t_ic"].clone().requires_grad_(True)
        x_pred_ic = self.network(t_ic)
        loss_ic_pos = (x_pred_ic - data["x_ic"])**2
        
        # 3. IC velocity loss: x_t(0) = 0
        x_t_ic = compute_gradient(x_pred_ic, t_ic)
        loss_ic_vel = (x_t_ic - 0.0)**2
        
        loss_ic = torch.mean(loss_ic_pos + loss_ic_vel)
        total_loss = lambda_ode * loss_ode + lambda_ic * loss_ic
        return total_loss, {"ode": loss_ode.item(), "ic": loss_ic.item()}
