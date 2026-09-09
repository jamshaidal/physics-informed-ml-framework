"""
Master PINN Training & Benchmarking CLI
Supports 1D Burgers' Equation and Damped Harmonic Oscillator.
Evaluates training convergence, residual norms, and generates publication plots.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.network import PINNNetwork
from solvers.harmonic_oscillator import HarmonicOscillatorPINN
from solvers.burgers_1d import BurgersPINNSolver

def train_harmonic(epochs=1500, lr=1e-3, out_dir="."):
    print("=================================================================")
    print("  TRAINING PINN: DAMPED HARMONIC OSCILLATOR")
    print("=================================================================")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1 input (t), 1 output (x)
    net = PINNNetwork(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, activation="tanh")
    solver = HarmonicOscillatorPINN(net)
    data = solver.generate_points(n_colloc=600)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_dict = solver.compute_loss(data)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.5e} (ODE: {loss_dict['ode']:.5e}, IC: {loss_dict['ic']:.5e})")
            
    # Evaluate against analytical solution
    t_test = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    x_pred = solver.predict(t_test)
    x_exact = solver.analytical_solution(t_test)
    
    l2_error = np.linalg.norm(x_pred - x_exact) / np.linalg.norm(x_exact)
    print(f"\n[BENCHMARK] Relative L2 Error vs Analytical Solution: {l2_error:.4e}")
    
    # Plot comparison
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(t_test, x_exact, label="Analytical Solution", color="black", lw=2)
    plt.plot(t_test, x_pred, "--", label="PINN Prediction", color="#e41a1c", lw=2)
    plt.title(f"Damped Harmonic Oscillator (Relative L2 Error = {l2_error:.2e})", fontsize=11, fontweight="bold")
    plt.xlabel("Time t (s)", fontsize=10)
    plt.ylabel("Displacement x(t)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    out_plot = os.path.join(out_dir, "pinn_harmonic_oscillator.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved benchmark figure: {out_plot}")
    return l2_error

def train_burgers(epochs=2000, lr=1e-3, out_dir="."):
    print("=================================================================")
    print("  TRAINING PINN: 1D VISCOUS BURGERS' EQUATION")
    print("=================================================================")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2 inputs (x, t), 1 output (u)
    net = PINNNetwork(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, activation="tanh")
    solver = BurgersPINNSolver(net)
    data = solver.generate_collocation_points(n_colloc=2500, n_ic=250, n_bc=250)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_dict = solver.compute_loss(data)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.5e} (PDE: {loss_dict['pde']:.5e})")
            
    # Generate 2D Spatio-Temporal Contour Plot
    x_grid = np.linspace(-1, 1, 100)
    t_grid = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_grid, t_grid)
    xt_flat = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    
    u_pred = solver.predict(xt_flat).reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    cp = ax.contourf(T, X, u_pred, levels=50, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Velocity Field u(x, t)")
    ax.set_title("1D Viscous Burgers' Equation: PINN Solution u(x, t)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time t", fontsize=10)
    ax.set_ylabel("Space x", fontsize=10)
    plt.tight_layout()
    
    out_plot = os.path.join(out_dir, "pinn_burgers_solution.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved spatio-temporal solution plot: {out_plot}")
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN Benchmarks")
    parser.add_argument("--problem", type=str, default="harmonic", choices=["harmonic", "burgers"], help="PDE problem to solve")
    parser.add_argument("--epochs", type=int, default=1500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.problem == "harmonic":
        train_harmonic(epochs=args.epochs, lr=args.lr, out_dir=base_dir)
    else:
        train_burgers(epochs=args.epochs, lr=args.lr, out_dir=base_dir)
