# Physics-Informed Neural Networks (PINN) Framework for Scientific Computing

[![CI Verification](https://github.com/jamshaidal/physics-informed-ml-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/jamshaidal/physics-informed-ml-framework/actions/workflows/ci.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Field](https://img.shields.io/badge/Field-Scientific%20Machine%20Learning-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular, research-grade PyTorch implementation of Physics-Informed Neural Networks (PINNs) designed for forward and inverse problems in differential equations, boundary-value constraints, and physical surrogate simulations.

---

## Mathematical Formulation

Standard deep learning models act as black-box approximators. A Physics-Informed Neural Network constrains the solution space by embedding the governing partial differential equation (PDE) directly into the loss function via **automatic differentiation (AD)**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}} + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}}$$

### Loss Components:
1. **Empirical Data Loss:**
   $$\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} |u(x_i, t_i) - u_i^{\text{obs}}|^2$$
2. **PDE Residual Loss (Collocation Points):**
   $$\mathcal{L}_{\text{pde}} = \frac{1}{N_c} \sum_{j=1}^{N_c} |\mathcal{N}[u_{\theta}](x_j, t_j) - f(x_j, t_j)|^2$$
   where $\mathcal{N}[\cdot]$ is the non-linear differential operator (e.g., Navier-Stokes, Schrödinger, Heat, or Wave equations) evaluated using PyTorch `torch.autograd.grad`.
3. **Boundary Condition (BC) & Initial Condition (IC) Losses:**
   Enforce physical boundary constraints $\mathcal{B}[u](x_b, t) = g(x_b, t)$ across the computational domain.

---

## Supported Physical Systems

- **Reaction-Diffusion Dynamics:** Non-linear chemical kinetics and pattern formation.
- **Wave & Helmholtz Equations:** High-frequency oscillatory wave propagation.
- **Schrödinger Equation Surrogates:** Complex quantum state evolution under parameterized potential wells.
- **Coupled Kinematic Constraints:** Conservation of momentum, energy, and Coulomb asymptotics.

---

## Evaluation & Convergence Metrics

| PDE Formulation | Optimizer Strategy | Collocation Points | Relative $L_2$ Error | Residual Norm $\|\mathcal{N}[u]\|$ |
| :--- | :--- | :--- | :--- | :--- |
| **Non-linear Diffusion** | Adam + L-BFGS | 10,000 | **$3.8 \times 10^{-4}$** | **$1.2 \times 10^{-5}$** |
| **Burgers' Equation** | Adam + L-BFGS | 20,000 | **$5.1 \times 10^{-4}$** | **$2.4 \times 10^{-5}$** |
| **Wave Propagation** | Multi-scale Fourier PINN | 15,000 | **$8.9 \times 10^{-4}$** | **$6.7 \times 10^{-5}$** |

---

## Quickstart & Verification

```bash
git clone https://github.com/jamshaidal/physics-informed-ml-framework.git
cd physics-informed-ml-framework
pip install -r requirements.txt

# Run automated verification test suite
python verify_pinn_benchmarks.py

# Train 1D Viscous Burgers' solver
python train_pinn.py --pde burgers --epochs 2000
```

---

## Citation

If you utilize this PINN architecture, PDE solvers, or benchmark suite in your research, please cite:

```bibtex
@software{ali2026pinn_framework,
  author       = {Ali, Muhammad Jamshaid},
  title        = {Physics-Informed Neural Networks Framework for Scientific Differential Equations},
  year         = {2026},
  url          = {https://github.com/jamshaidal/physics-informed-ml-framework}
}
```
You can also view the machine-readable [`CITATION.cff`](CITATION.cff) file or click **"Cite this repository"** in the GitHub sidebar.

---

## Inquiries

**Muhammad Jamshaid Ali**  
Computational Physics & Scientific Machine Learning Researcher  
- Email: [jamshaid8081@gmail.com](mailto:jamshaid8081@gmail.com)  
- LinkedIn: [linkedin.com/in/muhammad-jamshaid-ali-1687082a0](https://www.linkedin.com/in/muhammad-jamshaid-ali-1687082a0/)
