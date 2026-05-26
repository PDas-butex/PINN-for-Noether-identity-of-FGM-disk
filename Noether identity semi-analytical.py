# This code is created by P.Das from BUTex, Bangladesh
# pinn_noether_fgm_cylinder.py
# Single-file PyTorch PINN for:
# sigma(r), S(r), R(r), G(r), and conservation constant C
# ri=1, ro=2, Ei=1, Eo=2, nu_bar=0.7, sigma(ri)=-1, sigma(ro)=0

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# -----------------------------
# Problem parameters
# -----------------------------
ri = 1.0
ro = 2.0

Ei = 1.0
Eo = 2.0
Ep_const = (Eo - Ei) / (ro - ri)

nu = 0.3
vbar = 1.0 - nu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Material law: Voigt linear E(r)
# V=(r-ri)/(ro-ri)
# E=Eo*V + Ei*(1-V)
# -----------------------------
def E_fun(r):
    V = (r - ri) / (ro - ri)
    return Eo * V + Ei * (1.0 - V)

def Ep_fun(r):
    return torch.ones_like(r) * Ep_const

# -----------------------------
# Neural network
# outputs raw correction for sigma,
# and raw S, R, G
# -----------------------------
class PINN(nn.Module):
    def __init__(self, width=80, depth=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

        # learnable Noether constant
        self.C = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))

    def forward(self, r):
        # normalize r from [ri,ro] to [-1,1]
        x = 2.0 * (r - ri) / (ro - ri) - 1.0
        out = self.net(x)

        raw_sigma = out[:, 0:1]
        S = out[:, 1:2]
        R = out[:, 2:3]
        G = out[:, 3:4]

        # hard-enforce pressure boundary conditions:
        # sigma(ri)=-1, sigma(ro)=0
        sigma_base = -1.0 * (ro - r) / (ro - ri)
        bubble = (r - ri) * (ro - r)
        sigma = sigma_base + bubble * raw_sigma

        return sigma, S, R, G

model = PINN(width=96, depth=6).to(device)

# -----------------------------
# Autograd derivative
# -----------------------------
def grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

# -----------------------------
# PINN residuals
# -----------------------------
def residuals(r):
    r.requires_grad_(True)

    sigma, S, R, G = model(r)

    sig_p = grad(sigma, r)
    sig_pp = grad(sig_p, r)

    S_p = grad(S, r)
    R_p = grad(R, r)
    G_p = grad(G, r)

    E = E_fun(r)
    Ep = Ep_fun(r)

    # stress governing equation:
    # r^2 sigma'' + r sigma'(3 - E'/E r) - vbar E'/E r sigma = 0
    res_sigma = (
        r**2 * sig_pp
        + r * sig_p * (3.0 - (Ep / E) * r)
        - vbar * (Ep / E) * r * sigma
    )

    # generator equations from reduced Killing system:
    # S' = -2vbar/r S - vbar/r^2(2 - E'/E r)R + E/(2r^3)G
    # R' = 2S + 1/r(3 - E'/E r)R
    # G' = -8vbar(vbar-2)r/E S -4vbar(vbar-2)/E(2 - E'/E r)R + 2vbar/r G

    A = (2.0 - (Ep / E) * r)
    B = (3.0 - (Ep / E) * r)

    rhs_S = (
        -2.0 * vbar / r * S
        - vbar / r**2 * A * R
        + E / (2.0 * r**3) * G
    )

    rhs_R = 2.0 * S + (1.0 / r) * B * R

    rhs_G = (
        -8.0 * vbar * (vbar - 2.0) * r / E * S
        -4.0 * vbar * (vbar - 2.0) / E * A * R
        + 2.0 * vbar / r * G
    )

    res_S = S_p - rhs_S
    res_R = R_p - rhs_R
    res_G = G_p - rhs_G

    # Sigma = S*sigma
    Sigma = S * sigma

    # P = 1/2 G sigma^2
    P = 0.5 * G * sigma**2

    # Conservation law:
    # I = d_phi/d_sigma' * Sigma + (Phi - d_phi/d_sigma' sigma')R - P
    # equivalent paper form:
    # I = r/E [2r(vbar sigma + r sigma')Sigma
    #          + (2vbar sigma^2 - (r sigma')^2)R] - P
    I = (
        r / E
        * (
            2.0 * r * (vbar * sigma + r * sig_p) * Sigma
            + (2.0 * vbar * sigma**2 - (r * sig_p)**2) * R
        )
        - P
    )

    res_I = I - model.C

    return res_sigma, res_S, res_R, res_G, res_I, I, sigma, S, R, G

# -----------------------------
# Collocation points
# -----------------------------
N_col = 400
r_col = torch.linspace(ri, ro, N_col, device=device).view(-1, 1)

# generator initial conditions to avoid trivial zero generator
# same spirit as paper numerical section
r_left = torch.tensor([[ri]], dtype=torch.float64, device=device)

S_left_target = torch.tensor([[0.0]], dtype=torch.float64, device=device)
R_left_target = torch.tensor([[-1.0]], dtype=torch.float64, device=device)
G_left_target = torch.tensor([[1.0]], dtype=torch.float64, device=device)

# -----------------------------
# Loss function
# -----------------------------
def loss_fn():
    res_sigma, res_S, res_R, res_G, res_I, I, sigma, S, R, G = residuals(r_col)

    loss_sigma = torch.mean(res_sigma**2)
    loss_gen = torch.mean(res_S**2) + torch.mean(res_R**2) + torch.mean(res_G**2)
    loss_const = torch.mean(res_I**2)

    sigma_l, S_l, R_l, G_l = model(r_left)
    loss_ic_gen = (
        (S_l - S_left_target).pow(2).mean()
        + (R_l - R_left_target).pow(2).mean()
        + (G_l - G_left_target).pow(2).mean()
    )

    # sigma BC is already hard-enforced.
    # weights can be tuned if needed.
    total = (
        1.0 * loss_sigma
        + 1.0 * loss_gen
        + 10.0 * loss_const
        + 10.0 * loss_ic_gen
    )

    return total, loss_sigma, loss_gen, loss_const, loss_ic_gen

# -----------------------------
# Adam training
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_adam = 20000
for epoch in range(1, n_adam + 1):
    optimizer.zero_grad()
    total, ls, lg, lc, lic = loss_fn()
    total.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(
            f"Adam {epoch:6d} | total={total.item():.3e} | "
            f"sigma={ls.item():.3e} | gen={lg.item():.3e} | "
            f"const={lc.item():.3e} | ic={lic.item():.3e} | C={model.C.item():.8e}"
        )

# -----------------------------
# LBFGS refinement
# -----------------------------
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=3000,
    max_eval=3000,
    tolerance_grad=1e-12,
    tolerance_change=1e-12,
    history_size=100,
    line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()
    total, _, _, _, _ = loss_fn()
    total.backward()
    return total

print("Starting LBFGS...")
optimizer_lbfgs.step(closure)

total, ls, lg, lc, lic = loss_fn()
print("\nFinal losses")
print("total      =", total.item())
print("sigma ODE  =", ls.item())
print("gen ODE    =", lg.item())
print("constant   =", lc.item())
print("gen IC     =", lic.item())
print("Learned C  =", model.C.item())

# -----------------------------
# Evaluation
# -----------------------------
r_eval = torch.linspace(ri, ro, 500, device=device).view(-1, 1)
res_sigma, res_S, res_R, res_G, res_I, I, sigma, S, R, G = residuals(r_eval)

r_np = r_eval.detach().cpu().numpy().flatten()
sigma_np = sigma.detach().cpu().numpy().flatten()
S_np = S.detach().cpu().numpy().flatten()
R_np = R.detach().cpu().numpy().flatten()
G_np = G.detach().cpu().numpy().flatten()
I_np = I.detach().cpu().numpy().flatten()

C_val = model.C.detach().cpu().item()

print("\nConservation result")
print("C learned      =", C_val)
print("mean(I)        =", I_np.mean())
print("std(I)         =", I_np.std())
print("max|I-C|       =", np.max(np.abs(I_np - C_val)))
print("sigma(ri)      =", sigma_np[0])
print("sigma(ro)      =", sigma_np[-1])

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(r_np, sigma_np, linewidth=2.5)
plt.xlabel("r")
plt.ylabel(r"$\sigma_r(r)$")
plt.title("Radial stress from PINN")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(r_np, I_np, linewidth=2.5, label=r"$I(r)$")
plt.axhline(C_val, linestyle="--", linewidth=2.0, label=r"$C$")
plt.xlabel("r")
plt.ylabel("Conservation law")
plt.title("Noether conservation constant")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(r_np, S_np, linewidth=2.0, label="S(r)")
plt.plot(r_np, R_np, linewidth=2.0, label="R(r)")
plt.plot(r_np, G_np, linewidth=2.0, label="G(r)")
plt.xlabel("r")
plt.ylabel("Generator functions")
plt.title("PINN solution of generator system")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
