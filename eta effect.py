# semi_analytical_pinn_selected_eta_beta_full_domain.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Geometry
# ============================================================
ri, ro = 0.5, 1.0
L = ro - ri

# ============================================================
# Boundary conditions
# ============================================================
sigma_i = -1.0
sigma_o = 0.0

# ============================================================
# Material properties
# ============================================================
Ei = 348.43
Eo = 201.04
nui = 0.24
nuo = 0.3262

E_scale = Ei

# ============================================================
# Selected scenarios
# eta=0.2 case excluded
# First run: eta=1, beta=0.1
# ============================================================
cases = [
    {"eta": 1.0,    "beta": 0.1, "label": r"$\eta=1,\ \beta=0.1$"},
    {"eta": 0.0,    "beta": 0.0, "label": r"$\eta=0,\ \beta=0.0$"},
    {"eta": 5.0,    "beta": 0.1, "label": r"$\eta=5,\ \beta=0.1$"},
    {"eta": np.inf, "beta": 0.0, "label": r"$\eta=\infty,\ \beta=0.0$"},
]

# ============================================================
# Training controls
# ============================================================
N_col = 500
n_adam = 30000
lbfgs_iter = 2500

ic_weight = 50.0
const_weight = 10.0

eps_safe = 1.0e-12

# Full-domain collocation because eta=0.2 is excluded
r_col_base = torch.linspace(ri, ro, N_col, device=device).view(-1, 1)
r_eval_base = torch.linspace(ri, ro, 700, device=device).view(-1, 1)

r_left = torch.tensor([[ri]], dtype=torch.float64, device=device)

S_left_target = torch.tensor([[0.0]], dtype=torch.float64, device=device)
R_left_target = torch.tensor([[-1.0]], dtype=torch.float64, device=device)
Ghat_left_target = torch.tensor([[1.0]], dtype=torch.float64, device=device)

# ============================================================
# Autograd derivative
# ============================================================
def grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

# ============================================================
# FGM material law
# ============================================================
def material_fun(r, eta, beta):
    cos_term = 1.0 - beta * torch.cos(
        np.pi / L * (r - 0.5 * (ro + ri))
    )

    if np.isinf(eta):
        E = Eo * torch.ones_like(r) * cos_term
        nu = nuo * torch.ones_like(r) * cos_term
    else:
        base = (ro - r) / L

        if float(eta) == 0.0:
            pow_term = torch.ones_like(base)
        else:
            pow_term = base ** eta

        E_base = (Ei - Eo) * pow_term + Eo
        nu_base = (nui - nuo) * pow_term + nuo

        E = E_base * cos_term
        nu = nu_base * cos_term

    return E, nu

# ============================================================
# PINN: sigma(r), S(r), R(r), Ghat(r)
# Physical G = Ghat / E_scale
# ============================================================
class PINN(nn.Module):
    def __init__(self, width=96, depth=6):
        super().__init__()

        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 4)]

        self.net = nn.Sequential(*layers)
        self.C = nn.Parameter(torch.tensor([0.0], dtype=torch.float64, device=device))

    def forward(self, r):
        x = 2.0 * (r - ri) / L - 1.0
        out = self.net(x)

        raw_sigma = out[:, 0:1]
        S = out[:, 1:2]
        R = out[:, 2:3]
        Ghat = out[:, 3:4]

        # Hard BC:
        # sigma(ri)=-1, sigma(ro)=0
        sigma_base = sigma_i * (ro - r) / L + sigma_o * (r - ri) / L
        bubble = (r - ri) * (ro - r)
        sigma = sigma_base + bubble * raw_sigma

        return sigma, S, R, Ghat

# ============================================================
# Residuals
# ============================================================
def make_residuals(model, eta, beta):

    def residuals(r_in):
        r = r_in.clone().detach().requires_grad_(True)

        sigma, S, R, Ghat = model(r)
        G = Ghat / E_scale

        sig_p = grad(sigma, r)
        sig_pp = grad(sig_p, r)

        S_p = grad(S, r)
        R_p = grad(R, r)
        Ghat_p = grad(Ghat, r)

        E, nu = material_fun(r, eta, beta)

        Ep = grad(E, r)
        nup = grad(nu, r)

        E_safe = torch.clamp(E, min=eps_safe)

        vb = 1.0 - nu
        vb_safe = torch.clamp(vb, min=eps_safe)

        Ep_over_E = Ep / E_safe
        nup_over_vb = nup / vb_safe

        # ----------------------------------------------------
        # Stress ODE
        # ----------------------------------------------------
        res_sigma = (
            r**2 * sig_pp
            + (3.0 - Ep_over_E * r) * r * sig_p
            - (r * nup + vb * Ep_over_E * r) * sigma
        )

        # ----------------------------------------------------
        # Generator ODEs
        # Scaling:
        # physical G = Ghat/E_scale
        # E*G = (E/E_scale)*Ghat
        # ----------------------------------------------------
        Ebar = E / E_scale

        A1 = 1.0 - Ep_over_E * r - nup_over_vb * r
        A2 = 1.0 - Ep_over_E * r - 2.0 * nup_over_vb * r
        B = 3.0 - Ep_over_E * r

        res_gen_1 = (
            r * S_p
            + 2.0 * S
            + R_p
            + A1 * R / r
            - Ebar * Ghat_p / (4.0 * vb_safe * r)
        )

        res_gen_2 = (
            (2.0 / vb_safe) * r * S_p
            + 2.0 * S
            + R_p
            + A2 * R / r
            - Ebar * Ghat / (vb_safe * r**2)
        )

        res_gen_3 = (
            2.0 * S
            - R_p
            + B * R / r
        )

        # ----------------------------------------------------
        # Noether invariant
        # ----------------------------------------------------
        Gs = S * sigma
        P = 0.5 * G * sigma**2

        Phi = (
            r / E_safe
            * (
                2.0 * vb * sigma * (sigma + r * sig_p)
                + r**2 * sig_p**2
            )
        )

        Phi_sigp = (
            r / E_safe
            * (
                2.0 * vb * r * sigma
                + 2.0 * r**2 * sig_p
            )
        )

        I = Phi_sigp * Gs + (Phi - Phi_sigp * sig_p) * R - P
        res_I = I - model.C

        return (
            res_sigma,
            res_gen_1,
            res_gen_2,
            res_gen_3,
            res_I,
            I,
            sigma,
            S,
            R,
            G,
            Ghat,
            E,
            nu
        )

    return residuals

# ============================================================
# Loss
# ============================================================
def make_loss_fn(model, residuals):

    def loss_fn():
        (
            res_sigma,
            res1,
            res2,
            res3,
            res_I,
            I,
            sigma,
            S,
            R,
            G,
            Ghat,
            E,
            nu
        ) = residuals(r_col_base)

        loss_sigma = torch.mean(res_sigma**2)
        loss_gen = torch.mean(res1**2) + torch.mean(res2**2) + torch.mean(res3**2)
        loss_const = torch.mean(res_I**2)

        _, S_l, R_l, Ghat_l = model(r_left)

        loss_ic = (
            (S_l - S_left_target).pow(2).mean()
            + (R_l - R_left_target).pow(2).mean()
            + (Ghat_l - Ghat_left_target).pow(2).mean()
        )

        total = (
            loss_sigma
            + loss_gen
            + const_weight * loss_const
            + ic_weight * loss_ic
        )

        return total, loss_sigma, loss_gen, loss_const, loss_ic

    return loss_fn

# ============================================================
# Train one case
# ============================================================
def train_case(case_id, eta, beta, label):

    print("\n" + "=" * 80)
    print(f"Case {case_id + 1}: eta={eta}, beta={beta}")
    print("=" * 80)

    model = PINN().to(device)

    residuals = make_residuals(model, eta, beta)
    loss_fn = make_loss_fn(model, residuals)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5000,
        gamma=0.5
    )

    # ---------------- Adam ----------------
    for epoch in range(1, n_adam + 1):
        optimizer.zero_grad()

        total, ls, lg, lc, lic = loss_fn()

        if torch.isnan(total) or torch.isinf(total):
            print(f"NaN/Inf detected at epoch {epoch}")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()

        if epoch % 3000 == 0:
            print(
                f"Adam {epoch:6d} | "
                f"total={total.item():.3e} | "
                f"stress={ls.item():.3e} | "
                f"gen={lg.item():.3e} | "
                f"const={lc.item():.3e} | "
                f"ic={lic.item():.3e} | "
                f"C={model.C.item():.8e}"
            )

    # ---------------- Fix C before LBFGS ----------------
    _, _, _, _, _, I_tmp, *_ = residuals(r_col_base)

    with torch.no_grad():
        model.C.copy_(I_tmp.detach().mean())

    model.C.requires_grad_(False)

    print("C fixed before LBFGS =", model.C.item())

    # ---------------- LBFGS ----------------
    params_lbfgs = [p for p in model.parameters() if p.requires_grad]

    optimizer_lbfgs = torch.optim.LBFGS(
        params_lbfgs,
        lr=0.5,
        max_iter=lbfgs_iter,
        max_eval=lbfgs_iter,
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

    print(
        f"Final | total={total.item():.3e} | "
        f"stress={ls.item():.3e} | gen={lg.item():.3e} | "
        f"const={lc.item():.3e} | ic={lic.item():.3e} | "
        f"C={model.C.item():.8e}"
    )

    # ---------------- Evaluation ----------------
    (
        res_sigma,
        res1,
        res2,
        res3,
        res_I,
        I,
        sigma,
        S,
        R,
        G,
        Ghat,
        E,
        nu
    ) = residuals(r_eval_base)

    r_np = r_eval_base.detach().cpu().numpy().flatten()
    I_np = I.detach().cpu().numpy().flatten()
    C_val = model.C.detach().cpu().item()

    print("Conservation check:")
    print("C fixed      =", C_val)
    print("mean(I)      =", I_np.mean())
    print("std(I)       =", I_np.std())
    print("max|I-C|     =", np.max(np.abs(I_np - C_val)))
    print("rel std(I)   =", I_np.std() / (abs(I_np.mean()) + 1e-12))

    return {
        "eta": eta,
        "beta": beta,
        "label": label,
        "r": r_np,
        "I": I_np,
        "C": C_val,
        "I_minus_C": I_np - C_val,
        "sigma": sigma.detach().cpu().numpy().flatten(),
        "S": S.detach().cpu().numpy().flatten(),
        "R": R.detach().cpu().numpy().flatten(),
        "G": G.detach().cpu().numpy().flatten(),
        "Ghat": Ghat.detach().cpu().numpy().flatten(),
        "E": E.detach().cpu().numpy().flatten(),
        "nu": nu.detach().cpu().numpy().flatten()
    }

# ============================================================
# Run selected cases
# ============================================================
results = []

for i, case in enumerate(cases):
    results.append(
        train_case(
            i,
            case["eta"],
            case["beta"],
            case["label"]
        )
    )

# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(7, 5))
for out in results:
    plt.plot(out["r"], out["I_minus_C"], linewidth=2.4, label=out["label"])
plt.axhline(0.0, linestyle="--", linewidth=1.4)
plt.xlabel(r"$r$")
plt.ylabel(r"$I(r)-C$")
plt.title("Noether invariant error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for out in results:
    plt.plot(out["r"], out["I"], linewidth=2.4, label=out["label"])
plt.xlabel(r"$r$")
plt.ylabel(r"$I(r)$")
plt.title("Noether invariant")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for out in results:
    plt.plot(out["r"], out["sigma"], linewidth=2.4, label=out["label"])
plt.xlabel(r"$r$")
plt.ylabel(r"$\sigma_r(r)$")
plt.title("Radial stress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for out in results:
    plt.plot(out["r"], out["E"], linewidth=2.4, label=out["label"])
plt.xlabel(r"$r$")
plt.ylabel(r"$E(r)$")
plt.title("Elastic modulus profiles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for out in results:
    plt.plot(out["r"], out["nu"], linewidth=2.4, label=out["label"])
plt.xlabel(r"$r$")
plt.ylabel(r"$\nu(r)$")
plt.title("Poisson ratio profiles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# Nature-style plots: I(r) and invariant error I(r)-C
# Run this cell after all cases are completed
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

# ----------------------------
# Plot 1: I(r)
# ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)

for out in results:
    ax.plot(
        out["r"],
        out["I"],
        linewidth=2.6,
        label=out["label"]
    )

ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$I(r)$")
ax.set_title(r"Noether invariant", pad=10)

ax.tick_params(direction="in", top=True, right=True)
ax.grid(False)

ax.legend(
    fontsize=16,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),
    frameon=False,
    columnspacing=1.4,
    handlelength=2.8,
    handletextpad=0.6,
    borderaxespad=0.3
)

fig.tight_layout()
fig.savefig("Nature_style_Noether_invariant_I.png", dpi=600, bbox_inches="tight")
fig.savefig("Nature_style_Noether_invariant_I.pdf", bbox_inches="tight")
plt.show()

# ----------------------------
# Plot 2: I(r)-C
# ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)

for out in results:
    ax.plot(
        out["r"],
        out["I_minus_C"],
        linewidth=2.6,
        label=out["label"]
    )

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)

ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$I(r)-C$")
ax.set_title(r"Invariant error", pad=10)

ax.tick_params(direction="in", top=True, right=True)
ax.grid(False)

ax.legend(
    fontsize=16,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.68),
    frameon=False,
    columnspacing=1.4,
    handlelength=2.8,
    handletextpad=0.6,
    borderaxespad=0.3
)

fig.tight_layout()
fig.savefig("Nature_style_invariant_error.png", dpi=600, bbox_inches="tight")
fig.savefig("Nature_style_invariant_error.pdf", bbox_inches="tight")
plt.show()