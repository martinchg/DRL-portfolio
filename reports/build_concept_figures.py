# reports/build_concept_figures.py
# ============================================================
# Figures PÉDAGOGIQUES du rapport (schémas + illustrations de concepts).
# Autonome : numpy + matplotlib uniquement, aucun modèle ni réseau requis
# (sauf fig_seed_dispersion qui lit reports/seed_robustness.json s'il existe).
#
#   .venv/bin/python reports/build_concept_figures.py
# ============================================================
import os
import io
import json
import contextlib

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

C0, C1, CBH, CBAD = "#2ecc71", "#e67e22", "#3498db", "#e74c3c"
INK, MUTED = "#222222", "#888888"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"   figures/{name} ✅")


# ============================================================
# 1. Boucle MDP agent ↔ environnement
# ============================================================
def fig_mdp_loop():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                    linewidth=2, edgecolor=color, facecolor=color + "22"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)

    box(0.6, 2.2, 3.2, 1.6, "AGENT\npolitique $\\pi_\\theta(a\\,|\\,s)$\nréseau PPO", C0)
    box(6.2, 2.2, 3.2, 1.6, "ENVIRONNEMENT\nmarché simulé\n(TradingEnv)", CBH)

    # action : agent → env (haut)
    ax.add_patch(FancyArrowPatch((3.8, 3.4), (6.2, 3.4),
                 arrowstyle="-|>", mutation_scale=20, linewidth=2, color=C0))
    ax.text(5.0, 4.15, "action $a_t$", ha="center", fontsize=10, color=INK)
    ax.text(5.0, 3.62, "Hold / Long / Flat / Short", ha="center",
            fontsize=8, color=MUTED)

    # état + récompense : env → agent (bas)
    ax.add_patch(FancyArrowPatch((6.2, 2.6), (3.8, 2.6),
                 arrowstyle="-|>", mutation_scale=20, linewidth=2, color=CBH))
    ax.text(5.0, 1.75, "état $s_{t+1}$  +  récompense $r_t$", ha="center",
            fontsize=10, color=INK)
    ax.text(5.0, 1.28, "10 j × 5 features + position, PnL   |   $r_t=$ alpha du pas",
            ha="center", fontsize=8, color=MUTED)

    ax.text(5.0, 5.35, "Le trading comme processus de décision markovien (MDP)",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(5.0, 0.45,
            "L'agent maximise $\\mathbb{E}\\left[\\sum_t \\gamma^t r_t\\right]$"
            "   —   $\\gamma=0{,}99$  →  horizon $\\approx 100$ jours",
            ha="center", fontsize=10, color=INK)
    save(fig, "fig_mdp_loop.png")


# ============================================================
# 2. Pipeline du projet
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
    stages = [
        ("data_loader", "features,\nsplit, scaler", C0),
        ("environment", "MDP,\nfrais, stop", CBH),
        ("train", "PPO +\nVecNormalize", C1),
        ("evaluate", "full-split,\nrisk metrics", CBAD),
        ("reports", "rapport,\ndashboard", "#9b59b6"),
    ]
    w, gap = 1.85, 0.28
    for i, (title, sub, color) in enumerate(stages):
        x = 0.15 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.7), w, 1.2,
                    boxstyle="round,pad=0.1", linewidth=2,
                    edgecolor=color, facecolor=color + "20"))
        ax.text(x + w / 2, 1.5, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, 1.0, sub, ha="center", va="center",
                fontsize=8, color=MUTED)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.3), (x + w + gap, 1.3),
                         arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
                         color=MUTED))
    ax.text(5.5, 2.35, "Pipeline du projet — chaque étage testé indépendamment",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    save(fig, "fig_pipeline.png")


# ============================================================
# 3. Schéma walk-forward (fenêtres ancrées croissantes)
# ============================================================
def fig_walkforward_schema():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    years = list(range(2010, 2023))
    folds = [2018, 2019, 2020, 2021, 2022]
    ax.set_xlim(2009.5, 2023.5); ax.set_ylim(-0.6, len(folds) - 0.4)

    for row, test_year in enumerate(folds):
        y = len(folds) - 1 - row
        # train (ancré à 2010)
        ax.barh(y, test_year - 2010, left=2010, height=0.6,
                color=C0 + "cc", edgecolor=C0)
        # test (l'année suivante)
        ax.barh(y, 1, left=test_year, height=0.6,
                color=C1, edgecolor=C1)
        ax.text(2010 + (test_year - 2010) / 2, y, "train + val",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.text(test_year + 0.5, y, "test", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        ax.text(2009.3, y, f"fold {test_year}", ha="right", va="center",
                fontsize=9, color=INK)

    ax.set_yticks([])
    ax.set_xticks(years[::2])
    ax.set_xlabel("année")
    ax.spines["left"].set_visible(False)
    ax.set_title("Walk-forward ancré (anchored expanding) — chaque année de test "
                 "est prédite\npar un modèle réentraîné uniquement sur son passé",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0)
    save(fig, "fig_walkforward_schema.png")


# ============================================================
# 4. Processus d'Ornstein-Uhlenbeck : retour à la moyenne
# ============================================================
def fig_ou_process():
    rng = np.random.default_rng(1)
    T, dt, mu, sigma, X0 = 250, 1.0, 0.0, 0.15, 3.0
    fig, ax = plt.subplots(figsize=(9, 4))
    for kappa, color, lab in [(0.01, CBH, "$\\kappa=0{,}01$ (lent)"),
                              (0.04, C1, "$\\kappa=0{,}04$"),
                              (0.12, C0, "$\\kappa=0{,}12$ (rapide)")]:
        X = np.empty(T); X[0] = X0
        z = rng.standard_normal(T)
        for t in range(1, T):
            X[t] = X[t - 1] + kappa * (mu - X[t - 1]) * dt + sigma * np.sqrt(dt) * z[t]
        hl = np.log(2) / kappa
        ax.plot(X, color=color, linewidth=1.6,
                label=f"{lab}  —  demi-vie $t_{{1/2}}={hl:.0f}$ j")
    ax.axhline(mu, color=INK, linestyle="--", linewidth=1)
    ax.text(T * 0.985, mu + 0.12, "$\\mu$ (moyenne long terme)", ha="right",
            fontsize=9, color=INK)
    ax.set_xlabel("temps (jours)"); ax.set_ylabel("$X_t$")
    ax.set_title("Ornstein-Uhlenbeck : $dX_t=\\kappa(\\mu-X_t)\\,dt+\\sigma\\,dW_t$\n"
                 "l'écart à $\\mu$ se résorbe d'autant plus vite que $\\kappa$ est grand "
                 "(demi-vie $=\\ln 2/\\kappa$)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig_ou_process.png")


# ============================================================
# 5. Objectif clippé de PPO
# ============================================================
def fig_ppo_clip():
    eps = 0.2
    rho = np.linspace(0, 2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, A, color, title in [
        (axes[0], 1.0, C0, "Avantage $A_t>0$  (bonne action)"),
        (axes[1], -1.0, CBAD, "Avantage $A_t<0$  (mauvaise action)")]:
        unclipped = rho * A
        clipped = np.clip(rho, 1 - eps, 1 + eps) * A
        # PPO prend min(ρA, clip·A) dans les deux cas
        obj = np.minimum(unclipped, clipped)
        ax.plot(rho, unclipped, color=MUTED, linestyle=":", linewidth=1.4,
                label="non clippé $\\rho_t A_t$")
        ax.plot(rho, obj, color=color, linewidth=2.4, label="objectif PPO (min)")
        ax.axvspan(1 - eps, 1 + eps, color=color, alpha=0.08)
        ax.axvline(1, color=INK, linewidth=0.8, linestyle="--")
        ax.set_xlabel("ratio $\\rho_t=\\pi_\\theta/\\pi_{\\theta_{old}}$")
        ax.set_title(title, fontsize=10, pad=8)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("objectif")
    axes[0].text(1 + eps + 0.03, 1.35, "$1+\\epsilon$", fontsize=8, color=INK)
    axes[1].text(1 - eps - 0.42, -1.35, "$1-\\epsilon$", fontsize=8, color=INK)
    fig.suptitle("Le clipping PPO borne l'incitation : au-delà de $1\\pm\\epsilon$, "
                 "le gradient s'annule (plateau)\n→ aucune mise à jour ne « surexploite » "
                 "un lot d'expérience", fontsize=11, fontweight="bold", y=1.04)
    fig.subplots_adjust(top=0.80)
    save(fig, "fig_ppo_clip.png")


# ============================================================
# 6. Dispersion inter-seeds (vraies données si dispo)
# ============================================================
def fig_seed_dispersion():
    path = os.path.join(ROOT, "reports", "seed_robustness.json")
    if not os.path.exists(path):
        print("   (seed_robustness.json absent — figure ignorée)")
        return
    d = json.load(open(path))
    seeds = sorted(d["per_seed"], key=int)
    aapl = [d["per_seed"][s]["aapl_alpha"] * 100 for s in seeds]
    cross = [d["per_seed"][s]["cross_alpha_mean"] * 100 for s in seeds]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(seeds))
    ax.scatter(x, aapl, s=90, color=CBAD, zorder=3, label="alpha AAPL (1 actif)")
    ax.scatter(x, cross, s=90, color=C0, marker="s", zorder=3,
               label="alpha moyen (5 actifs)")
    ax.axhline(np.mean(aapl), color=CBAD, linestyle=":", linewidth=1.2)
    ax.axhline(np.mean(cross), color=C0, linestyle=":", linewidth=1.2)
    # bandes ±1σ
    ax.axhspan(np.mean(aapl) - np.std(aapl), np.mean(aapl) + np.std(aapl),
               color=CBAD, alpha=0.08)
    ax.axhspan(np.mean(cross) - np.std(cross), np.mean(cross) + np.std(cross),
               color=C0, alpha=0.12)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
    ax.set_ylabel("alpha test 2021-22 (%)")
    ax.set_title("Le bruit de seed se diversifie : loterie sur 1 actif "
                 f"(±{np.std(aapl):.0f} pts),\nstable sur 5 (±{np.std(cross):.0f} pts) "
                 "— même logique que le risque diversifiable de Markowitz",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "fig_seed_dispersion.png")


# ============================================================
# 7. Volatility clustering : GARCH vs bruit i.i.d.
# ============================================================
def fig_vol_clustering():
    rng = np.random.default_rng(3)
    n = 800
    omega, alpha, beta = 0.05, 0.10, 0.88
    sig2 = np.empty(n); eps = np.empty(n)
    sig2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    iid = np.std(eps) * rng.standard_normal(n)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True)
    axes[0].plot(eps, color=C1, linewidth=0.7)
    axes[0].set_title("GARCH(1,1) — les grandes variations se regroupent "
                      "(volatility clustering)", fontsize=10)
    axes[0].set_ylabel("rendement")
    axes[1].plot(iid, color=MUTED, linewidth=0.7)
    axes[1].set_title("Bruit gaussien i.i.d. (même variance) — aucun regroupement",
                      fontsize=10)
    axes[1].set_ylabel("rendement"); axes[1].set_xlabel("temps")
    fig.suptitle("Ce que la feature « volatility » capture : le regroupement de "
                 "volatilité,\nfait stylisé des marchés modélisé par GARCH (Engle, "
                 "Bollerslev)", fontsize=11, fontweight="bold")
    save(fig, "fig_vol_clustering.png")


# ============================================================
# 8. Frontière efficiente de Markowitz + ratio de Sharpe
# ============================================================
def fig_efficient_frontier():
    rng = np.random.default_rng(5)
    n_assets = 4
    mu = np.array([0.06, 0.10, 0.14, 0.09])
    vol = np.array([0.10, 0.18, 0.26, 0.15])
    corr = np.array([
        [1.0, 0.3, 0.2, 0.4], [0.3, 1.0, 0.5, 0.3],
        [0.2, 0.5, 1.0, 0.25], [0.4, 0.3, 0.25, 1.0]])
    cov = np.outer(vol, vol) * corr

    W = rng.dirichlet(np.ones(n_assets), 8000)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sharpe = rets / vols

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    sc = ax.scatter(vols * 100, rets * 100, c=sharpe, cmap="viridis", s=6, alpha=0.6)
    imax = int(np.argmax(sharpe))
    ax.scatter(vols[imax] * 100, rets[imax] * 100, marker="*", s=320,
               color=CBAD, edgecolor="black", zorder=5,
               label=f"max Sharpe = {sharpe[imax]:.2f}")
    # capital market line (r_f = 2%)
    rf = 0.02
    xs = np.linspace(0, vols.max() * 100, 50)
    ax.plot(xs, rf * 100 + sharpe[imax] * xs, color=CBAD, linestyle="--",
            linewidth=1.3, label="Capital Market Line")
    for i in range(n_assets):
        ax.scatter(vol[i] * 100, mu[i] * 100, marker="o", s=60,
                   edgecolor="black", facecolor="white", zorder=4)
        ax.text(vol[i] * 100 + 0.4, mu[i] * 100, f"actif {i+1}", fontsize=8)
    fig.colorbar(sc, ax=ax, label="ratio de Sharpe")
    ax.set_xlabel("volatilité annualisée (%)"); ax.set_ylabel("rendement attendu (%)")
    ax.set_title("Frontière efficiente de Markowitz : chaque point = un portefeuille\n"
                 "le Sharpe est la pente depuis $r_f$ ; la diversification déplace "
                 "le nuage vers le haut-gauche", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig_efficient_frontier.png")


if __name__ == "__main__":
    print("Figures conceptuelles :")
    fig_mdp_loop()
    fig_pipeline()
    fig_walkforward_schema()
    fig_ou_process()
    fig_ppo_clip()
    fig_seed_dispersion()
    fig_vol_clustering()
    fig_efficient_frontier()
    print("Terminé.")
