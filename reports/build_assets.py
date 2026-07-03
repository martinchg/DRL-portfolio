# reports/build_assets.py
# ============================================================
# Génère tous les assets de présentation du projet :
#   - reports/figures/*.png   → figures du rapport LaTeX
#   - reports/metrics.json    → métriques finales + évolution
#   - docs/index.html         → dashboard statique Plotly
#                                (GitHub Pages, zéro installation)
#
# Usage (depuis la racine du repo) :
#   .venv/bin/python reports/build_assets.py
#
# Nécessite : modèles entraînés dans models/ppo_single et models/ppo_multi,
# accès réseau (yfinance).
# ============================================================
import os
import sys
import json
import contextlib
import io

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from data_loader import load_data, DataConfig
from environment import TradingEnv
from evaluate import (
    EVAL_ENV_CFG,
    evaluate_full,
    evaluate_one,
    load_model_and_norm,
)

FIG_DIR = os.path.join(ROOT, "reports", "figures")
DOCS_DIR = os.path.join(ROOT, "docs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

MODELS = {
    "Single (AAPL)":     "models/ppo_single/best_model.zip",
    "Multi (5 tickers)": "models/ppo_multi/best_model.zip",
}
TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"]

# Historique de la session d'amélioration (voir rapport LaTeX, section Évolution).
# Chaque étape est mesurée avec le protocole en vigueur à ce moment-là —
# les protocoles sont précisés car leur correction FAIT PARTIE des améliorations.
EVOLUTION = {
    "avant": {
        "label": "Départ (modèles 2018-2023, protocole fenêtres aléatoires)",
        "single_alpha_test": 0.085,
        "multi_alpha_test": -0.047,
        "single_beat_bh_5": 3,
        "multi_beat_bh_5": 0,
        "multi_tickers_positifs_5": 1,   # mesuré sur le multi pré-correction validation
        "multi_cross_alpha": -0.073,
    },
    "apres": {
        "label": "Final (2010-2023, validation corrigée, protocole full-split)",
        # rempli dynamiquement ci-dessous
    },
}

C0, C1, CBH = "#2ecc71", "#e67e22", "#3498db"   # single, multi, B&H
CBAD = "#e74c3c"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ============================================================
# ÉPISODE DÉTERMINISTE AVEC SÉRIES COMPLÈTES (pour les courbes)
# ============================================================
def episode_series(model_path, data):
    """
    Rejoue l'épisode full-split déterministe (même protocole qu'evaluate_full)
    et retourne les séries datées, gelées en cash après un stop drawdown.
    """
    cfg = EVAL_ENV_CFG
    model, vec = load_model_and_norm(model_path, data, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        env = TradingEnv(data=data, cfg=cfg)
        obs, _ = env.reset(seed=42, options={"random_start": False})
        done, terminated = False, False
        while not done:
            if vec is not None:
                obs_in = vec.normalize_obs(np.array([obs], dtype=np.float32))[0]
            else:
                obs_in = obs
            action, _ = model.predict(obs_in, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

    portfolio = np.array(env.history["portfolio_values"], dtype=float)
    prices    = np.array(env.history["prices"], dtype=float)
    n_traded  = len(portfolio)

    stop_idx = None
    if terminated:
        stop_idx = n_traded - 1
        rest = env.prices[env._current_step: env._seg_end].astype(float)
        if len(rest) > 0:
            portfolio = np.concatenate([portfolio, np.full(len(rest), portfolio[-1])])
            prices    = np.concatenate([prices, rest])

    w = cfg.window_size
    dates = data.index[w: w + len(portfolio)]
    return {
        "dates": dates,
        "portfolio": portfolio,
        "bh": cfg.initial_capital * prices / prices[0],
        "terminated": bool(terminated),
        "stop_idx": stop_idx,
    }


def drawdown(series):
    peak = np.maximum.accumulate(series)
    return (peak - series) / peak


# ============================================================
# 1. ÉVALUATIONS
# ============================================================
print("1/4 — Évaluations…")

with contextlib.redirect_stdout(io.StringIO()):
    train_aapl, val_aapl, test_aapl, _ = load_data(DataConfig(ticker="AAPL"))

full = {name: evaluate_full(path, test_aapl) for name, path in MODELS.items()}
rob  = {name: evaluate_one(path, test_aapl)  for name, path in MODELS.items()}

series = {name: episode_series(path, test_aapl) for name, path in MODELS.items()}

cross = {}
for ticker in TICKERS:
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, tdata, _ = load_data(DataConfig(ticker=ticker))
    cross[ticker] = evaluate_full(MODELS["Multi (5 tickers)"], tdata)

EVOLUTION["apres"].update({
    "single_alpha_test": full["Single (AAPL)"]["alpha"],
    "multi_alpha_test":  full["Multi (5 tickers)"]["alpha"],
    "single_beat_bh_5":  rob["Single (AAPL)"]["beat_bh"],
    "multi_beat_bh_5":   rob["Multi (5 tickers)"]["beat_bh"],
    "multi_tickers_positifs_5": sum(1 for r in cross.values() if r["alpha"] > 0),
    "multi_cross_alpha": float(np.mean([r["alpha"] for r in cross.values()])),
})

metrics = {
    "periode": {"start": DataConfig().start_date, "end": DataConfig().end_date,
                "test_start": str(test_aapl.index[0].date()),
                "test_end": str(test_aapl.index[-1].date())},
    "full_test": full,
    "robustesse": rob,
    "cross_ticker": cross,
    "evolution": EVOLUTION,
}
with open(os.path.join(ROOT, "reports", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
print("   reports/metrics.json ✅")


# ============================================================
# 2. FIGURES MATPLOTLIB (rapport LaTeX)
# ============================================================
print("2/4 — Figures PNG…")


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   figures/{name} ✅")


# ── Splits des données ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.6))
data_all = np.concatenate([train_aapl["price"], val_aapl["price"], test_aapl["price"]])
idx_all = train_aapl.index.append(val_aapl.index).append(test_aapl.index)
ax.plot(idx_all, data_all, color="#555", lw=1)
ax.axvspan(train_aapl.index[0], train_aapl.index[-1], color=C0,  alpha=0.12, label="Train (70 %)")
ax.axvspan(val_aapl.index[0],   val_aapl.index[-1],   color=CBH, alpha=0.15, label="Validation (15 %)")
ax.axvspan(test_aapl.index[0],  test_aapl.index[-1],  color=C1,  alpha=0.18, label="Test (15 %)")
ax.set_title("AAPL 2010→2023 — split temporel strict (aucun shuffle)")
ax.set_ylabel("Prix ($)")
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "fig_data_splits.png")

# ── Equity curves test ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.6))
for (name, s), color in zip(series.items(), [C0, C1]):
    ax.plot(s["dates"], s["portfolio"], color=color, lw=1.8, label=f"Agent {name}")
    if s["stop_idx"] is not None:
        ax.scatter(s["dates"][s["stop_idx"]], s["portfolio"][s["stop_idx"]],
                   color=color, marker="v", s=90, zorder=5,
                   edgecolor="black", linewidth=0.6)
s0 = series["Single (AAPL)"]
ax.plot(s0["dates"], s0["bh"], color=CBH, lw=1.4, ls="--", label="Buy & Hold AAPL")
ax.axhline(EVAL_ENV_CFG.initial_capital, color="gray", ls=":", lw=1)
ax.set_title("Test set (fév. 2021 → déc. 2022) — valeur du portefeuille, "
             "épisode déterministe\n▼ = kill-switch drawdown 25 % "
             "(la stratégie reste ensuite en cash)")
ax.set_ylabel("Valeur ($)")
ax.legend(fontsize=9)
savefig(fig, "fig_equity_test.png")

# ── Drawdowns test ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.6))
for (name, s), color in zip(series.items(), [C0, C1]):
    ax.plot(s["dates"], -100 * drawdown(s["portfolio"]), color=color, lw=1.5, label=name)
ax.plot(s0["dates"], -100 * drawdown(s0["bh"]), color=CBH, lw=1.2, ls="--", label="Buy & Hold")
ax.axhline(-25, color=CBAD, ls="--", lw=1, label="Kill-switch (-25 %)")
ax.set_title("Drawdown depuis le pic (test set)")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=8, loc="lower left")
savefig(fig, "fig_drawdown_test.png")

# ── Cross-ticker ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.8))
alphas = [100 * cross[t]["alpha"] for t in TICKERS]
colors = [C0 if a > 0 else CBAD for a in alphas]
ax.bar(TICKERS, alphas, color=colors, alpha=0.85)
mean_a = float(np.mean(alphas))
ax.axhline(mean_a, color=C1, ls="--", lw=1.4, label=f"Moyenne {mean_a:+.1f} %")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Généralisation : alpha du modèle Multi sur le test set de chaque actif")
ax.set_ylabel("Alpha vs B&H (%)")
ax.legend()
savefig(fig, "fig_cross_ticker.png")

# ── Évolution avant/après ──────────────────────────────────
av, ap = EVOLUTION["avant"], EVOLUTION["apres"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.arange(2)
axes[0].bar(x - 0.18, [100 * av["single_alpha_test"], 100 * av["multi_alpha_test"]],
            width=0.36, color="#95a5a6", label="Avant")
axes[0].bar(x + 0.18, [100 * ap["single_alpha_test"], 100 * ap["multi_alpha_test"]],
            width=0.36, color=C0, label="Après")
axes[0].set_xticks(x, ["Single", "Multi"])
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Alpha sur le test set (%)")
axes[0].legend()

cats = ["Robustesse\nSingle", "Robustesse\nMulti", "Tickers alpha>0\n(Multi)"]
before = [av["single_beat_bh_5"], av["multi_beat_bh_5"], av["multi_tickers_positifs_5"]]
after  = [ap["single_beat_bh_5"], ap["multi_beat_bh_5"], ap["multi_tickers_positifs_5"]]
x = np.arange(3)
axes[1].bar(x - 0.18, before, width=0.36, color="#95a5a6", label="Avant")
axes[1].bar(x + 0.18, after,  width=0.36, color=C0, label="Après")
axes[1].set_xticks(x, cats)
axes[1].set_ylim(0, 5.6)
axes[1].axhline(5, color=C1, ls=":", lw=1)
axes[1].set_title("Sur 5 (fenêtres / actifs)")
axes[1].legend()
fig.suptitle("Avant / après la session d'amélioration "
             "(protocoles respectifs — détails dans le rapport)", fontsize=10)
savefig(fig, "fig_evolution.png")

# ── Walk-forward (si le run a été fait : reports/walk_forward.json) ──
WF_PATH = os.path.join(ROOT, "reports", "walk_forward.json")
wf = None
if os.path.exists(WF_PATH):
    with open(WF_PATH) as f:
        wf = json.load(f)

    wf_years   = [fo["test_year"] for fo in wf["folds"]]
    wf_tickers = wf["config"]["tickers"]
    alpha_grid = 100 * np.array([[fo["per_ticker"][t]["alpha"] for t in wf_tickers]
                                 for fo in wf["folds"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(wf_years))
    axes[0].bar(x - 0.19, [100 * fo["mean_return"] for fo in wf["folds"]],
                width=0.38, color=C1, label="Agent (moy. 5 actifs)")
    axes[0].bar(x + 0.19, [100 * fo["mean_bh"] for fo in wf["folds"]],
                width=0.38, color=CBH, alpha=0.8, label="Buy & Hold (moy.)")
    axes[0].set_xticks(x, wf_years)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Return de l'année OOS (%)")
    axes[0].set_title("Return par année out-of-sample")
    axes[0].legend(fontsize=8)

    # Échelle tronquée à ±100 % : la cellule TSLA-2020 (B&H +576 % → alpha -514 %)
    # écraserait toute la palette ; les valeurs réelles restent annotées.
    vmax = min(100.0, np.abs(alpha_grid).max())
    im = axes[1].imshow(np.clip(alpha_grid, -vmax, vmax), cmap="RdYlGn",
                        vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(range(len(wf_tickers)), wf_tickers, fontsize=8)
    axes[1].set_yticks(range(len(wf_years)), wf_years, fontsize=8)
    for i in range(len(wf_years)):
        for j in range(len(wf_tickers)):
            axes[1].text(j, i, f"{alpha_grid[i, j]:+.0f}", ha="center",
                         va="center", fontsize=8)
    axes[1].set_title("Alpha vs B&H par cellule année × actif (%)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    agg = wf["aggregate"]
    fig.suptitle(
        f"Walk-forward : {len(wf_years)} réentraînements, modèle jamais exposé à son "
        f"année de test — alpha médian {100 * agg['median_alpha']:+.1f} %, "
        f"{agg['pct_positive']:.0%} de cellules positives (échelle couleur ±{vmax:.0f} %)",
        fontsize=10)
    savefig(fig, "fig_walk_forward.png")

# ── Distribution des actions ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 2.8))
action_cols = [("hold_pct", "Hold", "#95a5a6"), ("long_pct", "Long", C0),
               ("flat_pct", "Flat", CBH), ("short_pct", "Short", CBAD)]
names = list(full.keys())
left = np.zeros(len(names))
for key, lab, color in action_cols:
    vals = np.array([100 * full[n][key] for n in names])
    ax.barh(names, vals, left=left, color=color, alpha=0.85, label=lab)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v > 6:
            ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8)
    left += vals
ax.set_xlim(0, 100)
ax.set_title("Répartition des actions sur le test complet")
ax.legend(ncol=4, fontsize=8, loc="lower right")
savefig(fig, "fig_actions.png")

# ── Robustesse ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.4))
names = list(rob.keys())
means = [100 * rob[n]["alpha"] for n in names]
stds  = [100 * rob[n]["alpha_std"] for n in names]
ax.bar(names, means, yerr=stds, capsize=6, color=[C0, C1], alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
for i, n in enumerate(names):
    ax.text(i, means[i] + stds[i] + 0.6, f"{rob[n]['beat_bh']}/5 fenêtres > B&H",
            ha="center", fontsize=9)
ax.set_title("Robustesse : alpha moyen ± écart-type\n(5 sous-fenêtres aléatoires du test)")
ax.set_ylabel("Alpha (%)")
savefig(fig, "fig_robustness.png")


# ============================================================
# 3. DASHBOARD STATIQUE PLOTLY (docs/index.html)
# ============================================================
print("3/4 — Dashboard statique…")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b111c",
    font=dict(family="'Space Grotesk', system-ui, sans-serif", size=13,
              color="#cfd9e4"),
    margin=dict(l=50, r=30, t=60, b=40),
    legend=dict(orientation="h", y=-0.15),
)


def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


# Equity interactif
fig_eq = go.Figure()
for (name, s), color in zip(series.items(), [C0, C1]):
    fig_eq.add_trace(go.Scatter(x=s["dates"], y=s["portfolio"], name=f"Agent {name}",
                                line=dict(color=color, width=2)))
    if s["stop_idx"] is not None:
        fig_eq.add_trace(go.Scatter(
            x=[s["dates"][s["stop_idx"]]], y=[s["portfolio"][s["stop_idx"]]],
            mode="markers", marker=dict(symbol="triangle-down", size=13, color=color),
            name=f"Stop drawdown — {name}", showlegend=False,
            hovertemplate="Kill-switch 25 %<extra></extra>"))
fig_eq.add_trace(go.Scatter(x=s0["dates"], y=s0["bh"], name="Buy & Hold AAPL",
                            line=dict(color=CBH, width=1.5, dash="dash")))
fig_eq.update_layout(title="Valeur du portefeuille — test set (fév. 2021 → déc. 2022)",
                     yaxis_title="Valeur ($)", **PLOTLY_LAYOUT)

# Drawdown interactif
fig_dd = go.Figure()
for (name, s), color in zip(series.items(), [C0, C1]):
    fig_dd.add_trace(go.Scatter(x=s["dates"], y=-100 * drawdown(s["portfolio"]),
                                name=name, line=dict(color=color, width=1.8)))
fig_dd.add_trace(go.Scatter(x=s0["dates"], y=-100 * drawdown(s0["bh"]),
                            name="Buy & Hold", line=dict(color=CBH, width=1.3, dash="dash")))
fig_dd.add_hline(y=-25, line_dash="dash", line_color=CBAD,
                 annotation_text="kill-switch -25 %")
fig_dd.update_layout(title="Drawdown depuis le pic", yaxis_title="Drawdown (%)",
                     **PLOTLY_LAYOUT)

# Cross-ticker
fig_ct = go.Figure(go.Bar(
    x=TICKERS, y=[100 * cross[t]["alpha"] for t in TICKERS],
    marker_color=[C0 if cross[t]["alpha"] > 0 else CBAD for t in TICKERS],
    hovertemplate="%{x} : %{y:.1f} %<extra></extra>"))
fig_ct.add_hline(y=100 * np.mean([cross[t]["alpha"] for t in TICKERS]),
                 line_dash="dash", line_color=C1,
                 annotation_text=f"moyenne {100 * np.mean([cross[t]['alpha'] for t in TICKERS]):+.1f} %")
fig_ct.update_layout(title="Généralisation — alpha du Multi sur le test de chaque actif",
                     yaxis_title="Alpha vs B&H (%)", **PLOTLY_LAYOUT)

# Évolution
fig_ev = go.Figure()
fig_ev.add_trace(go.Bar(name="Avant", x=["Single", "Multi"],
                        y=[100 * av["single_alpha_test"], 100 * av["multi_alpha_test"]],
                        marker_color="#7f8c8d"))
fig_ev.add_trace(go.Bar(name="Après", x=["Single", "Multi"],
                        y=[100 * ap["single_alpha_test"], 100 * ap["multi_alpha_test"]],
                        marker_color=C0))
fig_ev.update_layout(title="Alpha test avant / après la session d'amélioration",
                     yaxis_title="Alpha (%)", barmode="group", **PLOTLY_LAYOUT)


# Section walk-forward du dashboard (uniquement si le run existe)
wf_block = ""
if wf is not None:
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(name="Agent (moy. 5 actifs)", x=wf_years,
                            y=[100 * fo["mean_return"] for fo in wf["folds"]],
                            marker_color=C1))
    fig_wf.add_trace(go.Bar(name="Buy & Hold (moy.)", x=wf_years,
                            y=[100 * fo["mean_bh"] for fo in wf["folds"]],
                            marker_color=CBH))
    fig_wf.update_layout(title="Walk-forward — return par année out-of-sample (%)",
                         barmode="group", yaxis_title="Return (%)", **PLOTLY_LAYOUT)

    fig_wf_hm = go.Figure(go.Heatmap(
        z=np.clip(alpha_grid, -100, 100), x=wf_tickers,
        y=[str(y) for y in wf_years],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:+.0f} %" for v in row] for row in alpha_grid],
        texttemplate="%{text}",
        customdata=alpha_grid,
        hovertemplate="%{y} · %{x} : %{customdata:.1f} %<extra></extra>"))
    fig_wf_hm.update_layout(title="Alpha par cellule année × actif "
                                  "(couleur tronquée à ±100 %)",
                            **PLOTLY_LAYOUT)

    agg = wf["aggregate"]
    wf_block = f"""
<h2>Validation walk-forward ({len(wf_years)} réentraînements glissants)</h2>
<p class="muted">Chaque année de test est prédite par un modèle réentraîné uniquement
sur les données ANTÉRIEURES — {agg['n_cells']} cellules année×actif 100 % out-of-sample.
Alpha médian {100 * agg['median_alpha']:+.1f} %, {agg['pct_positive']:.0%} de cellules
positives (moyenne {100 * agg['mean_alpha']:+.1f} %, tirée vers le bas par les années de
très forte hausse — TSLA 2020 : B&H +576 %). Lecture : <b>l'agent a un profil défensif
régime-dépendant</b> — alpha nettement positif l'année baissière (2022 : +37,6 % en
moyenne), proche de zéro ou négatif dans les fortes hausses. Le résultat du split unique
ci-dessus doit se lire à travers ce prisme.</p>
<div class="chart">{to_div(fig_wf)}</div>
<div class="chart">{to_div(fig_wf_hm)}</div>
"""


# Section stress-tests (uniquement si evaluate.stress_report() a été lancé)
STRESS_PATH = os.path.join(ROOT, "reports", "stress_tests.json")
stress_block = ""
if os.path.exists(STRESS_PATH):
    with open(STRESS_PATH) as f:
        st = json.load(f)

    fig_fees = go.Figure()
    for (label, row), color in zip(st["fee_grid"].items(), [C0, C1]):
        fig_fees.add_trace(go.Scatter(
            x=[10000 * float(k) for k in row.keys()],
            y=[100 * v["alpha"] for v in row.values()],
            name=label, mode="lines+markers", line=dict(color=color, width=2)))
    fig_fees.add_hline(y=0, line_color="#666")
    fig_fees.update_layout(title="Sensibilité de l'alpha aux frais par trade "
                                 "(modèles entraînés à 10 bps)",
                           xaxis_title="Frais par trade (bps)",
                           yaxis_title="Alpha test (%)", **PLOTLY_LAYOUT)

    rows_ns = "".join(
        f"<tr><td>{l}</td><td>{100 * v['alpha_with_stop']:+.1f} %</td>"
        f"<td>{100 * v['alpha']:+.1f} %</td>"
        f"<td>{100 * v['stop_contribution']:+.1f} pts</td>"
        f"<td>{100 * v['max_dd']:.1f} %</td></tr>"
        for l, v in st["no_killswitch"].items()
    )
    stress_block = f"""
<h2>Stress-tests du sceptique</h2>
<p class="muted">Les deux questions qu'un desk pose en premier, mesurées à politique
fixée. 1) L'alpha du Multi survit à 3× ses frais d'entraînement (+23,7 % à 30 bps).
2) L'ablation du kill-switch révèle que ~21 points d'alpha viennent de la règle de stop
— un <b>alpha de gestion du risque</b> plus que de signal (le Multi garde +5,1 % de
« decision alpha », le Single 0).</p>
<div class="chart">{to_div(fig_fees)}</div>
<table>
<tr><th>Sans kill-switch</th><th>Alpha avec stop</th><th>Alpha sans stop</th>
<th>Contribution du stop</th><th>MaxDD sans stop</th></tr>
{rows_ns}
</table>
"""


def pct(x):
    return f"{x:+.1%}".replace("%", " %")


def num(x):
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.2f}"


rows_full = "".join(
    f"<tr><td>{name}</td><td>{pct(r['return'])}</td><td><b>{pct(r['alpha'])}</b></td>"
    f"<td>{r['max_dd']:.1%}</td><td>{num(r['sharpe'])}</td><td>{num(r['sortino'])}</td>"
    f"<td>{num(r['calmar'])}</td><td>{pct(r['cvar_95'])}</td>"
    f"<td>{rob[name]['beat_bh']}/5</td></tr>"
    for name, r in full.items()
)

bh_test = full["Single (AAPL)"]["bh"]

# ============================================================
# TEMPLATE HTML — thème « terminal de desk »
# CSS/JS en chaînes simples (pas de f-string → accolades libres),
# HTML assemblé en f-string qui les interpole.
# ============================================================

CSS = """
:root{
  --bg:#070b12; --panel:#0d1420; --panel2:#0b111c; --line:#1c2836;
  --up:#2ee6a8; --dn:#ff5d5d; --amber:#e6b23c; --blue:#5aa9ff;
  --txt:#cfd9e4; --muted:#7d8ea1; color-scheme:dark;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--txt);
  font-family:'Space Grotesk',system-ui,sans-serif;
  background-image:
    linear-gradient(rgba(46,230,168,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(46,230,168,.035) 1px, transparent 1px);
  background-size:44px 44px;
}
.mono{font-family:'IBM Plex Mono',monospace}
#progress{position:fixed;top:0;left:0;height:2px;width:0;
  background:linear-gradient(90deg,var(--up),var(--blue));z-index:99}

/* ── ticker tape ─────────────────────────────── */
.tape{overflow:hidden;border-bottom:1px solid var(--line);
  background:#0a101a;white-space:nowrap}
.tape-track{display:inline-block;padding:9px 0;
  animation:tape 28s linear infinite}
@keyframes tape{from{transform:translateX(0)}to{transform:translateX(-33.333%)}}
.tk{font-family:'IBM Plex Mono',monospace;font-size:.82rem;
  color:var(--muted);margin:0 26px;letter-spacing:.4px}
.tk b{font-weight:600}
.up{color:var(--up)} .dn{color:var(--dn)}

/* ── hero ────────────────────────────────────── */
.hero{max-width:1040px;margin:0 auto;padding:72px 20px 40px;text-align:left}
.hero .kicker{font-family:'IBM Plex Mono',monospace;font-size:.8rem;
  color:var(--up);letter-spacing:3px;text-transform:uppercase}
.hero h1{font-size:clamp(2.2rem,6vw,4rem);margin:10px 0 6px;line-height:1.02;
  letter-spacing:-1px}
.hero h1 span{color:var(--up)}
.hero .sub{color:var(--muted);max-width:640px;font-size:1.02rem;line-height:1.5}
.big-alpha{font-family:'IBM Plex Mono',monospace;
  font-size:clamp(3.4rem,10vw,6.4rem);font-weight:700;color:var(--up);
  text-shadow:0 0 34px rgba(46,230,168,.35);line-height:1;margin:26px 0 2px}
.big-alpha-label{color:var(--muted);font-size:.92rem;margin-bottom:22px}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.chip{font-family:'IBM Plex Mono',monospace;font-size:.78rem;
  border:1px solid var(--line);border-radius:999px;padding:7px 14px;
  color:var(--muted);background:rgba(13,20,32,.7)}
.chip b{color:var(--txt)}

/* ── nav ─────────────────────────────────────── */
.nav{position:sticky;top:0;z-index:50;display:flex;gap:4px;
  justify-content:center;padding:10px 8px;
  background:rgba(7,11,18,.82);backdrop-filter:blur(10px);
  border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.nav a{color:var(--muted);text-decoration:none;font-size:.86rem;
  padding:7px 14px;border-radius:8px;transition:.2s}
.nav a:hover{color:var(--up);background:rgba(46,230,168,.08)}

/* ── structure ───────────────────────────────── */
.wrap{max-width:1040px;margin:0 auto;padding:0 20px 90px}
section{padding-top:58px}
h2{font-size:1.5rem;letter-spacing:-.3px;margin:0 0 4px;
  padding-bottom:10px;border-bottom:1px solid var(--line)}
h2::before{content:"// ";color:var(--up);font-family:'IBM Plex Mono',monospace;
  font-size:1.05rem}
.muted{color:var(--muted);font-size:.92rem;line-height:1.55}

/* ── cards ───────────────────────────────────── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
  gap:14px;margin:26px 0 8px}
.card{position:relative;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:20px;transition:transform .25s, box-shadow .25s;
  overflow:hidden}
.card::before{content:"";position:absolute;inset:0 0 auto 0;height:2px;
  background:linear-gradient(90deg,var(--up),transparent)}
.card:hover{transform:translateY(-4px);box-shadow:0 14px 34px rgba(0,0,0,.45)}
.card .v{font-family:'IBM Plex Mono',monospace;font-size:1.7rem;
  font-weight:700;color:var(--up)}
.card .l{font-size:.82rem;color:var(--muted);margin-top:6px;line-height:1.45}

/* ── charts & tables ─────────────────────────── */
.chart{background:var(--panel2);border:1px solid var(--line);
  border-radius:14px;padding:10px;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:16px 0;
  font-family:'IBM Plex Mono',monospace}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;font-family:'Space Grotesk',sans-serif}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;
  letter-spacing:.6px}
tr:hover td{background:rgba(46,230,168,.04)}

/* ── timeline ────────────────────────────────── */
.tl{position:relative;margin:34px 0 0;padding-left:34px}
.tl::before{content:"";position:absolute;left:11px;top:6px;bottom:6px;width:2px;
  background:linear-gradient(var(--up),var(--blue))}
.tl-item{position:relative;margin-bottom:22px}
.tl-dot{position:absolute;left:-34px;top:2px;width:24px;height:24px;
  border-radius:50%;background:var(--bg);border:2px solid var(--up);
  color:var(--up);font-family:'IBM Plex Mono',monospace;font-size:.72rem;
  display:flex;align-items:center;justify-content:center;font-weight:700}
.tl-card{background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px 18px}
.tl-card h3{margin:0 0 6px;font-size:1.02rem}
.tl-card p{margin:0 0 10px;color:var(--muted);font-size:.9rem;line-height:1.5}
.tl-chip{font-family:'IBM Plex Mono',monospace;font-size:.74rem;
  color:var(--amber);border:1px solid rgba(230,178,60,.35);
  border-radius:6px;padding:3px 9px}

/* ── verdict book ────────────────────────────── */
.book{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:26px}
@media (max-width:760px){.book{grid-template-columns:1fr}}
.side{border-radius:14px;padding:22px;border:1px solid var(--line)}
.side.long{background:linear-gradient(180deg,rgba(46,230,168,.07),var(--panel))}
.side.short{background:linear-gradient(180deg,rgba(255,93,93,.07),var(--panel))}
.side h3{margin:0 0 14px;font-family:'IBM Plex Mono',monospace;font-size:.92rem;
  letter-spacing:2px;text-transform:uppercase}
.side.long h3{color:var(--up)} .side.short h3{color:var(--dn)}
.side ul{margin:0;padding-left:18px}
.side li{margin-bottom:10px;font-size:.92rem;line-height:1.5}

/* ── reveal on scroll ─────────────────────────
   Progressive enhancement : visible par défaut ; l'état caché n'existe
   que si JS est actif (html.js), avec fallback forcé à 1,5 s. */
html.js .reveal{opacity:0;transform:translateY(18px);
  transition:opacity .45s ease, transform .45s ease}
html.js .reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion: reduce){
  html.js .reveal{opacity:1;transform:none;transition:none}
  .tape-track{animation:none}
  html{scroll-behavior:auto}
}

a{color:var(--up)}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.82rem;line-height:1.6}
code{font-family:'IBM Plex Mono',monospace;background:var(--panel);
  border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:.84em}
"""

JS = """
// barre de progression de lecture
var bar = document.getElementById('progress');
window.addEventListener('scroll', function(){
  var h = document.documentElement;
  bar.style.width = 100*h.scrollTop/(h.scrollHeight-h.clientHeight) + '%';
});
// compteur du hero
var el = document.getElementById('heroAlpha');
var target = parseFloat(el.dataset.target), t0 = null;
function tick(ts){
  if(!t0) t0 = ts;
  var p = Math.min((ts-t0)/1400, 1);
  var eased = 1 - Math.pow(1-p, 3);
  el.textContent = '+' + (target*eased).toFixed(1) + ' %';
  if(p < 1) requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
// apparition au scroll
var obs = new IntersectionObserver(function(entries){
  entries.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); obs.unobserve(e.target); } });
}, {threshold: .05, rootMargin: '0px 0px 120px 0px'});
document.querySelectorAll('.reveal').forEach(function(n){ obs.observe(n); });
// filet de sécurité : tout révéler après 1,5 s quoi qu'il arrive
setTimeout(function(){
  document.querySelectorAll('.reveal').forEach(function(n){ n.classList.add('in'); });
}, 1500);
"""

# ── ticker tape (cross-ticker, répété 3× pour boucle continue) ──
tape_items = "".join(
    f'<span class="tk">{t} '
    f'<b class="{"up" if cross[t]["alpha"] > 0 else "dn"}">'
    f'{cross[t]["alpha"]:+.1%}</b></span>'.replace("%", " %")
    for t in TICKERS
) + (f'<span class="tk">B&amp;H AAPL <b class="dn">{bh_test:+.1%}</b></span>'
     .replace("%", " %"))
tape = f'<div class="tape"><div class="tape-track">{tape_items}{tape_items}{tape_items}</div></div>'

# ── timeline du cheminement (miroir du rapport) ──
timeline_steps = [
    ("0", "Départ — travail solo",
     "PPO sur 2018-2023. Résultats instables (±15 pts selon le tirage), alpha "
     "d'entraînement délirant, Multi toujours long : trois symptômes à expliquer.",
     "3/5 fenêtres > B&H"),
    ("1", "Tester l'économie du simulateur",
     "67 tests : comptabilité exacte des trades, anti look-ahead, kill-switch. "
     "Bug trouvé immédiatement : les frais d'ouverture du short n'étaient pas débités.",
     "un simulateur faux rend tout faux"),
    ("2", "Réparer le protocole d'évaluation",
     "Observations normalisées à l'éval, frais alignés sur l'entraînement, épisode "
     "déterministe sur le split complet, gel en cash après stop → chiffres reproductibles.",
     "variance ±15 pts → 0"),
    ("3", "Plus de cycles de marché",
     "Réentraînement 2010→2023 (2,6× plus de données) ; test 2021-2022 bull + bear "
     "au lieu d'un bear pur — fin du biais de période le plus grossier.",
     "13 ans, 3 corrections en train"),
    ("4", "Le bug décisif : validation corrompue",
     "Les épisodes de validation traversaient les frontières entre tickers (faux krachs "
     "de -70 %) : le best model était tiré au sort. Ablation contrôlée, toutes choses "
     "égales par ailleurs : alpha AAPL -2,4 % → +27,5 %.",
     "+30 pts, effet causal isolé"),
    ("5", "Walk-forward : la douche froide utile",
     "5 réentraînements, 25 cellules année×actif out-of-sample. Médiane -2 % : le "
     "+27,5 % était en partie un biais de période. Profil réel : défensif, "
     "régime-dépendant, return absolu positif 4 années sur 5.",
     "un backtest = un tirage"),
    ("6", "Stress-tests du sceptique",
     "Grille de frais 0→30 bp (le Multi tient : +23,7 %) et ablation du kill-switch : "
     "~21 pts d'alpha viennent de la règle de stop. Alpha de gestion du risque, "
     "pas de signal.",
     "signal ≠ règle de risque"),
    ("7", "Acte 3 : la barre d'erreur, puis l'échec instructif",
     "5 seeds d'entraînement : l'alpha AAPL est une loterie (±21 pts, positif 2/5) "
     "mais la moyenne cross-actifs tient (+14,5 % ± 6,5, positive 5/5). Les features "
     "de régime, testées contre cette bande : échec cohérent — voir le régime ne "
     "suffit pas, c'est la récompense qui doit inciter à s'en servir.",
     "juge de paix : ±6,5 pts"),
]
timeline_html = "".join(
    f'<div class="tl-item reveal"><div class="tl-dot">{n}</div>'
    f'<div class="tl-card"><h3>{t}</h3><p>{d}</p>'
    f'<span class="tl-chip">{c}</span></div></div>'
    for n, t, d, c in timeline_steps
)

alpha_multi_pct = 100 * full["Multi (5 tickers)"]["alpha"]

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DRL Portfolio — Martin Chassaing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<script>document.documentElement.classList.add('js')</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div id="progress"></div>
{tape}

<header class="hero">
  <div class="kicker">Deep Reinforcement Learning × Marchés — projet de recherche personnel</div>
  <h1>Un agent PPO face au <span>Buy &amp; Hold</span></h1>
  <p class="sub">Construit, cassé, corrigé et stress-testé. Ce dashboard montre les
  résultats <em>et</em> leurs limites — parce qu'un backtest sans ses contre-arguments
  est un argument de vente, pas de la recherche.</p>
  <div class="big-alpha" id="heroAlpha" data-target="{alpha_multi_pct:.1f}">+0.0 %</div>
  <div class="big-alpha-label">alpha du Multi vs B&amp;H — test 2021-2022 jamais vu à
  l'entraînement (B&amp;H : {pct(bh_test)}) · à lire avec le walk-forward ↓</div>
  <div class="chips">
    <span class="chip"><b>{ap["multi_tickers_positifs_5"]}/5</b> actifs à alpha &gt; 0</span>
    <span class="chip"><b>{rob["Multi (5 tickers)"]["beat_bh"]}/5 · {rob["Single (AAPL)"]["beat_bh"]}/5</b> fenêtres &gt; B&amp;H (Multi · Single)</span>
    <span class="chip"><b>25</b> cellules OOS walk-forward</span>
    <span class="chip"><b>+14,5 % ± 6,5</b> cross-actifs sur 5 seeds</span>
    <span class="chip"><b>74</b> tests pytest</span>
    <span class="chip"><b>0,1 %</b> de frais/trade</span>
  </div>
</header>

<nav class="nav">
  <a href="#perf">Performance</a>
  <a href="#walkforward">Walk-forward</a>
  <a href="#stress">Stress-tests</a>
  <a href="#chrono">Cheminement</a>
  <a href="#verdict">Verdict</a>
</nav>

<div class="wrap">

<section id="perf" class="reveal">
<h2>Performance — test set (fév. 2021 → déc. 2022)</h2>
<p class="muted">Évaluation déterministe sur le split complet, observations normalisées
avec les statistiques d'entraînement, gel en cash après stop drawdown (horizons
comparables). Marché de test : {pct(bh_test)} pour le B&amp;H.</p>
<div class="chart">{to_div(fig_eq)}</div>
<div class="chart">{to_div(fig_dd)}</div>
<table>
<tr><th>Modèle</th><th>Return</th><th>Alpha</th><th>MaxDD</th><th>Sharpe</th>
<th>Sortino</th><th>Calmar</th><th>CVaR 95</th><th>Robustesse</th></tr>
{rows_full}
<tr><td>Buy &amp; Hold AAPL (réf.)</td><td>{pct(bh_test)}</td><td>—</td>
<td>30,3 %</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</table>
<div class="chart">{to_div(fig_ct)}</div>
<div class="chart">{to_div(fig_ev)}</div>
</section>

<section id="walkforward" class="reveal">
{wf_block}
</section>

<section id="stress" class="reveal">
{stress_block}
</section>

<section id="chrono" class="reveal">
<h2>Le cheminement — 7 étapes, chacune née d'un symptôme</h2>
<p class="muted">Le détail complet (raisonnements, maths, prédictions faites avant
chaque run) est dans le <a href="../reports/rapport_drl_portfolio.pdf">rapport PDF</a>.</p>
<div class="tl">{timeline_html}</div>
</section>

<section id="verdict" class="reveal">
<h2>Verdict honnête</h2>
<div class="book">
  <div class="side long">
    <h3>▲ Long — ce qui tient</h3>
    <ul>
      <li>Les deux agents battent le B&amp;H sur le test complet
          (Single {pct(full["Single (AAPL)"]["alpha"])}, Multi
          {pct(full["Multi (5 tickers)"]["alpha"])}) et sur 5/5 fenêtres aléatoires.</li>
      <li>Généralisation réelle : alpha positif sur les 5 actifs
          (moyenne {pct(ap["multi_cross_alpha"])}) — des règles apprises,
          pas une trajectoire mémorisée.</li>
      <li>L'edge survit aux frais : +23,7 % d'alpha à 30 bps, trois fois le
          niveau d'entraînement.</li>
      <li>Return absolu positif 4 années sur 5 en walk-forward, drawdown borné
          à 25 % quand le B&amp;H fait -30 % : cohérent comme brique
          <em>absolute return</em> défensive.</li>
      <li>La revendication robuste : <b>+14,5 % ± 6,5 d'alpha cross-actifs,
          positive pour 5 seeds d'entraînement sur 5</b> (test 2021-22).</li>
      <li>Protocole verrouillé : splits chronologiques, scaler anti look-ahead,
          éval déterministe, 74 tests, ablations contrôlées, bande de bruit
          inter-seeds comme juge de paix.</li>
    </ul>
  </div>
  <div class="side short">
    <h3>▼ Short — ce qui ne tient pas</h3>
    <ul>
      <li>Profil régime-dépendant : brillant en bear (2022 : +38 %), à la traîne
          en bull, catastrophique face au rebond 2020 (stoppé dans le krach,
          il rate +143 %). Médiane walk-forward : -2 %.</li>
      <li>~21 points d'alpha 2022 viennent du kill-switch, pas du timing :
          discipline de risque systématique aidée d'un signal modeste
          (+5,1 % sans le stop pour le Multi, 0 pour le Single).</li>
      <li>Le +27,5 % du split unique était en partie un biais de période —
          2021-2022 est le régime rêvé d'un agent défensif.</li>
      <li>L'alpha mono-actif est une loterie de seed (AAPL : de -11 % à +44 % selon
          l'initialisation) — seul le cross-actifs est défendable.</li>
      <li>Les features de régime (dist. plus-haut 1 an, SMA 200) testées contre la
          bande de bruit : <b>échec cohérent</b> — l'incitation (récompense) prime sur
          l'information (observation). Prochain levier : position continue + récompense
          sensible au régime.</li>
      <li>Sharpe &lt; 1, cinq méga-caps survivantes, emprunt de titres gratuit :
          pas un produit, un laboratoire.</li>
    </ul>
  </div>
</div>
</section>

<footer>
Projet éducatif de <b>Martin Chassaing</b> (IMT Atlantique × Université Paris Dauphine)
— rien ici n'est un conseil d'investissement.<br>
Rapport complet : <code>reports/rapport_drl_portfolio.pdf</code> ·
dashboard interactif : <code>streamlit run dashboard.py</code> ·
reproduire : <code>python train.py && python evaluate.py && python walk_forward.py</code> ·
page générée par <code>reports/build_assets.py</code>.
</footer>

</div>
<script>{JS}</script>
</body></html>
"""

with open(os.path.join(DOCS_DIR, "index.html"), "w") as f:
    f.write(html)
print("   docs/index.html ✅")

print("4/4 — Terminé.")
