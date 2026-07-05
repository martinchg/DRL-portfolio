# validate_diffusion.py
"""
Validation pré-enregistrée du générateur de scénarios (Phase 1 diffusion).

Deux modes :
  python validate_diffusion.py --calibration-only
      Bandes réelles + baselines, AVANT tout entraînement DDPM.
      Fige le protocole (pré-enregistrement) dans reports/diffusion_validation.json.
  python validate_diffusion.py
      Réutilise la calibration existante si compatible (les bandes ne bougent
      PAS après coup — c'est le contrat du pré-enregistrement), ajoute le DDPM
      et rend le verdict É1-É6.

Sorties : reports/diffusion_validation.json, reports/figures/diffusion_*.png
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_multi_ticker_data, DataConfig
from diffusion.dataset import (
    WindowConfig, extract_windows, segment_returns,
    compute_norm, normalize, denormalize,
)
from diffusion.metrics import (
    ValidationCriteria, acf, ks_statistic,
    nn_distances_cross, nn_distances_loo,
    build_real_bands, generator_summary, judge_criteria,
)
from diffusion.baselines import (
    sample_gaussian_iid, sample_bootstrap_iid,
    fit_garch_per_segment, sample_garch_fitted,
)
from diffusion.discriminative import DiscConfig, discriminative_score

REPORT_JSON = "reports/diffusion_validation.json"
FIG_DIR     = "reports/figures"
MODEL_DIR   = "models/diffusion_v1"      # défaut, remplaçable par --model-dir
SEED        = 42

COLORS = {
    "real"          : "#666666",
    "gaussian_iid"  : "#1f77b4",
    "bootstrap_iid" : "#ff7f0e",
    "garch"         : "#2ca02c",
    "ddpm"          : "#d62728",
}
LABELS = {
    "real"          : "Réel (train 2010-2019)",
    "gaussian_iid"  : "B0 — gaussienne i.i.d.",
    "bootstrap_iid" : "B1 — bootstrap i.i.d.",
    "garch"         : "B2 — GARCH(1,1)-t",
    "ddpm"          : "DDPM",
}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    raise TypeError(f"Non sérialisable : {type(o)}")


# ============================================================
# DONNÉES RÉELLES
# ============================================================
def load_real(win_cfg: WindowConfig):
    train_df, _, _ = load_multi_ticker_data(DataConfig())
    windows, meta = extract_windows(train_df, win_cfg)
    seg_rets = segment_returns(train_df)
    return windows, meta, seg_rets


# ============================================================
# CALIBRATION (pré-enregistrement)
# ============================================================
def run_calibration(real, meta, seg_rets, win_cfg, crit, disc_cfg):
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    mu, sigma = compute_norm(real)
    real_norm = normalize(real, mu, sigma)
    groups = meta["segment_id"].values
    pooled = np.concatenate(list(seg_rets.values()))

    print("→ Bandes de calibration (K tirages par blocs sous le réel)…")
    bands = build_real_bands(real, crit, rng=np.random.default_rng(SEED), meta=meta)

    print("→ Distances NN leave-one-out du réel (référence É6)…")
    nn_loo = nn_distances_loo(real_norm, meta, win_cfg.window)
    nn_ref = {
        "p10"    : float(np.percentile(nn_loo, crit.nn_ref_percentile)),
        "p25"    : float(np.percentile(nn_loo, 25)),
        "median" : float(np.percentile(nn_loo, 50)),
    }

    # ---- É1 : le réel jugé contre ses propres bandes ----
    sub_idx = rng.choice(len(real), crit.n_eval_windows, replace=False)
    real_sub_summary = generator_summary(real[sub_idx], crit.max_lag)
    real_verdict = judge_criteria(real_sub_summary, bands, crit)

    # ---- Sanité du juge : entrelacé ≈ 0.5 / dérive temporelle = contexte ----
    # Splits SYMÉTRIQUES (les deux camps par blocs purgés) — un split
    # asymétrique fait apprendre l'époque au juge, pas le réalisme.
    # block/embargo remis à l'échelle du sous-échantillonnage [::step].
    print("→ Juge discriminatif : sanité (entrelacé) et dérive (split temporel)…")
    from dataclasses import replace
    step = max(1, len(real) // 2000)          # ~1000 fenêtres par camp
    cfg_rr = replace(disc_cfg,
                     block=max(8, disc_cfg.block // (2 * step)),
                     embargo=max(4, disc_cfg.embargo // (2 * step)))
    even, odd = np.arange(0, len(real), 2)[::step], np.arange(1, len(real), 2)[::step]
    disc_interleaved = discriminative_score(
        real_norm[even], real_norm[odd],
        real_groups=groups[even], synth_groups=groups[odd], cfg=cfg_rr)

    first_half, second_half = [], []
    for sid in np.unique(groups):
        idx = np.flatnonzero(groups == sid)
        half = len(idx) // 2
        first_half.append(idx[:half])
        second_half.append(idx[half:])
    fh = np.concatenate(first_half)[::step]
    sh = np.concatenate(second_half)[::step]
    disc_timesplit = discriminative_score(
        real_norm[sh], real_norm[fh],
        real_groups=groups[sh], synth_groups=groups[fh], cfg=cfg_rr)

    # ---- Baselines ----
    print("→ Baselines : B0 gaussienne, B1 bootstrap, B2 GARCH(1,1)-t…")
    n = crit.n_eval_windows
    gens = {
        "gaussian_iid": sample_gaussian_iid(
            n, win_cfg.window, float(pooled.mean()), float(pooled.std()),
            rng=np.random.default_rng(SEED + 1)),
        "bootstrap_iid": sample_bootstrap_iid(
            n, win_cfg.window, pooled, rng=np.random.default_rng(SEED + 2)),
    }
    garch_params = fit_garch_per_segment(seg_rets)
    seg_weights = meta.groupby("segment_id").size().to_dict()
    gens["garch"] = sample_garch_fitted(
        n, win_cfg.window, garch_params, seg_weights,
        rng=np.random.default_rng(SEED + 3))

    baselines = {}
    for name, wins in gens.items():
        print(f"   … {name} (métriques + juge discriminatif)")
        baselines[name] = evaluate_generator(
            wins, real, real_norm, groups, bands, crit, disc_cfg,
            mu, sigma, nn_ref["p10"], rng)
    garch_acc = baselines["garch"]["disc"]["acc"]

    # É5 rétroactif pour les baselines maintenant que l'ancre GARCH existe
    for name in baselines:
        v = judge_criteria(
            {k: baselines[name]["summary"][k]
             for k in ("moments", "acf_r", "acf_absr", "acf_absr_sum")},
            bands, crit,
            disc_acc=baselines[name]["disc"]["acc"], garch_disc_acc=garch_acc,
            nn_median=baselines[name]["nn_median"], nn_real_ref=nn_ref["p10"])
        baselines[name]["verdict"] = v

    # ---- Contrôle du protocole (É1) ----
    # É2 du GARCH : REPORTÉ mais non exigé — le MLE donne ν̂ < 4 sur les
    # actions daily (4ᵉ moment infini) et α²·(K_z−1) > 1−(α+β)² : la kurtosis
    # inconditionnelle du GARCH fitté est structurellement infinie. Le contrôle
    # des marginales reste assuré par B1 (bootstrap = marginales exactes).
    checks = {
        "reel_passe_E2"        : real_verdict["E2_queues"]["pass"],
        "reel_passe_E3"        : real_verdict["E3_acf_parasite"]["pass"],
        "reel_passe_E4"        : real_verdict["E4_clustering"]["pass"],
        "juge_entrelace_hasard": abs(disc_interleaved["acc"] - 0.5) <= 0.10,
        "B0_echoue_E2"         : not baselines["gaussian_iid"]["verdict"]["E2_queues"]["pass"],
        "B0_echoue_E4"         : not baselines["gaussian_iid"]["verdict"]["E4_clustering"]["pass"],
        "B1_passe_E2"          : baselines["bootstrap_iid"]["verdict"]["E2_queues"]["pass"],
        "B1_echoue_E4"         : not baselines["bootstrap_iid"]["verdict"]["E4_clustering"]["pass"],
        "garch_passe_E3"       : baselines["garch"]["verdict"]["E3_acf_parasite"]["pass"],
        "garch_passe_E4"       : baselines["garch"]["verdict"]["E4_clustering"]["pass"],
    }
    protocol_ok = all(checks.values())

    calibration = {
        "phase"     : "calibration",
        "created"   : time.strftime("%Y-%m-%d %H:%M:%S"),
        "duree_s"   : round(time.time() - t0, 1),
        "seed"      : SEED,
        "config"    : {
            "window"          : win_cfg.window,
            "stride"          : win_cfg.stride,
            "n_real_windows"  : int(len(real)),
            "windows_par_segment": {str(k): int(v) for k, v in seg_weights.items()},
            "source"          : "split TRAIN RL (70 % par ticker, 5 tickers, 2010→)",
        },
        "criteria"  : crit.to_dict(),
        "disc_config": {"hidden": disc_cfg.hidden, "lr": disc_cfg.lr,
                        "epochs": disc_cfg.epochs, "batch": disc_cfg.batch},
        "norm"      : {"mu": mu, "sigma": sigma},
        "bands"     : bands,
        "nn_loo_real": nn_ref,
        "real"      : {"summary": real_sub_summary, "verdict_E1": real_verdict},
        "real_vs_real_disc": {
            "entrelace_sanite" : disc_interleaved,
            "split_temporel_derive": disc_timesplit,
        },
        "garch_params_par_segment": {str(k): v for k, v in garch_params.items()},
        "baselines" : baselines,
        "protocole_E1": {"checks": checks, "ok": protocol_ok},
    }
    return calibration, gens


def evaluate_generator(wins, real, real_norm, groups, bands, crit, disc_cfg,
                       mu, sigma, nn_ref_p10, rng):
    """Résumé + juge discriminatif + NN + KS pour un générateur (verdict à part)."""
    wins_norm = normalize(wins, mu, sigma)
    disc = discriminative_score(real_norm, wins_norm, real_groups=groups, cfg=disc_cfg)
    nn = nn_distances_cross(wins_norm, real_norm)
    sub = real[rng.choice(len(real), min(len(real), 1000), replace=False)]
    return {
        "summary"   : generator_summary(wins, crit.max_lag),
        "disc"      : disc,
        "nn_median" : float(np.median(nn)),
        "nn_p10"    : float(np.percentile(nn, 10)),
        "ks_vs_real": ks_statistic(wins, sub),
        "verdict"   : None,   # rempli quand l'ancre GARCH est connue
    }


# ============================================================
# DDPM (mode complet)
# ============================================================
def evaluate_ddpm(calibration, real, real_norm, meta, win_cfg, crit, disc_cfg,
                  model_dir=MODEL_DIR):
    """Échantillonne le DDPM entraîné et le juge contre la calibration FIGÉE."""
    from diffusion.ddpm import load_ddpm     # import paresseux (torch model)
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"→ Chargement du DDPM ({model_dir}) sur {device}…")
    ddpm, ckpt_cfg = load_ddpm(model_dir, device=device)

    mu, sigma = calibration["norm"]["mu"], calibration["norm"]["sigma"]
    n = crit.n_eval_windows

    print(f"→ Échantillonnage de {n} fenêtres (T={ckpt_cfg['T']} pas)…")
    t0 = time.time()
    z = ddpm.sample(n, win_cfg.window, seed=SEED)
    wins = denormalize(z, mu, sigma).astype(np.float32)
    print(f"   {time.time() - t0:.0f}s")

    rng = np.random.default_rng(SEED)
    groups = meta["segment_id"].values
    bands = calibration["bands"]
    res = evaluate_generator(wins, real, real_norm, groups, bands, crit, disc_cfg,
                             mu, sigma, calibration["nn_loo_real"]["p10"], rng)
    res["verdict"] = judge_criteria(
        {k: res["summary"][k] for k in ("moments", "acf_r", "acf_absr", "acf_absr_sum")},
        bands, crit,
        disc_acc=res["disc"]["acc"],
        garch_disc_acc=calibration["baselines"]["garch"]["disc"]["acc"],
        nn_median=res["nn_median"],
        nn_real_ref=calibration["nn_loo_real"]["p10"])
    res["checkpoint_config"] = ckpt_cfg
    return res, wins


# ============================================================
# FIGURES
# ============================================================
def make_figures(named, real, real_norm, calibration, crit, win_cfg):
    """named : dict nom → fenêtres brutes (N, L), sans 'real'."""
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    bands = calibration["bands"]
    sub = real[rng.choice(len(real), 1000, replace=False)]

    # ---- 1. Marginales : densité log + QQ ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(-0.12, 0.12, 121)
    axes[0].hist(sub.ravel(), bins=bins, density=True, log=True,
                 color=COLORS["real"], alpha=0.45, label=LABELS["real"])
    for name, w in named.items():
        h, e = np.histogram(w.ravel(), bins=bins, density=True)
        centers = (e[:-1] + e[1:]) / 2
        axes[0].plot(centers, np.where(h > 0, h, np.nan),
                     color=COLORS[name], lw=1.4, label=LABELS[name])
    axes[0].set_title("Rendements poolés (densité, échelle log)")
    axes[0].set_xlabel("log-rendement journalier")
    axes[0].legend(fontsize=8)

    qs = np.linspace(0.001, 0.999, 199)
    q_real = np.quantile(sub.ravel(), qs)
    for name, w in named.items():
        axes[1].plot(q_real, np.quantile(w.ravel(), qs),
                     color=COLORS[name], lw=1.4, label=LABELS[name])
    lim = [q_real.min(), q_real.max()]
    axes[1].plot(lim, lim, "k--", lw=1, alpha=0.6)
    axes[1].set_title("QQ-plot vs réel (0.1 % → 99.9 %)")
    axes[1].set_xlabel("quantiles réels")
    axes[1].set_ylabel("quantiles générés")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diffusion_marginals.png", dpi=130)
    plt.close(fig)

    # ---- 2. ACF r et |r| avec bandes réelles ----
    lags = np.arange(1, crit.max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [
        (axes[0], "acf_r", "ACF des rendements (É3 : doit rester ≈ 0)"),
        (axes[1], "acf_absr", "ACF de |r| (É4 : volatility clustering)"),
    ]:
        b = bands[key]
        ax.fill_between(lags, b["lo"], b["hi"], color=COLORS["real"], alpha=0.25,
                        label="bande réelle 2.5-97.5 %")
        ax.plot(lags, b["median"], color=COLORS["real"], lw=1.2, ls=":",
                label="médiane réelle")
        for name, w in named.items():
            series = acf(np.abs(w) if key == "acf_absr" else w, crit.max_lag)
            ax.plot(lags, series, color=COLORS[name], lw=1.4, marker="o",
                    ms=3, label=LABELS[name])
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("lag (jours)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diffusion_acf.png", dpi=130)
    plt.close(fig)

    # ---- 3. Trajectoires cumulées ----
    panels = [("real", sub)] + list(named.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8),
                             sharey=True)
    for ax, (name, w) in zip(np.atleast_1d(axes), panels):
        idx = rng.choice(len(w), 30, replace=False)
        paths = np.cumsum(w[idx], axis=1).T * 100
        ax.plot(paths, color=COLORS[name], alpha=0.3, lw=0.8)
        ax.set_title(LABELS[name], fontsize=9)
        ax.set_xlabel("jours")
    np.atleast_1d(axes)[0].set_ylabel("rendement cumulé (%)")
    fig.suptitle("30 trajectoires par générateur (fenêtres de "
                 f"{win_cfg.window} jours)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diffusion_trajectories.png", dpi=130)
    plt.close(fig)

    # ---- 4. Distances NN (anti-mémorisation) ----
    mu_, sigma_ = calibration["norm"]["mu"], calibration["norm"]["sigma"]
    fig, axes = plt.subplots(1, 2 if "ddpm" in named else 1,
                             figsize=(12 if "ddpm" in named else 6.5, 4.5))
    ax0 = np.atleast_1d(axes)[0]
    p10 = calibration["nn_loo_real"]["p10"]
    for name, w in named.items():
        d = nn_distances_cross(normalize(w, mu_, sigma_), real_norm)
        ax0.hist(d, bins=60, density=True, histtype="step",
                 color=COLORS[name], label=LABELS[name])
    ax0.axvline(p10, color="k", ls="--", lw=1,
                label=f"p10 NN réel LOO = {p10:.1f} (seuil É6)")
    ax0.set_title("Distance au plus proche voisin réel")
    ax0.set_xlabel("distance L2 (fenêtres normalisées)")
    ax0.legend(fontsize=7)

    if "ddpm" in named:
        w = named["ddpm"]
        wn = normalize(w, mu_, sigma_)
        d = nn_distances_cross(wn, real_norm)
        closest = np.argsort(d)[:3]
        ax1 = np.atleast_1d(axes)[1]
        for i, ci in enumerate(closest):
            # retrouve le voisin réel exact
            d2 = ((real_norm - wn[ci]) ** 2).sum(axis=1)
            nn_idx = int(np.argmin(d2))
            off = i * 8
            ax1.plot(np.cumsum(w[ci]) * 100 + off, color=COLORS["ddpm"], lw=1.2,
                     label="DDPM" if i == 0 else None)
            ax1.plot(np.cumsum(denormalize(real_norm[nn_idx], mu_, sigma_)) * 100 + off,
                     color=COLORS["real"], lw=1.2, ls="--",
                     label="voisin réel" if i == 0 else None)
        ax1.set_title("3 paires les plus proches (décalées verticalement)")
        ax1.set_xlabel("jours")
        ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diffusion_nn_distance.png", dpi=130)
    plt.close(fig)

    # ---- 5. Stats par fenêtre ----
    from diffusion.metrics import window_stats
    stats_real = window_stats(sub)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [("vol", "vol réalisée (σ des rendements)", 60),
             ("max_dd", "max drawdown intra-fenêtre", 60),
             ("terminal", "rendement cumulé terminal", 60)]
    for ax, (key, title, nb) in zip(axes, specs):
        ax.hist(stats_real[key], bins=nb, density=True, color=COLORS["real"],
                alpha=0.45, label=LABELS["real"])
        for name, w in named.items():
            s = window_stats(w)
            ax.hist(s[key], bins=nb, density=True, histtype="step",
                    color=COLORS[name], label=LABELS[name])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Ce que l'environnement RL « voit » d'une fenêtre", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diffusion_window_stats.png", dpi=130)
    plt.close(fig)

    print(f"→ 5 figures écrites dans {FIG_DIR}/diffusion_*.png")


# ============================================================
# MAIN
# ============================================================
def _calibration_compatible(existing, win_cfg, crit):
    try:
        return (existing["config"]["window"] == win_cfg.window
                and existing["config"]["stride"] == win_cfg.stride
                and existing["criteria"] == crit.to_dict()
                and existing["seed"] == SEED)
    except (KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-only", action="store_true",
                        help="fige bandes + baselines sans toucher au DDPM")
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help="dossier d'expérience du DDPM à juger")
    args = parser.parse_args()

    win_cfg = WindowConfig()
    crit = ValidationCriteria()
    disc_cfg = DiscConfig()

    print("→ Chargement des données réelles (split TRAIN RL)…")
    real, meta, seg_rets = load_real(win_cfg)
    mu, sigma = compute_norm(real)
    real_norm = normalize(real, mu, sigma)
    print(f"   {len(real)} fenêtres de {win_cfg.window} jours "
          f"({len(seg_rets)} segments)")

    # ---- Calibration : réutilisée si déjà figée et compatible ----
    existing = None
    if os.path.exists(REPORT_JSON):
        with open(REPORT_JSON) as f:
            existing = json.load(f)

    named_gens = {}
    if existing is not None and _calibration_compatible(existing, win_cfg, crit):
        print("→ Calibration existante réutilisée (bandes FIGÉES, contrat de "
              "pré-enregistrement).")
        report = existing
        if not args.calibration_only:
            # re-matérialise les baselines pour les figures (mêmes seeds)
            pooled = np.concatenate(list(seg_rets.values()))
            named_gens["gaussian_iid"] = sample_gaussian_iid(
                crit.n_eval_windows, win_cfg.window,
                float(pooled.mean()), float(pooled.std()),
                rng=np.random.default_rng(SEED + 1))
            named_gens["bootstrap_iid"] = sample_bootstrap_iid(
                crit.n_eval_windows, win_cfg.window, pooled,
                rng=np.random.default_rng(SEED + 2))
            garch_params = {int(k): v for k, v in
                            report["garch_params_par_segment"].items()}
            seg_weights = meta.groupby("segment_id").size().to_dict()
            named_gens["garch"] = sample_garch_fitted(
                crit.n_eval_windows, win_cfg.window, garch_params, seg_weights,
                rng=np.random.default_rng(SEED + 3))
    else:
        if existing is not None:
            print("⚠️  Calibration existante incompatible avec la config → recalcul.")
        report, named_gens = run_calibration(real, meta, seg_rets,
                                             win_cfg, crit, disc_cfg)

    # ---- DDPM (mode complet) ----
    if not args.calibration_only:
        if not os.path.exists(os.path.join(args.model_dir, "checkpoint.pt")):
            raise SystemExit(
                f"Pas de checkpoint dans {args.model_dir} — entraîner d'abord "
                "(python train_diffusion.py) ou lancer --calibration-only.")
        ddpm_res, ddpm_wins = evaluate_ddpm(report, real, real_norm, meta,
                                            win_cfg, crit, disc_cfg,
                                            model_dir=args.model_dir)
        ddpm_res["model_dir"] = args.model_dir
        report["ddpm"] = ddpm_res
        report["phase"] = "complet"
        named_gens["ddpm"] = ddpm_wins

        v = ddpm_res["verdict"]
        print("\n════════ VERDICT DDPM (critères pré-enregistrés) ════════")
        for key in ("E2_queues", "E3_acf_parasite", "E4_clustering",
                    "E5_discriminatif", "E6_memorisation"):
            if key in v:
                print(f"   {'✅' if v[key]['pass'] else '❌'} {key}")
        print(f"   → {'GO' if v['all_pass'] else 'NO-GO'} "
              "(branchement RL Phase 2)" )

    # ---- Sorties ----
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"→ Rapport écrit : {REPORT_JSON}")

    make_figures(named_gens, real, real_norm, report, crit, win_cfg)

    if report["phase"] == "calibration":
        ok = report["protocole_E1"]["ok"]
        print(f"\n════════ PROTOCOLE (É1) : {'✅ valide' if ok else '❌ CASSÉ'} ════════")
        for k, val in report["protocole_E1"]["checks"].items():
            print(f"   {'✅' if val else '❌'} {k}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
