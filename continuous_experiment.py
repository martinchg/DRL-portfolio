# continuous_experiment.py
# ============================================================
# Acte 5 — position continue et aversion au risque dépendante du régime.
#
# Diagnostic (Acte 3 / walk-forward) : le profil tout-ou-rien se fait
# kill-switcher à pleine taille dans le krach 2020 (fold : alpha -146 %)
# puis gèle et rate le rebond. Le levier n'est pas l'observation (falsifié
# à l'Acte 3b) mais le DIMENSIONNEMENT et l'incitation à dérisquer.
#
# Deux bras pré-enregistrés, un delta chacun :
#   A : w ∈ [-1, 1] continu, récompense INCHANGÉE   → isole l'effet sizing
#   B : A + pénalité λ·σ̂·|w| (λ = 0.1, UNE valeur)  → paie le dérisquage
#       quand le marché est nerveux, AVANT le kill-switch
#
# Critères GO/NO-GO figés avant les runs (PREDICTION.md) :
#   C1 : alpha cross-ticker moyen (test) ≥ +10.7 %  (headline 17.2 − bande 6.5)
#   C2 : fold 2020 walk-forward > -50 %             (headline : -146 %)
#   C3 : fold 2022 walk-forward > +20 %             (headline : +37.6 %)
#   C4 : médiane walk-forward ≥ -2 %                (headline)
#
# Usage :
#   .venv/bin/python continuous_experiment.py --dry   # plomberie, ~5 min
#   .venv/bin/python continuous_experiment.py         # réel, ~2 h
# Sorties :
#   models/ppo_continuous/ , models/ppo_continuous_risk/
#   reports/walk_forward_continuous[_risk].json
#   reports/continuous_experiment.json
# ============================================================
import argparse
import contextlib
import io
import json
import os
from dataclasses import replace

import numpy as np

from data_loader import DataConfig, _download, load_data, load_multi_ticker_data
from evaluate import EVAL_ENV_CFG, evaluate_full
from train import TrainConfig, set_all_seeds, train
from walk_forward import WalkForwardConfig, run_walk_forward

TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"]

# ent_coef 0.05 → 0.01 en continu : sur une politique gaussienne, le bonus
# d'entropie discret gonfle l'écart-type → jitter de position → frais.
ENT_COEF_CONT = 0.01

ENV_A = replace(EVAL_ENV_CFG, continuous=True)
ENV_B = replace(EVAL_ENV_CFG, continuous=True, risk_aversion=0.1)

CRITERES = {
    "C1_cross_alpha_min" : 0.107,
    "C2_fold2020_min"    : -0.50,
    "C3_fold2022_min"    : 0.20,
    "C4_mediane_min"     : -0.02,
}


def _model_path(save_dir: str, name: str) -> str:
    best = os.path.join(save_dir, "best_model.zip")
    return best if os.path.exists(best) else os.path.join(save_dir, f"{name}.zip")


def run_arm(label, env_cfg, save_dir, wf_dir, wf_json,
            ts_train, ts_wf, raw_data):
    print("=" * 60)
    print(f"  BRAS {label}")
    print("=" * 60)
    set_all_seeds(42)

    # ── 1. Entraînement (mêmes données que le headline) ──────────
    with contextlib.redirect_stdout(io.StringIO()):
        train_m, val_m, _ = load_multi_ticker_data(DataConfig(tickers=TICKERS))
    name = os.path.basename(save_dir.rstrip("/"))
    cfg_train = TrainConfig(
        total_timesteps = ts_train,
        seed            = 42,
        ent_coef        = ENT_COEF_CONT,
        model_name      = name,
        save_dir        = save_dir,
        log_dir         = os.path.join("logs", name) + os.sep,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        train(train_m, val_m, env_cfg, cfg_train)
    model_path = _model_path(save_dir, name)
    print(f"  ✅ candidat : {model_path}")

    # ── 2. Cross-ticker (protocole headline) ─────────────────────
    cross = {}
    for ticker in TICKERS:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, test_data, _ = load_data(DataConfig(ticker=ticker))
        r = evaluate_full(model_path, test_data, env_cfg)
        cross[ticker] = {"alpha": r["alpha"], "return": r["return"],
                         "max_dd": r["max_dd"], "n_trades": r["n_trades"],
                         "long_pct": r["long_pct"], "short_pct": r["short_pct"]}
        print(f"  {ticker:<6} return {r['return']:+7.1%} | alpha {r['alpha']:+7.1%}"
              f" | trades {r['n_trades']}")
    cross_mean = float(np.mean([c["alpha"] for c in cross.values()]))
    print(f"  → cross-ticker moyen {cross_mean:+.1%} (headline +17.2 %)")

    # ── 3. Walk-forward (le vrai juge) ───────────────────────────
    wf = run_walk_forward(
        WalkForwardConfig(timesteps=ts_wf, ent_coef=ENT_COEF_CONT,
                          out_dir=wf_dir, json_path=wf_json),
        raw_data=raw_data, env_cfg=env_cfg,
    )
    by_year = {str(f["test_year"]): f["mean_alpha"] for f in wf["folds"]}

    # ── 4. Verdict C1-C4 ─────────────────────────────────────────
    verdict = {
        "C1_cross_alpha" : {"value": cross_mean,
                            "pass": cross_mean >= CRITERES["C1_cross_alpha_min"]},
        "C2_fold_2020"   : {"value": by_year.get("2020"),
                            "pass": (by_year.get("2020") is not None
                                     and by_year["2020"] > CRITERES["C2_fold2020_min"])},
        "C3_fold_2022"   : {"value": by_year.get("2022"),
                            "pass": (by_year.get("2022") is not None
                                     and by_year["2022"] > CRITERES["C3_fold2022_min"])},
        "C4_mediane_wf"  : {"value": wf["aggregate"]["median_alpha"],
                            "pass": (wf["aggregate"]["median_alpha"]
                                     >= CRITERES["C4_mediane_min"])},
    }
    verdict["all_pass"] = all(v["pass"] for v in verdict.values()
                              if isinstance(v, dict))
    for k, v in verdict.items():
        if isinstance(v, dict):
            val = v["value"]
            print(f"  {'✅' if v['pass'] else '❌'} {k} : "
                  f"{val:+.1%}" if val is not None else f"  ❓ {k}")
    return {"cross_ticker": cross, "cross_alpha_mean": cross_mean,
            "walk_forward_by_year": by_year,
            "walk_forward_aggregate": wf["aggregate"],
            "verdict": verdict}


def main(dry: bool):
    suffix   = "_dry" if dry else ""
    ts_train = 5_000 if dry else 800_000
    ts_wf    = 5_000 if dry else 400_000

    # Télécharge une seule fois pour les deux walk-forwards
    print("→ Téléchargement des prix (une fois pour les deux bras)…")
    raw_data = {t: _download(t, "2010-01-01", "2023-01-01") for t in TICKERS}

    arm_a = run_arm(
        "A — sizing continu seul", ENV_A,
        f"models/ppo_continuous{suffix}/",
        f"models/walk_forward_continuous{suffix}",
        f"reports/walk_forward_continuous{suffix}.json",
        ts_train, ts_wf, raw_data)

    arm_b = run_arm(
        "B — sizing + aversion risque λ=0.1", ENV_B,
        f"models/ppo_continuous_risk{suffix}/",
        f"models/walk_forward_continuous_risk{suffix}",
        f"reports/walk_forward_continuous_risk{suffix}.json",
        ts_train, ts_wf, raw_data)

    out = {
        "dry_run"  : dry,
        "criteres" : CRITERES,
        "config"   : {"ent_coef": ENT_COEF_CONT, "lambda_B": 0.1,
                      "timesteps_train": ts_train, "timesteps_wf": ts_wf},
        "bras_A"   : arm_a,
        "bras_B"   : arm_b,
        "headline_reference": {"cross_alpha": 0.172, "fold_2020": -1.464,
                               "fold_2022": 0.376, "mediane_wf": -0.020},
    }
    path = f"reports/continuous_experiment{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {path}")
    print(f"  A all_pass : {arm_a['verdict']['all_pass']} | "
          f"B all_pass : {arm_b['verdict']['all_pass']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true",
                        help="plomberie bout-en-bout à 5k steps (~5 min)")
    args = parser.parse_args()
    main(args.dry)
