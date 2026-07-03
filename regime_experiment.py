# regime_experiment.py
# ============================================================
# Acte 3b — l'agent peut-il distinguer les régimes de marché ?
#
# Diagnostic du walk-forward : l'agent applique la même prudence dans un
# krach durable et dans un rebond en V, car ses 10 jours de features
# court-terme ne contiennent pas l'information de régime (« underfitting
# de représentation », cf. rapport).
#
# Traitement testé : 2 features de régime dans l'observation (52 → 72) :
#   - dist_high_252 : position vs plus-haut 1 an (drawdown de l'actif)
#   - trend_200     : position vs moyenne mobile 200 jours
#
# Prédiction consignée AVANT le run : amélioration du fold 2020 (le pire,
# -146 %) sans dégrader 2022 ; succès si la médiane walk-forward dépasse
# la bande de bruit inter-seeds mesurée par seed_robustness.py.
#
# Protocole : un levier à la fois — mêmes données 2010-2023, même config
# d'entraînement, même seed 42, seuls les inputs changent. Le modèle est
# entraîné dans un dossier SÉPARÉ : le headline n'est promu que si le
# régime gagne hors bande de seed.
#
# Usage :
#   .venv/bin/python regime_experiment.py     # ~35 min (train + walk-forward)
# Sortie :
#   models/ppo_multi_regime/                  # le modèle candidat
#   reports/walk_forward_regime.json          # walk-forward avec régime
#   reports/regime_experiment.json            # synthèse
# ============================================================
import os
import io
import json
import contextlib
from dataclasses import replace

import numpy as np

from data_loader import (FEATURES, REGIME_FEATURES, DataConfig,
                         load_data, load_multi_ticker_data)
from evaluate import EVAL_ENV_CFG, evaluate_full
from train import TrainConfig, set_all_seeds, train
from walk_forward import WalkForwardConfig, run_walk_forward

TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"]

# Config env avec observation élargie (log_returns reste en position 0)
ENV_REGIME = replace(EVAL_ENV_CFG, features=tuple(FEATURES + REGIME_FEATURES))


def main():
    # ── 1. Entraînement du candidat (mêmes réglages que le headline) ──
    print("=" * 60)
    print("  1/3 — Entraînement Multi + features de régime (800k, seed 42)")
    print("=" * 60)
    set_all_seeds(42)

    cfg_data = DataConfig(tickers=TICKERS, regime_features=True)
    with contextlib.redirect_stdout(io.StringIO()):
        train_m, val_m, _ = load_multi_ticker_data(cfg_data)

    cfg_train = TrainConfig(
        total_timesteps = 800_000,
        seed            = 42,
        model_name      = "ppo_multi_regime",
        save_dir        = "models/ppo_multi_regime/",
        log_dir         = "logs/multi_regime/",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        train(train_m, val_m, ENV_REGIME, cfg_train)
    model_path = "models/ppo_multi_regime/best_model.zip"
    print("  ✅ modèle candidat sauvegardé")

    # ── 2. Évaluation au protocole headline ────────────────────────
    print("\n  2/3 — Évaluation full-split + cross-ticker")
    cross = {}
    for ticker in TICKERS:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, test_data, _ = load_data(
                DataConfig(ticker=ticker, regime_features=True))
        r = evaluate_full(model_path, test_data, ENV_REGIME)
        cross[ticker] = {"alpha": r["alpha"], "return": r["return"],
                         "max_dd": r["max_dd"]}
        print(f"  {ticker:<6} return {r['return']:+7.1%} | alpha {r['alpha']:+7.1%}")

    alphas = [c["alpha"] for c in cross.values()]
    print(f"  → cross-ticker moyen {np.mean(alphas):+.1%} | "
          f"positifs {sum(a > 0 for a in alphas)}/5 "
          f"(headline : +17.2 %, 5/5)")

    # ── 3. Walk-forward avec régime (le vrai juge) ──────────────────
    print("\n  3/3 — Walk-forward avec features de régime (5 folds)")
    wf_cfg = WalkForwardConfig(
        regime_features = True,
        out_dir         = "models/walk_forward_regime",
        json_path       = "reports/walk_forward_regime.json",
    )
    wf = run_walk_forward(wf_cfg, env_cfg=ENV_REGIME)

    # ── Synthèse ───────────────────────────────────────────────────
    out = {
        "aapl_test": cross["AAPL"],
        "cross_ticker": cross,
        "cross_alpha_mean": float(np.mean(alphas)),
        "cross_positive": int(sum(a > 0 for a in alphas)),
        "walk_forward_aggregate": wf["aggregate"],
        "walk_forward_by_year": {
            str(f["test_year"]): f["mean_alpha"] for f in wf["folds"]
        },
    }
    with open("reports/regime_experiment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  → reports/regime_experiment.json")


if __name__ == "__main__":
    main()
