# seed_robustness.py
# ============================================================
# Question : le +27,5 % d'alpha du Multi est-il un coup de chance de seed ?
#
# Le RL profond est notoirement sensible au hasard d'initialisation
# (poids du réseau, ordre d'exploration). Un seul entraînement = un
# tirage ; sans barre d'erreur inter-seeds, impossible de dire si une
# amélioration future est réelle ou si c'est du bruit.
#
# Protocole : réentraîner le Multi à l'identique (données 2010-2023,
# 800k steps, config finale) en ne changeant QUE le seed, puis évaluer
# chaque modèle exactement comme le headline (full-split AAPL test +
# généralisation cross-ticker).
#
# Usage :
#   .venv/bin/python seed_robustness.py     # ~35 min (4 entraînements)
# Sortie :
#   models/seeds/multi_<seed>/              # un modèle par seed
#   reports/seed_robustness.json
# ============================================================
import os
import io
import json
import contextlib

import numpy as np

from data_loader import DataConfig, load_data, load_multi_ticker_data
from evaluate import EVAL_ENV_CFG, evaluate_full
from train import TrainConfig, set_all_seeds, train

TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"]
NEW_SEEDS = [7, 123, 777, 2024]          # le seed 42 = modèle headline existant
HEADLINE = ("42 (headline)", "models/ppo_multi/best_model.zip")


def evaluate_model(model_path):
    """Même protocole que le headline : full-split AAPL + cross-ticker."""
    cross = {}
    for ticker in TICKERS:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, test_data, _ = load_data(DataConfig(ticker=ticker))
        cross[ticker] = evaluate_full(model_path, test_data, EVAL_ENV_CFG)["alpha"]
    return {
        "aapl_alpha": cross["AAPL"],
        "cross_alpha_mean": float(np.mean(list(cross.values()))),
        "cross_positive": int(sum(1 for a in cross.values() if a > 0)),
        "cross_detail": cross,
    }


def main():
    cfg_data = DataConfig(tickers=TICKERS)
    with contextlib.redirect_stdout(io.StringIO()):
        train_m, val_m, _ = load_multi_ticker_data(cfg_data)

    results = {}

    for seed in NEW_SEEDS:
        print(f"\n{'='*60}\n  SEED {seed} — entraînement Multi (800k steps)\n{'='*60}")
        set_all_seeds(seed)
        cfg_train = TrainConfig(
            total_timesteps = 800_000,
            seed            = seed,
            model_name      = "ppo_multi",
            save_dir        = f"models/seeds/multi_{seed}/",
            log_dir         = f"models/seeds/multi_{seed}/logs/",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(train_m, val_m, EVAL_ENV_CFG, cfg_train)

        r = evaluate_model(f"models/seeds/multi_{seed}/best_model.zip")
        results[str(seed)] = r
        print(f"  alpha AAPL {r['aapl_alpha']:+.1%} | "
              f"cross moyen {r['cross_alpha_mean']:+.1%} | "
              f"positifs {r['cross_positive']}/5")

    # Le headline (seed 42) évalué au même protocole
    print(f"\n  SEED {HEADLINE[0]} — évaluation du modèle existant")
    results["42"] = evaluate_model(HEADLINE[1])
    r = results["42"]
    print(f"  alpha AAPL {r['aapl_alpha']:+.1%} | "
          f"cross moyen {r['cross_alpha_mean']:+.1%} | "
          f"positifs {r['cross_positive']}/5")

    # ── Agrégat ────────────────────────────────────────────
    aapl   = [r["aapl_alpha"] for r in results.values()]
    crossm = [r["cross_alpha_mean"] for r in results.values()]
    agg = {
        "n_seeds": len(results),
        "aapl_alpha_mean": float(np.mean(aapl)),
        "aapl_alpha_std": float(np.std(aapl)),
        "aapl_alpha_min": float(np.min(aapl)),
        "aapl_alpha_max": float(np.max(aapl)),
        "aapl_positive_seeds": int(sum(1 for a in aapl if a > 0)),
        "cross_mean_mean": float(np.mean(crossm)),
        "cross_mean_std": float(np.std(crossm)),
    }
    out = {"per_seed": results, "aggregate": agg}
    os.makedirs("reports", exist_ok=True)
    with open("reports/seed_robustness.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ROBUSTESSE AU SEED — {agg['n_seeds']} seeds")
    print(f"  Alpha AAPL   : {agg['aapl_alpha_mean']:+.1%} ± {agg['aapl_alpha_std']:.1%} "
          f"(min {agg['aapl_alpha_min']:+.1%}, max {agg['aapl_alpha_max']:+.1%})")
    print(f"  Seeds alpha>0 sur AAPL : {agg['aapl_positive_seeds']}/{agg['n_seeds']}")
    print(f"  Cross-ticker moyen : {agg['cross_mean_mean']:+.1%} ± {agg['cross_mean_std']:.1%}")
    print(f"  → reports/seed_robustness.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
