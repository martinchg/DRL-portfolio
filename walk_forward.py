# walk_forward.py
# ============================================================
# Validation WALK-FORWARD (anchored / expanding window)
#
# Pourquoi ce module existe :
#   Un backtest sur UN split temporel est UN tirage — sa performance dépend
#   de la période de test choisie (2022 bear ≠ 2019 bull). Le walk-forward
#   est le standard du backtesting sérieux : on réentraîne le modèle en ne
#   voyant que le passé, on le teste sur l'année suivante, et on répète en
#   avançant. Chaque année de test est 100 % out-of-sample, et l'agrégation
#   sur plusieurs années/actifs donne une DISTRIBUTION de performances,
#   pas un chiffre unique.
#
# Schéma (test_years = [2018, ..., 2022]) :
#   fold 2018 : train+val 2010→2017  |  test 2018
#   fold 2019 : train+val 2010→2018  |  test 2019
#   ...
#   fold 2022 : train+val 2010→2021  |  test 2022
#
# Chaque fold réentraîne un modèle multi-ticker complet (config identique
# au pipeline principal, timesteps réduits) et l'évalue sur l'année de test
# de CHAQUE ticker → grille folds × tickers d'alphas out-of-sample.
#
# Usage :
#   .venv/bin/python walk_forward.py          # ~25 min (5 folds × 400k steps)
# Sortie :
#   models/walk_forward/fold_<year>/          # modèles par fold
#   reports/walk_forward.json                 # résultats agrégés
# ============================================================
import os
import io
import json
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import (DataConfig, _download, _build_features,
                         _scale_features, active_features)
from environment import EnvConfig
from evaluate import EVAL_ENV_CFG, evaluate_full


# ============================================================
# CONFIG
# ============================================================
@dataclass
class WalkForwardConfig:
    tickers    : List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "SPY", "TSLA"
    ])
    start_date : str = "2010-01-01"
    # Années entièrement out-of-sample. Chaque année = un fold.
    test_years : List[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022])
    # Part du train réservée à la validation (sélection du best model)
    val_ratio  : float = 0.15
    # Timesteps par fold — réduits vs pipeline principal (800k) car 5 folds.
    timesteps  : int = 400_000
    # Features de régime (Acte 3) : dist_high_252 + trend_200 dans l'observation
    regime_features : bool = False
    # Override d'ent_coef pour les folds (Acte 5 : 0.01 en continu — sur une
    # gaussienne, le 0.05 discret gonfle l'écart-type → bruit de position).
    # None = valeur par défaut de TrainConfig.
    ent_coef   : Optional[float] = None
    out_dir    : str = "models/walk_forward"
    json_path  : str = "reports/walk_forward.json"


@dataclass
class Fold:
    test_year  : int
    train_start: str
    test_start : str   # = fin du train/val
    test_end   : str


# ============================================================
# LOGIQUE DE DÉCOUPE (pure → testée unitairement)
# ============================================================
def make_folds(cfg: WalkForwardConfig) -> List[Fold]:
    """Folds anchored : le train commence toujours à start_date (fenêtre croissante)."""
    folds = []
    for year in cfg.test_years:
        folds.append(Fold(
            test_year   = year,
            train_start = cfg.start_date,
            test_start  = f"{year}-01-01",
            test_end    = f"{year + 1}-01-01",
        ))
    return folds


def split_fold_data(features: pd.DataFrame, fold: Fold, val_ratio: float,
                    feature_cols: Optional[list] = None):
    """
    Découpe un DataFrame de features (index datetime) pour un fold.

    Règle anti look-ahead : le scaler est fitté UNIQUEMENT sur la partie
    train (avant la validation), puis appliqué à val et test. Les features
    elles-mêmes sont des fenêtres backward-looking → les calculer sur toute
    la série avant découpe n'introduit aucune information future.

    Retourne (train, val, test) scalés.
    """
    pre_test = features.loc[features.index < fold.test_start]
    test     = features.loc[(features.index >= fold.test_start)
                            & (features.index < fold.test_end)]

    n_pre     = len(pre_test)
    train_end = int(n_pre * (1.0 - val_ratio))

    # Scale : fit sur train uniquement, transform sur tout le fold
    fold_data = pd.concat([pre_test, test])
    fold_data, _ = _scale_features(fold_data, train_end, feature_cols)

    train = fold_data.iloc[:train_end].copy()
    val   = fold_data.iloc[train_end:n_pre].copy()
    test  = fold_data.iloc[n_pre:].copy()
    return train, val, test


def build_fold_datasets(raw: Dict[str, pd.DataFrame], fold: Fold,
                        val_ratio: float, data_cfg: DataConfig):
    """
    Construit les datasets multi-ticker d'un fold :
      - train/val concaténés avec segment_id (épisodes confinés par ticker)
      - tests par ticker (dict), chacun scalé avec le scaler de SON fold/ticker
    """
    trains, vals, tests = [], [], {}
    cols = active_features(data_cfg)
    for seg_id, (ticker, df) in enumerate(raw.items()):
        features = _build_features(df, data_cfg)
        train, val, test = split_fold_data(features, fold, val_ratio, cols)

        for split in (train, val):
            split["ticker"]     = ticker
            split["segment_id"] = seg_id
        trains.append(train)
        vals.append(val)
        tests[ticker] = test

    train_combined = pd.concat(trains, ignore_index=True)
    val_combined   = pd.concat(vals,   ignore_index=True)
    return train_combined, val_combined, tests


# ============================================================
# RUN
# ============================================================
def run_walk_forward(cfg: WalkForwardConfig = WalkForwardConfig(),
                     raw_data: Optional[Dict[str, pd.DataFrame]] = None,
                     env_cfg: EnvConfig = EVAL_ENV_CFG) -> dict:
    """
    Exécute le walk-forward complet.

    raw_data : dict {ticker: DataFrame['price']} — injectable pour les tests
    (données synthétiques, pas de réseau). Si None, télécharge via yfinance.
    """
    # Import ici : train.py charge SB3/torch, inutile pour les tests de découpe
    from train import train, TrainConfig, set_all_seeds

    data_cfg = DataConfig(regime_features=cfg.regime_features)
    end_date = f"{max(cfg.test_years) + 1}-01-01"

    if raw_data is None:
        raw_data = {t: _download(t, cfg.start_date, end_date) for t in cfg.tickers}

    folds   = make_folds(cfg)
    results = {"config": {"tickers": cfg.tickers, "test_years": cfg.test_years,
                          "timesteps": cfg.timesteps, "start_date": cfg.start_date},
               "folds": []}

    for fold in folds:
        print(f"\n{'='*60}\n  FOLD {fold.test_year} — train {fold.train_start} → "
              f"{fold.test_start} | test {fold.test_year}\n{'='*60}")

        set_all_seeds(42)
        train_data, val_data, tests = build_fold_datasets(
            raw_data, fold, cfg.val_ratio, data_cfg
        )
        print(f"  Train : {len(train_data)} jours | Val : {len(val_data)} jours")

        fold_dir = os.path.join(cfg.out_dir, f"fold_{fold.test_year}") + os.sep
        overrides = {} if cfg.ent_coef is None else {"ent_coef": cfg.ent_coef}
        cfg_train = TrainConfig(
            total_timesteps = cfg.timesteps,
            model_name      = "ppo_wf",
            save_dir        = fold_dir,
            log_dir         = os.path.join(fold_dir, "logs") + os.sep,
            **overrides,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(train_data, val_data, env_cfg, cfg_train)

        model_path = os.path.join(fold_dir, "best_model.zip")
        if not os.path.exists(model_path):           # eval_freq jamais atteint (tests tiny)
            model_path = os.path.join(fold_dir, "ppo_wf.zip")

        per_ticker = {}
        for ticker, test_data in tests.items():
            r = evaluate_full(model_path, test_data, env_cfg)
            per_ticker[ticker] = {
                "return"  : r["return"],  "bh"     : r["bh"],
                "alpha"   : r["alpha"],   "sharpe" : r["sharpe"],
                "max_dd"  : r["max_dd"],
                "terminated_early": r["terminated_early"],
            }
            print(f"  {ticker:<6} return {r['return']:>+7.1%} | "
                  f"B&H {r['bh']:>+7.1%} | alpha {r['alpha']:>+7.1%}")

        alphas = [m["alpha"] for m in per_ticker.values()]
        results["folds"].append({
            "test_year"  : fold.test_year,
            "per_ticker" : per_ticker,
            "mean_alpha" : float(np.mean(alphas)),
            "mean_return": float(np.mean([m["return"] for m in per_ticker.values()])),
            "mean_bh"    : float(np.mean([m["bh"] for m in per_ticker.values()])),
        })
        print(f"  → alpha moyen du fold : {np.mean(alphas):+.1%}")

    all_alphas = [m["alpha"] for f in results["folds"] for m in f["per_ticker"].values()]
    results["aggregate"] = {
        "mean_alpha"       : float(np.mean(all_alphas)),
        "median_alpha"     : float(np.median(all_alphas)),
        "std_alpha"        : float(np.std(all_alphas)),
        "pct_positive"     : float(np.mean([a > 0 for a in all_alphas])),
        "n_cells"          : len(all_alphas),
        "worst_cell_alpha" : float(np.min(all_alphas)),
        "best_cell_alpha"  : float(np.max(all_alphas)),
    }

    os.makedirs(os.path.dirname(cfg.json_path), exist_ok=True)
    with open(cfg.json_path, "w") as f:
        json.dump(results, f, indent=2)

    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD TERMINÉ — {agg['n_cells']} cellules année×actif OOS")
    print(f"  Alpha moyen   : {agg['mean_alpha']:+.1%} (médiane {agg['median_alpha']:+.1%})")
    print(f"  Cellules alpha>0 : {agg['pct_positive']:.0%}")
    print(f"  Pire / meilleure : {agg['worst_cell_alpha']:+.1%} / {agg['best_cell_alpha']:+.1%}")
    print(f"  → {cfg.json_path}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    run_walk_forward()
