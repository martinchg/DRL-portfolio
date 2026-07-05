# diffusion/dataset.py
"""
Extraction des fenêtres de log-rendements pour le générateur.

Deux pièges gérés ici, symétriques des pièges connus du repo :
- les rendements sont RECALCULÉS depuis la colonne `price` brute — la colonne
  `log_returns` du pipeline data_loader est passée au RobustScaler, la réutiliser
  fausserait la distribution apprise (miroir inverse du bug « obs non normalisées ») ;
- aucune fenêtre ne traverse une frontière de `segment_id` — sinon on fabrique de
  faux krachs aux jonctions entre tickers (même piège que les épisodes RL).
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    # [GESTIONNAIRE] Longueur L des fenêtres générées (jours de bourse).
    # 256 ≈ 1 an : puissance de 2 (U-Net), assez long pour exhiber le
    # volatility clustering et servir de segment d'épisode RL en Phase 2.
    window : int = 256

    # [TECHNIQUE] Pas entre deux fenêtres glissantes. 1 = recouvrement maximal
    # (~10 000 fenêtres sur le train 5 tickers) — standard pour l'entraînement
    # génératif ; les métriques anti-mémorisation excluent les voisins chevauchants.
    stride : int = 1


# ============================================================
# RENDEMENTS
# ============================================================
def log_returns_from_price(prices: np.ndarray) -> np.ndarray:
    """log(p_t / p_{t-1}) depuis les prix BRUTS (jamais la colonne scalée)."""
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 2:
        raise ValueError("Il faut au moins 2 prix pour des rendements.")
    if np.any(prices <= 0):
        raise ValueError("Prix négatif ou nul : colonne 'price' corrompue.")
    return np.diff(np.log(prices))


def segment_returns(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Rendements par segment (un segment = un ticker en multi-ticker).
    Sans colonne segment_id : segment unique 0.
    """
    if 'segment_id' in df.columns:
        return {
            int(sid): log_returns_from_price(
                df.loc[df['segment_id'] == sid, 'price'].values
            )
            for sid in sorted(df['segment_id'].unique())
        }
    return {0: log_returns_from_price(df['price'].values)}


# ============================================================
# FENÊTRAGE
# ============================================================
def extract_windows(
    df  : pd.DataFrame,
    cfg : WindowConfig = WindowConfig(),
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fenêtres glissantes de rendements, confinées à leur segment.

    Returns
    -------
    windows : (N, L) float32 — rendements bruts (non normalisés)
    meta    : DataFrame (segment_id, ticker, start) aligné sur windows ;
              start = indice du premier rendement dans son segment.
    """
    seg_rets = segment_returns(df)

    tickers = {}
    if 'ticker' in df.columns and 'segment_id' in df.columns:
        tickers = (
            df.groupby('segment_id')['ticker'].first().to_dict()
        )

    wins, meta = [], []
    for sid, rets in seg_rets.items():
        for start in range(0, len(rets) - cfg.window + 1, cfg.stride):
            wins.append(rets[start:start + cfg.window])
            meta.append((sid, tickers.get(sid, str(sid)), start))

    if not wins:
        raise ValueError(
            f"Aucune fenêtre : segments trop courts pour window={cfg.window}."
        )

    windows = np.stack(wins).astype(np.float32)
    meta = pd.DataFrame(meta, columns=['segment_id', 'ticker', 'start'])
    return windows, meta


# ============================================================
# NORMALISATION GLOBALE
# ============================================================
# Un seul couple (μ, σ) poolé sur tout le train — PAS par fenêtre ni par ticker :
# les écarts de vol entre tickers (TSLA ~3× SPY) font partie de la distribution
# à apprendre ; normaliser par fenêtre écraserait les régimes de volatilité.

def compute_norm(windows: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(windows)), float(np.std(windows))


def normalize(windows: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (windows - mu) / sigma


def denormalize(z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return z * sigma + mu


def save_norm(path: str, mu: float, sigma: float, cfg: WindowConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'mu': mu, 'sigma': sigma, **asdict(cfg)}, f, indent=2)


def load_norm(path: str) -> Tuple[float, float, WindowConfig]:
    with open(path) as f:
        d = json.load(f)
    return d['mu'], d['sigma'], WindowConfig(window=d['window'], stride=d['stride'])


# ============================================================
# PIPELINE COMPLET (réseau)
# ============================================================
def load_train_windows(
    cfg      : WindowConfig = WindowConfig(),
    data_cfg = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fenêtres du split TRAIN du RL uniquement (≈ 2010→2019, 5 tickers poolés).
    Zéro fuite vers val/test : condition pour brancher la Phase 2 proprement.
    """
    from data_loader import load_multi_ticker_data, DataConfig
    data_cfg = data_cfg or DataConfig()
    train, _, _ = load_multi_ticker_data(data_cfg)
    return extract_windows(train, cfg)
