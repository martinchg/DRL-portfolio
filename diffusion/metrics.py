# diffusion/metrics.py
"""
Protocole de validation des samples : faits stylisés + calibration sur le réel.

Principe : aucun seuil arbitraire. Chaque métrique reçoit une bande
d'acceptation = distribution d'échantillonnage de la MÊME métrique calculée
sur des tirages de fenêtres RÉELLES (K tirages de N fenêtres). Un générateur
est jugé contre ces bandes ; le réel-vs-réel doit passer par construction (É1).

Toutes les métriques travaillent sur des fenêtres (N, L) de rendements BRUTS
(non normalisés), symétriquement pour le réel et le synthétique.
"""
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps


# ============================================================
# CRITÈRES PRÉ-ENREGISTRÉS (GO / NO-GO)
# ============================================================
@dataclass
class ValidationCriteria:
    """
    Figés AVANT l'entraînement du DDPM et écrits dans le JSON de calibration.
    É1 (réel-vs-réel passe tout) est vérifié par le script de validation.
    """
    # É2 — queues : kurtosis excès poolée ≥ ce plancher ET dans la bande réelle
    kurtosis_min      : float = 1.0
    # É3 — autocorrélation parasite : ACF(r) dans la bande sur ces lags
    acf_r_lags        : Tuple[int, ...] = (1, 2, 3, 4, 5)
    # É4 — clustering : ACF(|r|) lag 1 > 0 et dans la bande ; somme lags 1..max_lag dans la bande
    max_lag           : int = 20
    # É5 — réalisme global : acc discriminative ≤ acc GARCH + marge
    disc_margin       : float = 0.05
    # É6 — mémorisation : médiane NN synthétique ≥ ce percentile des NN LOO réels
    nn_ref_percentile : float = 10.0
    # Bandes de calibration
    band_lo           : float = 2.5
    band_hi           : float = 97.5
    n_eval_windows    : int   = 1000
    n_band_draws      : int   = 200

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuple → list : le dict doit être comparable à sa version JSON
        # (contrat de réutilisation de la calibration figée)
        d['acf_r_lags'] = list(d['acf_r_lags'])
        return d


# ============================================================
# MÉTRIQUES ÉLÉMENTAIRES
# ============================================================
def acf(windows: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    ACF moyenne par fenêtre (estimateur biaisé standard, normalisé par c0),
    moyennée sur les fenêtres. Shape (max_lag,), lags 1..max_lag.

    Calculée PAR fenêtre puis moyennée : concaténer les fenêtres créerait des
    artefacts aux jonctions (même logique que le confinement par segment).
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("windows doit être (N, L)")
    L = x.shape[1]
    if max_lag >= L:
        raise ValueError(f"max_lag={max_lag} >= L={L}")

    x = x - x.mean(axis=1, keepdims=True)
    c0 = (x * x).sum(axis=1) / L                      # (N,)
    valid = c0 > 1e-18                                # fenêtres non constantes

    out = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        ck = (x[:, :-k] * x[:, k:]).sum(axis=1) / L   # (N,)
        rho = np.zeros(len(ck))
        rho[valid] = ck[valid] / c0[valid]
        out[k - 1] = rho.mean()
    return out


def pooled_moments(windows: np.ndarray) -> Dict[str, float]:
    """Moments et quantiles des rendements poolés (fenêtres aplaties)."""
    r = np.asarray(windows, dtype=np.float64).ravel()
    qs = [0.001, 0.01, 0.05, 0.95, 0.99, 0.999]
    return {
        'mean'            : float(r.mean()),
        'std'             : float(r.std()),
        'skew'            : float(sps.skew(r)),
        'kurtosis_excess' : float(sps.kurtosis(r, fisher=True)),
        **{f'q_{q}': float(np.quantile(r, q)) for q in qs},
    }


def ks_statistic(windows_a: np.ndarray, windows_b: np.ndarray,
                 max_points: int = 100_000,
                 rng: Optional[np.random.Generator] = None) -> float:
    """Stat KS entre rendements poolés (p-value sans objet : données dépendantes)."""
    rng = rng or np.random.default_rng(0)
    a = np.asarray(windows_a, dtype=np.float64).ravel()
    b = np.asarray(windows_b, dtype=np.float64).ravel()
    if len(a) > max_points:
        a = rng.choice(a, max_points, replace=False)
    if len(b) > max_points:
        b = rng.choice(b, max_points, replace=False)
    return float(sps.ks_2samp(a, b).statistic)


def window_stats(windows: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Stats PAR fenêtre — ce que l'environnement RL « voit » d'un épisode :
    vol réalisée, max drawdown du chemin de prix, rendement cumulé terminal.
    """
    r = np.asarray(windows, dtype=np.float64)
    prices = np.exp(np.cumsum(r, axis=1))
    peak = np.maximum.accumulate(prices, axis=1)
    dd = 1.0 - prices / peak
    return {
        'vol'      : r.std(axis=1),
        'max_dd'   : dd.max(axis=1),
        'terminal' : r.sum(axis=1),
    }


# ============================================================
# ANTI-MÉMORISATION : DISTANCES AU PLUS PROCHE VOISIN
# ============================================================
def nn_distances_cross(
    queries : np.ndarray,
    pool    : np.ndarray,
    block   : int = 512,
) -> np.ndarray:
    """
    Distance L2 de chaque fenêtre de `queries` à sa plus proche fenêtre de
    `pool`. Fenêtres attendues NORMALISÉES (même μ/σ des deux côtés).
    """
    q = np.asarray(queries, dtype=np.float64)
    p = np.asarray(pool, dtype=np.float64)
    p_sq = (p * p).sum(axis=1)                        # (M,)
    out = np.empty(len(q))
    for i in range(0, len(q), block):
        qb = q[i:i + block]
        d2 = (qb * qb).sum(axis=1)[:, None] + p_sq[None, :] - 2.0 * qb @ p.T
        out[i:i + block] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def nn_distances_loo(
    windows    : np.ndarray,
    meta       : pd.DataFrame,
    window_len : int,
    block      : int = 512,
) -> np.ndarray:
    """
    Distance NN leave-one-out ENTRE fenêtres réelles, en excluant les voisins
    chevauchants (même segment et |Δstart| < window_len) — sans cette exclusion,
    le stride 1 rend la distance NN triviale (fenêtre décalée d'un jour).
    C'est la référence « à quelle distance le réel est-il de lui-même ». (É6)
    """
    w = np.asarray(windows, dtype=np.float64)
    seg = meta['segment_id'].values
    start = meta['start'].values
    w_sq = (w * w).sum(axis=1)

    out = np.empty(len(w))
    for i in range(0, len(w), block):
        wb = w[i:i + block]
        d2 = (wb * wb).sum(axis=1)[:, None] + w_sq[None, :] - 2.0 * wb @ w.T
        same_seg = seg[i:i + block, None] == seg[None, :]
        overlap = np.abs(start[i:i + block, None] - start[None, :]) < window_len
        d2[same_seg & overlap] = np.inf              # inclut soi-même
        out[i:i + block] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


# ============================================================
# BANDES DE CALIBRATION (distribution d'échantillonnage sous le réel)
# ============================================================
def _draw_indices(
    n           : int,
    sample_size : int,
    rng         : np.random.Generator,
    meta        : Optional[pd.DataFrame] = None,
    run         : int = 256,
) -> np.ndarray:
    """
    Indices d'un tirage de calibration.

    Sans meta : tirage i.i.d. avec remise (fenêtres supposées indépendantes).
    Avec meta : tirage par RUNS contigus de `run` fenêtres au sein d'un segment.
    Les fenêtres stride-1 se chevauchent : des tirages i.i.d. rééchantillonnent
    presque la même série → variance d'échantillonnage écrasée (bandes
    faussement étroites — le protocole v1 rejetait le bootstrap à tort).
    Les runs contigus incluent/excluent des régimes ENTIERS → variance honnête.
    """
    if meta is None:
        return rng.choice(n, size=sample_size, replace=True)

    seg_ids = meta['segment_id'].values
    segments = [np.flatnonzero(seg_ids == s) for s in np.unique(seg_ids)]
    sizes = np.array([len(s) for s in segments], dtype=np.float64)
    probs = sizes / sizes.sum()

    parts, count = [], 0
    while count < sample_size:
        seg = segments[rng.choice(len(segments), p=probs)]
        start = rng.integers(0, max(1, len(seg) - run))
        take = seg[start:start + run]
        parts.append(take)
        count += len(take)
    return np.concatenate(parts)[:sample_size]


def sampling_band(
    metric_fn   : Callable[[np.ndarray], np.ndarray],
    windows     : np.ndarray,
    n_draws     : int = 200,
    sample_size : int = 1000,
    rng         : Optional[np.random.Generator] = None,
    q_lo        : float = 2.5,
    q_hi        : float = 97.5,
    meta        : Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    """
    K tirages de `sample_size` fenêtres réelles → K valeurs de la métrique →
    bande percentile. metric_fn peut renvoyer un scalaire ou un vecteur
    (ex : ACF par lag → bande par lag). Voir _draw_indices pour le tirage
    par blocs quand meta est fourni.
    """
    rng = rng or np.random.default_rng(0)
    n = len(windows)
    draws = []
    for _ in range(n_draws):
        idx = _draw_indices(n, sample_size, rng, meta)
        draws.append(np.atleast_1d(np.asarray(metric_fn(windows[idx]), dtype=np.float64)))
    draws = np.stack(draws)                           # (K, D)
    return {
        'lo'     : np.percentile(draws, q_lo, axis=0),
        'hi'     : np.percentile(draws, q_hi, axis=0),
        'median' : np.percentile(draws, 50.0, axis=0),
    }


def build_real_bands(
    real_windows : np.ndarray,
    crit         : ValidationCriteria,
    rng          : Optional[np.random.Generator] = None,
    meta         : Optional[pd.DataFrame] = None,
) -> Dict[str, dict]:
    """Toutes les bandes nécessaires aux critères É2-É4, prêtes pour le JSON."""
    rng = rng or np.random.default_rng(42)
    kw = dict(n_draws=crit.n_band_draws, sample_size=crit.n_eval_windows,
              q_lo=crit.band_lo, q_hi=crit.band_hi, meta=meta)

    bands = {
        'kurtosis_excess': sampling_band(
            lambda w: pooled_moments(w)['kurtosis_excess'], real_windows, rng=rng, **kw),
        'acf_r': sampling_band(
            lambda w: acf(w, crit.max_lag), real_windows, rng=rng, **kw),
        'acf_absr': sampling_band(
            lambda w: acf(np.abs(w), crit.max_lag), real_windows, rng=rng, **kw),
        'acf_absr_sum': sampling_band(
            lambda w: acf(np.abs(w), crit.max_lag).sum(), real_windows, rng=rng, **kw),
    }
    return {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in bands.items()}


# ============================================================
# RÉSUMÉ D'UN GÉNÉRATEUR + VERDICT
# ============================================================
def generator_summary(windows: np.ndarray, max_lag: int = 20) -> dict:
    """Toutes les métriques descriptives d'un jeu de fenêtres, JSON-ready."""
    ws = window_stats(windows)
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    return {
        'n_windows'    : int(len(windows)),
        'moments'      : pooled_moments(windows),
        'acf_r'        : acf(windows, max_lag).tolist(),
        'acf_absr'     : acf(np.abs(windows), max_lag).tolist(),
        'acf_absr_sum' : float(acf(np.abs(windows), max_lag).sum()),
        'window_stats' : {
            name: {f'q_{q}': float(np.quantile(arr, q)) for q in qs}
            for name, arr in ws.items()
        },
    }


def _in_band(value: float, lo: float, hi: float) -> bool:
    return bool(lo <= value <= hi)


def judge_criteria(
    summary        : dict,
    bands          : dict,
    crit           : ValidationCriteria,
    disc_acc       : Optional[float] = None,
    garch_disc_acc : Optional[float] = None,
    nn_median      : Optional[float] = None,
    nn_real_ref    : Optional[float] = None,
) -> dict:
    """
    Verdict É2-É6 d'un générateur contre les bandes réelles.
    É5/É6 ne sont évalués que si leurs entrées sont fournies.
    Chaque critère → {pass, value, band/threshold} : lisible dans le JSON.
    """
    out = {}

    kurt = summary['moments']['kurtosis_excess']
    b = bands['kurtosis_excess']
    out['E2_queues'] = {
        'pass'  : kurt >= crit.kurtosis_min and _in_band(kurt, b['lo'][0], b['hi'][0]),
        'value' : kurt,
        'band'  : [b['lo'][0], b['hi'][0]],
        'floor' : crit.kurtosis_min,
    }

    acf_r = summary['acf_r']
    b = bands['acf_r']
    checks = {k: _in_band(acf_r[k - 1], b['lo'][k - 1], b['hi'][k - 1])
              for k in crit.acf_r_lags}
    out['E3_acf_parasite'] = {
        'pass'     : all(checks.values()),
        'value'    : {f'lag_{k}': acf_r[k - 1] for k in crit.acf_r_lags},
        'per_lag'  : {f'lag_{k}': v for k, v in checks.items()},
    }

    # É4 asymétrique sur la somme : le risque couvert est l'ABSENCE de
    # clustering (bootstrap/gaussienne → somme ≈ 0), pas son excès. La borne
    # basse discrimine ; un dépassement haut (sur-persistance, cas GARCH fitté)
    # est signalé en warning sans invalider — l'estimateur fenêtré est de toute
    # façon biaisé aux lags longs.
    absr1 = summary['acf_absr'][0]
    b1 = bands['acf_absr']
    bs = bands['acf_absr_sum']
    s = summary['acf_absr_sum']
    out['E4_clustering'] = {
        'pass' : (absr1 > 0.0
                  and _in_band(absr1, b1['lo'][0], b1['hi'][0])
                  and s >= bs['lo'][0]),
        'warning_surpersistance' : bool(s > bs['hi'][0]),
        'lag1'      : absr1,
        'lag1_band' : [b1['lo'][0], b1['hi'][0]],
        'sum'       : s,
        'sum_band'  : [bs['lo'][0], bs['hi'][0]],
    }

    if disc_acc is not None and garch_disc_acc is not None:
        out['E5_discriminatif'] = {
            'pass'      : disc_acc <= garch_disc_acc + crit.disc_margin,
            'value'     : disc_acc,
            'threshold' : garch_disc_acc + crit.disc_margin,
        }

    if nn_median is not None and nn_real_ref is not None:
        out['E6_memorisation'] = {
            'pass'      : nn_median >= nn_real_ref,
            'value'     : nn_median,
            'threshold' : nn_real_ref,
        }

    out['all_pass'] = all(v['pass'] for k, v in out.items() if isinstance(v, dict))
    return out
