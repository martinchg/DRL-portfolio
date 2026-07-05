# diffusion/baselines.py
"""
Générateurs de référence pour calibrer le protocole de validation.

Chaque baseline a un rôle précis :
- B0 gaussienne i.i.d.  : contrôle négatif — doit ÉCHOUER queues (É2) et clustering (É4)
- B1 bootstrap i.i.d.   : marginales parfaites par construction — doit ÉCHOUER É4
                          (le rééchantillonnage détruit le volatility clustering)
- B2 GARCH(1,1)-t       : la barre classique à battre — 4 paramètres, passe É2-É4 ;
                          si le DDPM ne fait pas mieux au score discriminatif (É5),
                          il ne justifie pas sa complexité.

Si les baselines n'échouent pas là où elles doivent, c'est le PROTOCOLE qui est
faux (É1), pas les modèles.

`arch` n'est utilisé que pour le FIT (MLE, la partie délicate) ; la simulation
est faite ici en numpy vectorisé — déterminisme contrôlé par notre Generator,
là où arch.simulate dépend d'un état aléatoire interne.
"""
from typing import Dict, Optional

import numpy as np


# ============================================================
# B0 / B1 — i.i.d.
# ============================================================
def sample_gaussian_iid(
    n_windows : int,
    window    : int,
    mu        : float,
    sigma     : float,
    rng       : Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.normal(mu, sigma, size=(n_windows, window)).astype(np.float32)


def sample_bootstrap_iid(
    n_windows      : int,
    window         : int,
    pooled_returns : np.ndarray,
    rng            : Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Tirage i.i.d. avec remise dans les rendements réels poolés."""
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(pooled_returns), size=(n_windows, window))
    return np.asarray(pooled_returns, dtype=np.float32)[idx]


# ============================================================
# Simulateur GARCH(1,1) numpy (innovations normales ou Student-t)
# ============================================================
def simulate_garch(
    n_windows : int,
    window    : int,
    omega     : float,
    alpha     : float,
    beta      : float,
    mu        : float = 0.0,
    nu        : Optional[float] = None,
    rng       : Optional[np.random.Generator] = None,
    burn      : int   = 500,
) -> np.ndarray:
    """
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} ; r_t = μ + σ_t z_t.
    z ~ N(0,1) si nu=None, sinon Student-t(ν) standardisée (variance 1).
    α + β < 1 requis (stationnarité) ; départ à la variance inconditionnelle
    + burn-in pour oublier l'initialisation.
    """
    if alpha + beta >= 1.0:
        raise ValueError("GARCH non stationnaire : alpha + beta >= 1")
    rng = rng or np.random.default_rng(0)
    total = window + burn

    if nu is None:
        z = rng.standard_normal((n_windows, total))
    else:
        if nu <= 2.0:
            raise ValueError("Student-t : nu > 2 requis (variance finie)")
        z = rng.standard_t(nu, size=(n_windows, total)) * np.sqrt((nu - 2.0) / nu)

    r = np.empty((n_windows, total))
    var = np.full(n_windows, omega / (1.0 - alpha - beta))
    for t in range(total):
        eps = np.sqrt(var) * z[:, t]
        r[:, t] = mu + eps
        var = omega + alpha * eps ** 2 + beta * var
    return r[:, burn:].astype(np.float32)


# ============================================================
# B2 — GARCH(1,1)-t fitté par ticker (fit via arch, simulation numpy)
# ============================================================
# Fit PAR SEGMENT (= par ticker) : fitter sur les rendements concaténés
# mélangerait les échelles de vol (TSLA ~3× SPY) et créerait de faux clusters
# aux jonctions — même famille de piège que segment_id côté RL.
_ARCH_SCALE = 100.0   # arch préfère des rendements en % (stabilité numérique)


def fit_garch_per_segment(
    seg_returns: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    """
    GARCH(1,1) à innovations Student-t, moyenne constante, par segment.
    Paramètres reconvertis dans l'échelle des rendements bruts :
    mu et √ω sont ×1/100, alpha/beta/nu sont sans échelle.
    """
    from arch import arch_model
    params = {}
    for sid, rets in seg_returns.items():
        am = arch_model(
            np.asarray(rets, dtype=np.float64) * _ARCH_SCALE,
            mean='Constant', vol='GARCH', p=1, q=1, dist='t',
        )
        res = am.fit(disp='off', show_warning=False)
        p = res.params
        params[sid] = {
            'mu'    : float(p['mu']) / _ARCH_SCALE,
            'omega' : float(p['omega']) / _ARCH_SCALE ** 2,
            'alpha' : float(p['alpha[1]']),
            'beta'  : float(p['beta[1]']),
            'nu'    : float(p['nu']),
        }
    return params


def sample_garch_fitted(
    n_windows   : int,
    window      : int,
    fitted      : Dict[int, Dict[str, float]],
    seg_weights : Optional[Dict[int, float]] = None,
    rng         : Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Fenêtres simulées : chaque fenêtre tire d'abord son ticker (proportionnel
    au poids du segment, ex : nombre de fenêtres réelles), puis un chemin
    GARCH indépendant du modèle fitté de ce ticker.
    """
    rng = rng or np.random.default_rng(0)
    sids = sorted(fitted.keys())

    if seg_weights is None:
        probs = np.full(len(sids), 1.0 / len(sids))
    else:
        w = np.array([seg_weights[s] for s in sids], dtype=np.float64)
        probs = w / w.sum()

    counts = rng.multinomial(n_windows, probs)
    parts = []
    for sid, count in zip(sids, counts):
        if count == 0:
            continue
        p = fitted[sid]
        parts.append(simulate_garch(
            count, window,
            omega=p['omega'], alpha=p['alpha'], beta=p['beta'],
            mu=p['mu'], nu=p['nu'], rng=rng,
        ))
    windows = np.concatenate(parts, axis=0)
    return windows[rng.permutation(len(windows))]
