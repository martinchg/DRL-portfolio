# conftest.py — fixtures partagées
# Données synthétiques uniquement : aucun test unitaire ne dépend du réseau.
import os
import sys

# Backend headless AVANT tout import de matplotlib (via data_loader/environment)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

# Rend les modules du projet importables depuis tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataConfig, _build_features  # noqa: E402


def make_price_df(n=600, seed=0, mu=0.0004, sigma=0.02, start=100.0):
    """Série de prix GBM avec index de jours ouvrés."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(mu, sigma, n)
    prices = start * np.exp(np.cumsum(log_ret))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"price": prices}, index=idx)


def make_manual_features(prices) -> pd.DataFrame:
    """
    Construit un DataFrame avec les colonnes attendues par TradingEnv
    à partir de prix contrôlés exactement (pas de dropna, pas de NaN).
    Permet des assertions comptables exactes sur les trades.
    """
    prices = np.asarray(prices, dtype=float)
    log_ret = np.zeros_like(prices)
    log_ret[1:] = np.log(prices[1:] / prices[:-1])
    vol = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.0).values
    return pd.DataFrame({
        "price": prices,
        "log_returns": log_ret,
        "volatility": vol,
        "rsi": np.full(len(prices), 0.5),
        "macd_norm": np.zeros(len(prices)),
        "momentum_5": np.zeros(len(prices)),
    })


@pytest.fixture(scope="session")
def price_df():
    """Prix bruts GBM (colonne 'price' uniquement)."""
    return make_price_df(n=600, seed=0)


@pytest.fixture(scope="session")
def gbm_features(price_df):
    """Prix GBM passés dans le vrai feature engineering du projet."""
    return _build_features(price_df, DataConfig())


@pytest.fixture
def flat_data():
    """Prix constant à 100 : comptabilité exacte quel que soit le départ aléatoire."""
    return make_manual_features(np.full(260, 100.0))


@pytest.fixture
def declining_data():
    """Prix qui perd exactement 3 %/jour : comportement identique quel que soit le départ."""
    t = np.arange(260)
    return make_manual_features(100.0 * 0.97 ** t)


@pytest.fixture
def rising_data():
    """Prix qui gagne exactement 3 %/jour."""
    t = np.arange(260)
    return make_manual_features(100.0 * 1.03 ** t)


@pytest.fixture
def segment_data():
    """Données type multi-ticker : 2 segments de 200 jours avec segment_id."""
    a = make_manual_features(np.full(200, 100.0))
    a["segment_id"] = 0
    a["ticker"] = "AAA"
    b = make_manual_features(np.linspace(50.0, 80.0, 200))
    b["segment_id"] = 1
    b["ticker"] = "BBB"
    return pd.concat([a, b], ignore_index=True)
