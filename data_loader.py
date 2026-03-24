import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIG CENTRALISÉE
# ============================================================
@dataclass
class DataConfig:
    ticker:        str   = "AAPL"
    start_date:    str   = "2018-01-01"
    end_date:      str   = "2024-01-01"
    train_ratio:   float = 0.7
    val_ratio:     float = 0.15
    # test = 1 - train - val = 0.15
    
    vol_window:    int   = 20    # Fenêtre volatilité rolling
    rsi_window:    int   = 14    # Fenêtre RSI
    sma_window:    int   = 50    # Fenêtre SMA (mean reversion)
    max_regimes:   int   = 5     # Max composantes GMM (sélection par BIC)

FEATURES = [
    'log_returns',
    'volatility',
    'rsi',
    'dist_to_sma',
    'market_regime',
]


# ============================================================
# 1. TÉLÉCHARGEMENT
# ============================================================
def _download(cfg: DataConfig) -> pd.DataFrame:
    """Télécharge et nettoie les données brutes yfinance."""
    
    df = yf.download(
        cfg.ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=True,   # Close = prix ajusté directement
        progress=False
    )
    
    if df.empty:
        raise ValueError(f"Aucune donnée pour {cfg.ticker}. Vérifie le ticker.")
    
    # Gestion MultiIndex yfinance >= 0.2.x
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # On garde uniquement ce dont on a besoin
    df = df[['Close']].copy()
    df.columns = ['price']
    df.dropna(inplace=True)
    
    print(f"✅ {cfg.ticker} | {df.shape[0]} jours | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    
    return df


# ============================================================
# 2. FEATURE ENGINEERING (5 features core)
# ============================================================
def _build_features(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Construit les 5 features core.
    IMPORTANT : le GMM est fit uniquement sur le train set
    pour éviter tout data leakage.
    """
    data = df.copy()
    
    # --------------------------------------------------
    # A. Log Returns
    # Hypothèse GBM : log(S_t / S_{t-1}) ~ N(μ, σ²)
    # --------------------------------------------------
    data['log_returns'] = np.log(data['price'] / data['price'].shift(1))
    
    # --------------------------------------------------
    # B. Volatilité Rolling (proxy GARCH)
    # σ_t = std(log_returns, window=20)
    # Capture les régimes calme/nerveux
    # --------------------------------------------------
    data['volatility'] = (
        data['log_returns']
        .rolling(window=cfg.vol_window)
        .std()
    )
    
    # --------------------------------------------------
    # C. RSI - Momentum
    # RSI > 70 → surachat | RSI < 30 → survente
    # Normalisé en [0, 1] pour stabiliser l'apprentissage
    # --------------------------------------------------
    rsi = RSIIndicator(close=data['price'], window=cfg.rsi_window)
    data['rsi'] = rsi.rsi() / 100.0   # → [0, 1]
    
    # --------------------------------------------------
    # D. Distance à la SMA (Signal Ornstein-Uhlenbeck)
    # x_t = (S_t - SMA_t) / SMA_t
    # Positif → au dessus de la moyenne (vendre ?)
    # Négatif → en dessous (acheter ?)
    # --------------------------------------------------
    sma = SMAIndicator(close=data['price'], window=cfg.sma_window).sma_indicator()
    data['dist_to_sma'] = (data['price'] - sma) / sma
    
    # --------------------------------------------------
    # E. Market Regime (GMM) - Fit sur train seulement
    # 3 états : Bull / Bear / Sideways
    # Nombre optimal choisi par BIC
    # --------------------------------------------------
    data.dropna(inplace=True)   # Drop NaN AVANT le GMM
    
    train_end = int(len(data) * cfg.train_ratio)
    train_data = data[['log_returns', 'volatility']].iloc[:train_end]
    
    n_regimes = _select_gmm_components(train_data, cfg.max_regimes)
    
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=42,
        n_init=5            # Plusieurs initialisations → plus stable
    )
    gmm.fit(train_data)
    data['market_regime'] = gmm.predict(data[['log_returns', 'volatility']])
    
    print(f"🧠 GMM : {n_regimes} régimes détectés")
    print(f"   Distribution : {data['market_regime'].value_counts().to_dict()}")
    
    return data, gmm


def _select_gmm_components(data: pd.DataFrame, max_k: int) -> int:
    """Sélectionne le nombre optimal de composantes GMM via BIC."""
    bic_scores = {}
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(data)
        bic_scores[k] = gmm.bic(data)
    
    optimal_k = min(bic_scores, key=bic_scores.get)
    return optimal_k


# ============================================================
# 3. NORMALISATION (RobustScaler - résistant aux outliers)
# ============================================================
def _scale_features(
    data: pd.DataFrame,
    train_end: int,
    val_end: int
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Normalise les features avec RobustScaler.
    Fit UNIQUEMENT sur le train set → pas de data leakage.
    'market_regime' est catégoriel → pas normalisé.
    """
    features_to_scale = ['log_returns', 'volatility', 'rsi', 'dist_to_sma']
    
    scaler = RobustScaler()  # Médiane + IQR → robuste aux outliers de marché
    
    data = data.copy()
    data.loc[
        data.index[:train_end], features_to_scale
    ] = scaler.fit_transform(data[features_to_scale].iloc[:train_end])
    
    data.loc[
        data.index[train_end:], features_to_scale
    ] = scaler.transform(data[features_to_scale].iloc[train_end:])
    
    return data, scaler


# ============================================================
# 4. SPLIT TRAIN / VAL / TEST
# ============================================================
def _split(
    data: pd.DataFrame,
    cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict (pas de shuffle !)."""
    
    n = len(data)
    train_end = int(n * cfg.train_ratio)
    val_end   = int(n * (cfg.train_ratio + cfg.val_ratio))
    
    train = data.iloc[:train_end].copy()
    val   = data.iloc[train_end:val_end].copy()
    test  = data.iloc[val_end:].copy()
    
    print(f"\n📊 Split :")
    print(f"   Train : {len(train)} jours "
          f"({train.index[0].date()} → {train.index[-1].date()})")
    print(f"   Val   : {len(val)} jours "
          f"({val.index[0].date()} → {val.index[-1].date()})")
    print(f"   Test  : {len(test)} jours "
          f"({test.index[0].date()} → {test.index[-1].date()})")
    
    return train, val, test


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================
def load_data(cfg: DataConfig = DataConfig()):
    """
    Pipeline complet :
    Download → Features → Scale → Split
    
    Returns:
        train, val, test  : DataFrames avec FEATURES + 'price'
        scaler            : pour inverse_transform si besoin
        gmm               : pour réutilisation en live
    """
    print(f"\n{'='*50}")
    print(f"  DRL Portfolio — Data Pipeline")
    print(f"{'='*50}\n")
    
    # 1. Download
    df = _download(cfg)
    
    # 2. Features
    data, gmm = _build_features(df, cfg)
    
    # 3. Calcul des indices de split
    n = len(data)
    train_end = int(n * cfg.train_ratio)
    val_end   = int(n * (cfg.train_ratio + cfg.val_ratio))
    
    # 4. Normalisation (fit sur train uniquement)
    data, scaler = _scale_features(data, train_end, val_end)
    
    # 5. Split
    train, val, test = _split(data, cfg)
    
    print(f"\n✅ Pipeline terminé | Features : {FEATURES}")
    print(f"{'='*50}\n")
    
    return train, val, test, scaler, gmm


# ============================================================
# 6. VISUALISATION
# ============================================================
def plot_data(train, val, test):
    """Visualise prix, features et régimes."""
    
    data = pd.concat([train, val, test])
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('DRL Portfolio — Feature Overview', fontsize=14, y=1.01)
    
    # --- Prix + régimes en couleur ---
    ax = axes[0]
    colors = {0: '#2ecc71', 1: '#e74c3c', 2: '#3498db', 3: '#f39c12'}
    for regime in data['market_regime'].unique():
        mask = data['market_regime'] == regime
        ax.scatter(
            data.index[mask], data['price'][mask],
            s=3, color=colors.get(regime, 'gray'),
            label=f'Régime {regime}'
        )
    # Ligne de séparation train/val/test
    ax.axvline(train.index[-1], color='black', linestyle='--',
               linewidth=1, label='Train|Val')
    ax.axvline(val.index[-1],   color='gray',  linestyle='--',
               linewidth=1, label='Val|Test')
    ax.set_title('Prix & Régimes de Marché (GMM)')
    ax.legend(markerscale=4, fontsize=8)
    ax.set_ylabel('Prix ($)')
    
    # --- Volatilité ---
    ax = axes[1]
    ax.plot(data.index, data['volatility'],
            color='#e67e22', linewidth=1)
    ax.axvline(train.index[-1], color='black', linestyle='--', linewidth=1)
    ax.axvline(val.index[-1],   color='gray',  linestyle='--', linewidth=1)
    ax.set_title('Volatilité (Log Returns Rolling Std)')
    ax.set_ylabel('Volatilité (scaled)')
    
    # --- RSI ---
    ax = axes[2]
    ax.plot(data.index, data['rsi'],
            color='#9b59b6', linewidth=1)
    ax.axhline(0.7, color='red',   linestyle=':', linewidth=1, label='Surachat (0.7)')
    ax.axhline(0.3, color='green', linestyle=':', linewidth=1, label='Survente (0.3)')
    ax.axvline(train.index[-1], color='black', linestyle='--', linewidth=1)
    ax.axvline(val.index[-1],   color='gray',  linestyle='--', linewidth=1)
    ax.set_title('RSI (Momentum)')
    ax.legend(fontsize=8)
    ax.set_ylabel('RSI (scaled)')
    
    # --- Distance SMA ---
    ax = axes[3]
    ax.plot(data.index, data['dist_to_sma'],
            color='#1abc9c', linewidth=1)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(train.index[-1], color='black', linestyle='--', linewidth=1)
    ax.axvline(val.index[-1],   color='gray',  linestyle='--', linewidth=1)
    ax.set_title('Distance à la SMA50 (Mean Reversion)')
    ax.set_ylabel('Dist (scaled)')
    
    plt.tight_layout()
    plt.show()


def plot_correlation(train):
    """Heatmap de corrélation sur le train set uniquement."""
    
    corr = train[FEATURES].corr()
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0,
        vmin=-1, vmax=1,
        square=True, linewidths=0.5
    )
    plt.title('Corrélation entre Features (Train Set)')
    plt.tight_layout()
    plt.show()
    
    # Alerte si corrélation trop forte
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = abs(corr.iloc[i, j])
            if val > 0.8:
                print(f"⚠️  Forte corrélation : "
                      f"{corr.columns[i]} ↔ {corr.columns[j]} = {val:.2f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    cfg = DataConfig(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2024-01-01",
    )
    
    train, val, test, scaler, gmm = load_data(cfg)
    
    print(train[FEATURES].describe().round(3))
    
    plot_data(train, val, test)
    plot_correlation(train)