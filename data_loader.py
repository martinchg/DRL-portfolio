# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass, field
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
@dataclass
class DataConfig:
    # ── Données ──────────────────────────────────────────────────────
    # [GESTIONNAIRE] Ticker pour le mode single-asset.
    # Exemples : "AAPL", "MSFT", "SPY", "BTC-USD", "GLD"
    ticker      : str   = "AAPL"

    # [GESTIONNAIRE] Période d'historique utilisée pour entraîner ET tester.
    # Plus la période est longue → plus de données → meilleure généralisation.
    # La période doit contenir plusieurs cycles de marché (bull + bear).
    #
    # Recommandations :
    # "2010-01-01" → "2023-01-01" = 13 ans ✅ (défaut)
    #   Train 2010-2019 : correction 2011, bear 2015, bear 2018
    #   Val   2019-2021 : COVID crash + recovery (agent apprend à shorter)
    #   Test  2021-2023 : bull 2021 + bear 2022
    #
    # "2015-01-01" → "2023-01-01" = 8 ans
    #   Moins de data mais inclut quand même COVID
    #
    # "2018-01-01" → "2023-01-01" = 5 ans (ancien défaut, trop court)
    #   Test = uniquement 2022 bear → très difficile à battre sans short
    start_date  : str   = "2010-01-01"
    end_date    : str   = "2023-01-01"

    # [GESTIONNAIRE] Liste de tickers pour l'entraînement multi-asset.
    # Diversifier les actifs force l'agent à apprendre des patterns généraux
    # plutôt que de mémoriser la trajectoire d'un seul actif.
    # ⚠️  Limitation connue : les épisodes peuvent croiser les frontières entre actifs
    #    dans les données concaténées → entraînement imparfait mais acceptable.
    tickers     : List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "SPY", "TSLA"
    ])
    multi_ticker : bool = False

    # ── Découpage train/val/test ──────────────────────────────────────
    # [GESTIONNAIRE] Proportions du split temporel (chronologique strict).
    # train  = premières données (l'agent apprend dessus)
    # val    = données intermédiaires (sélection du meilleur modèle)
    # test   = dernières données (évaluation finale, jamais vu pendant l'entraînement)
    # Règle : test doit couvrir au moins 6-12 mois pour être significatif.
    train_ratio : float = 0.7
    val_ratio   : float = 0.15
    # test_ratio  = 0.15 (automatique : 1 - train - val)

    # ── Features techniques ───────────────────────────────────────────
    # [GESTIONNAIRE] Fenêtre pour la volatilité rolling (proxy du risque court-terme).
    # 10 = volatilité très réactive  |  20 = standard  |  60 = volatilité lente
    vol_window  : int   = 20

    # [GESTIONNAIRE] Fenêtre RSI (Relative Strength Index).
    # 9  = RSI rapide (day trading)  |  14 = standard  |  21 = RSI lent
    rsi_window  : int   = 14

    sma_window  : int   = 50
    max_regimes : int   = 5

    # ── Features de régime (Acte 3) ──────────────────────────────────
    # [GESTIONNAIRE] Ajoute dist_high_252 et trend_200 aux features.
    # Motivation : le walk-forward a montré que l'agent ne distingue pas
    # un krach durable d'un rebond en V — ses features court-terme (10 j)
    # ne contiennent pas l'information de régime.
    # ⚠️  Change la taille d'observation (52 → 72) : modèles incompatibles
    #    avec les anciens ; coûte ~252 jours de warm-up en début de données.
    regime_features : bool = False


FEATURES = [
    'log_returns',
    'volatility',
    'rsi',
    'macd_norm',
    'momentum_5',
]

# Features de régime (optionnelles, cf. DataConfig.regime_features) :
# l'état LONG-terme du marché, invisible dans une fenêtre de 10 jours.
REGIME_FEATURES = [
    'dist_high_252',   # position vs plus-haut 1 an : -0.3 = 30 % sous le pic
    'trend_200',       # position vs moyenne mobile 200 j : signe = régime praticien
]


def active_features(cfg) -> list:
    """Liste des features actives selon la config données."""
    return FEATURES + (REGIME_FEATURES if cfg.regime_features else [])


# ============================================================
# 1. TÉLÉCHARGEMENT
# ============================================================
def _download(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Télécharge et nettoie les données brutes yfinance."""

    df = yf.download(
        ticker,
        start      = start_date,
        end        = end_date,
        auto_adjust = True,
        progress   = False
    )

    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}.")

    # Gestion MultiIndex yfinance >= 0.2.x
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()
    df.columns = ['price']
    df.dropna(inplace=True)

    print(f"✅ {ticker} | {df.shape[0]} jours | "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def _build_features(
    df  : pd.DataFrame,
    cfg : DataConfig
) -> pd.DataFrame:
    """
    Construit les 3 features core :
    log_returns, volatility, rsi
    """
    data = df.copy()

    # A. Log Returns
    data['log_returns'] = np.log(
        data['price'] / data['price'].shift(1)
    )

    # B. Volatilité Rolling (proxy GARCH)
    data['volatility'] = (
        data['log_returns']
        .rolling(window=cfg.vol_window)
        .std()
    )

    # C. RSI normalisé [0, 1]
    rsi = RSIIndicator(close=data['price'], window=cfg.rsi_window)
    data['rsi'] = rsi.rsi() / 100.0

    # D. MACD normalisé (EMA12 - EMA26) / EMA26
    # Mesure le momentum directionnel : positif = tendance haussière
    ema12 = data['price'].ewm(span=12, adjust=False).mean()
    ema26 = data['price'].ewm(span=26, adjust=False).mean()
    data['macd_norm'] = (ema12 - ema26) / (ema26 + 1e-8)

    # E. Momentum 5 jours (return glissant)
    # Signal court-terme : combien le prix a bougé sur 5 jours
    data['momentum_5'] = data['price'].pct_change(5)

    # F/G. Features de régime (optionnelles) — l'état LONG-terme du marché
    if cfg.regime_features:
        # F. Distance au plus-haut 1 an : drawdown de l'actif lui-même.
        #    ≈ 0 → au plus-haut (bull) | -0.3 → 30 % sous le pic (krach/bear)
        roll_max = data['price'].rolling(252, min_periods=252).max()
        data['dist_high_252'] = data['price'] / roll_max - 1.0

        # G. Tendance longue : au-dessus/en-dessous de la SMA 200 jours,
        #    LE proxy de régime des praticiens (golden/death cross).
        sma200 = data['price'].rolling(200, min_periods=200).mean()
        data['trend_200'] = data['price'] / sma200 - 1.0

    # Drop NaN
    data.dropna(inplace=True)

    return data


# ============================================================
# 3. NORMALISATION
# ============================================================
def _scale_features(
    data         : pd.DataFrame,
    train_end    : int,
    feature_cols : list = None,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Normalise avec RobustScaler.
    Fit UNIQUEMENT sur le train set.
    feature_cols : colonnes à scaler (défaut : FEATURES de base).
    """
    cols   = feature_cols if feature_cols is not None else FEATURES
    data   = data.copy()
    scaler = RobustScaler()

    data.loc[
        data.index[:train_end], cols
    ] = scaler.fit_transform(
        data[cols].iloc[:train_end]
    )

    data.loc[
        data.index[train_end:], cols
    ] = scaler.transform(
        data[cols].iloc[train_end:]
    )

    return data, scaler


# ============================================================
# 4. SPLIT
# ============================================================
def _split(
    data : pd.DataFrame,
    cfg  : DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict — pas de shuffle."""

    n         = len(data)
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
# 5. PIPELINE SINGLE TICKER
# ============================================================
def load_data(
    cfg : DataConfig = DataConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Pipeline complet single ticker :
    Download → Features → Scale → Split
    """
    print(f"\n{'='*50}")
    print(f"  DRL Portfolio — Data Pipeline")
    print(f"{'='*50}\n")

    # 1. Download
    df = _download(cfg.ticker, cfg.start_date, cfg.end_date)

    # 2. Features
    data = _build_features(df, cfg)

    # 3. Indices de split
    n         = len(data)
    train_end = int(n * cfg.train_ratio)
    val_end   = int(n * (cfg.train_ratio + cfg.val_ratio))

    # 4. Normalisation (sur toutes les features actives)
    data, scaler = _scale_features(data, train_end, active_features(cfg))

    # 5. Split
    train, val, test = _split(data, cfg)

    print(f"\n✅ Pipeline terminé | Features : {FEATURES}")
    print(f"{'='*50}\n")

    return train, val, test, scaler


# ============================================================
# 6. PIPELINE MULTI-TICKER (anti-overfitting)
# ============================================================
def load_multi_ticker_data(
    cfg : DataConfig = DataConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline multi-ticker :
    Force l'agent à apprendre des patterns GÉNÉRAUX
    et non spécifiques à un seul actif.

    - Train set  : tous les tickers mélangés (shuffled)
    - Val set    : tous les tickers dans l'ordre
    - Test set   : tous les tickers dans l'ordre
    """
    print(f"\n{'='*50}")
    print(f"  DRL Portfolio — Multi-Ticker Pipeline")
    print(f"  Tickers : {cfg.tickers}")
    print(f"{'='*50}\n")

    trains, vals, tests = [], [], []

    for ticker in cfg.tickers:
        try:
            # Download
            df = _download(ticker, cfg.start_date, cfg.end_date)

            # Features
            data = _build_features(df, cfg)

            # Split AVANT normalisation
            n         = len(data)
            train_end = int(n * cfg.train_ratio)
            val_end   = int(n * (cfg.train_ratio + cfg.val_ratio))

            # Normalisation fit sur train uniquement
            data, _ = _scale_features(data, train_end, active_features(cfg))

            # Split
            train = data.iloc[:train_end].copy()
            val   = data.iloc[train_end:val_end].copy()
            test  = data.iloc[val_end:].copy()

            # Ajoute le ticker et un id de segment (pour les épisodes cohérents).
            # segment_id sur TOUS les splits : sans lui, les épisodes de val/test
            # traversent les frontières entre tickers → sauts de prix fictifs
            # (ex : GOOGL ~1000$ → SPY ~300$ = faux crash de -70%) qui polluent
            # la sélection du best_model par l'EvalCallback.
            seg_id           = len(trains)
            train['ticker']  = ticker
            val['ticker']    = ticker
            test['ticker']   = ticker
            train['segment_id'] = seg_id
            val['segment_id']   = seg_id
            test['segment_id']  = seg_id

            trains.append(train)
            vals.append(val)
            tests.append(test)

            print(f"   ✅ {ticker} | "
                  f"Train={len(train)} | "
                  f"Val={len(val)} | "
                  f"Test={len(test)}")

        except Exception as e:
            print(f"   ❌ {ticker} ignoré : {e}")
            continue

    # Combine — ordre temporel conservé par ticker (pas de shuffle)
    # L'environnement utilise segment_id pour confiner les épisodes
    # à un seul ticker et éviter de croiser les frontières.
    train_combined = pd.concat(trains, ignore_index=True)
    val_combined   = pd.concat(vals,   ignore_index=False)
    test_combined  = pd.concat(tests,  ignore_index=False)

    print(f"\n📊 Dataset combiné :")
    print(f"   Train : {len(train_combined)} jours "
          f"({len(trains)} tickers × ~{len(trains[0])} jours)")
    print(f"   Val   : {len(val_combined)} jours")
    print(f"   Test  : {len(test_combined)} jours")
    print(f"\n✅ Multi-ticker pipeline terminé | Features : {FEATURES}")
    print(f"{'='*50}\n")

    return train_combined, val_combined, test_combined


# ============================================================
# 7. VISUALISATION
# ============================================================
def plot_data(train, val, test, ticker=""):
    """Visualise prix, features et régimes."""

    data = pd.concat([train, val, test])

    # Exclut la colonne ticker si présente
    plot_features = [f for f in FEATURES if f in data.columns]

    fig, axes = plt.subplots(
        len(plot_features) + 1, 1,
        figsize=(14, 4 * (len(plot_features) + 1)),
        sharex=True
    )
    fig.suptitle(
        f'DRL Portfolio — Feature Overview {ticker}',
        fontsize=14
    )

    # Prix
    ax = axes[0]
    if 'price' in data.columns:
        ax.plot(data.index, data['price'],
                color='#64ffda', linewidth=1.5)
        ax.axvline(train.index[-1], color='white',
                   linestyle='--', linewidth=1, label='Train|Val')
        ax.axvline(val.index[-1], color='orange',
                   linestyle='--', linewidth=1, label='Val|Test')
        ax.set_title('Prix')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    feature_colors = {
        'log_returns' : '#2ecc71',
        'volatility'  : '#e67e22',
        'rsi'         : '#9b59b6',
    }

    for i, feat in enumerate(plot_features, 1):
        ax = axes[i]
        ax.plot(
            data.index, data[feat],
            color=feature_colors.get(feat, 'white'),
            linewidth=1.2
        )

        # Lignes de référence
        if feat == 'rsi':
            ax.axhline(0.7, color='red',
                       linestyle=':', linewidth=1, label='Surachat')
            ax.axhline(0.3, color='green',
                       linestyle=':', linewidth=1, label='Survente')
            ax.legend(fontsize=8)
        if feat in ['log_returns', 'dist_to_sma']:
            ax.axhline(0, color='white',
                       linewidth=0.8, alpha=0.5)

        ax.axvline(train.index[-1], color='white',
                   linestyle='--', linewidth=1)
        ax.axvline(val.index[-1], color='orange',
                   linestyle='--', linewidth=1)
        ax.set_title(feat)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_correlation(train):
    """Heatmap de corrélation sur le train set."""

    features_to_plot = [f for f in FEATURES if f in train.columns]
    corr = train[features_to_plot].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0,
        vmin=-1, vmax=1,
        square=True, linewidths=0.5
    )
    plt.title('Corrélation entre Features (Train Set)')
    plt.tight_layout()
    plt.show()

    # Alertes
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = abs(corr.iloc[i, j])
            if val > 0.8:
                print(f"⚠️  Forte corrélation : "
                      f"{corr.columns[i]} ↔ "
                      f"{corr.columns[j]} = {val:.2f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ── Single ticker ──────────────────────────────────────
    cfg = DataConfig(
        ticker     = "AAPL",
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    train, val, test, scaler = load_data(cfg)
    print(train[FEATURES].describe().round(3))
    plot_data(train, val, test, ticker="AAPL")
    plot_correlation(train)

    # ── Multi-ticker (anti-overfitting) ───────────────────
    cfg_multi = DataConfig(
        tickers    = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"],
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    train_m, val_m, test_m = load_multi_ticker_data(cfg_multi)
    print(f"\nTrain multi : {train_m.shape}")
    print(f"Val   multi : {val_m.shape}")
    print(f"Test  multi : {test_m.shape}")