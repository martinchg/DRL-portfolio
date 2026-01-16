import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def fetch_and_process_data(ticker, start_date="2018-01-01", end_date="2024-01-01"):
    """
    Récupère les données et crée les 'Features' pour l'IA.
    C'est ici qu'on intègre la logique financière/stochastique.
    """
    print(f"--- Chargement des données pour {ticker} ---")
    
    # 1. Téléchargement des données brutes
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Si le téléchargement échoue ou structure vide
    if df.empty:
        raise ValueError("Aucune donnée téléchargée. Vérifie le ticker ou ta connexion.")

    # On garde une copie propre
    data = df[['Adj Close', 'Volume']].copy()
    data.columns = ['price', 'volume']

    # --- FEATURE ENGINEERING (Le cerveau mathématique) ---
    
    # A. Log Returns (Rendements Logarithmiques)
    # Plus stable pour les modèles stochastiques que les % simples.
    # Hypothèse souvent utilisée : les log-returns suivent une loi Normale (dans Black-Scholes).
    data['log_returns'] = np.log(data['price'] / data['price'].shift(1))

    # B. Volatilité Roulante (Rolling Volatility)
    # L'IA doit savoir si le marché est calme ou nerveux (Régime switch).
    # Calculée sur 20 jours (approx 1 mois de trading).
    data['volatility'] = data['log_returns'].rolling(window=20).std()

    # C. RSI (Relative Strength Index)
    # Indicateur de Momentum : Est-ce que le marché est "surchauffé" ?
    rsi = RSIIndicator(close=data['price'], window=14)
    data['rsi'] = rsi.rsi()

    # D. Distance à la Moyenne (Mean Reversion Signal)
    # Processus type Ornstein-Uhlenbeck : le prix tend à revenir vers sa moyenne.
    sma_50 = SMAIndicator(close=data['price'], window=50).sma_indicator()
    data['dist_to_sma'] = (data['price'] - sma_50) / sma_50

    # Nettoyage des NaN (les 50 premiers jours où on ne peut pas calculer la moyenne)
    data.dropna(inplace=True)

    print(f"Données prêtes : {data.shape[0]} lignes.")
    return data

# --- TEST DIRECT ---
if __name__ == "__main__":
    # On teste avec Bitcoin (BTC-USD) ou Apple (AAPL)
    try:
        df_test = fetch_and_process_data("BTC-USD")
        
        # Aperçu des données que l'IA va "voir"
        print(df_test.head())
        
        # Visualisation pour vérifier
        plt.figure(figsize=(12, 6))
        
        # Prix
        plt.subplot(2, 1, 1)
        plt.plot(df_test['price'], label='Prix BTC')
        plt.title('Prix Bitcoin')
        plt.legend()
        
        # Volatilité (Ce que l'IA utilise pour évaluer le risque)
        plt.subplot(2, 1, 2)
        plt.plot(df_test['volatility'], color='orange', label='Volatilité (Risk)')
        plt.title('Volatilité (Log Returns Rolling Std)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur : {e}")