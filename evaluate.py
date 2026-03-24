# evaluate.py
# Lance ça SANS réentraîner

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_loader import load_data, DataConfig
from environment import TradingEnv, EnvConfig
import numpy as np
import pandas as pd

def evaluate(
    model_path      : str = "models/ppo_aapl.zip",
    vec_norm_path   : str = "models/vec_normalize.pkl",
    ticker          : str = "AAPL",
    start_date      : str = "2019-01-01",
    end_date        : str = "2023-01-01",
):
    # 1. Charge les données
    cfg_data = DataConfig(
        ticker     = ticker,
        start_date = start_date,
        end_date   = end_date,
    )
    _, _, test_data, _, _ = load_data(cfg_data)

    # 2. Charge le modèle sauvegardé
    model = PPO.load(model_path)
    print(f"✅ Modèle chargé : {model_path}")

    # 3. Évalue sur N seeds
    seeds   = [42, 123, 456, 789, 1337]
    results = []

    for seed in seeds:
        cfg_env = EnvConfig()
        env     = TradingEnv(
            data        = test_data,
            cfg         = cfg_env,
            render_mode = "human"
        )
        obs, _ = env.reset(seed=seed)
        done   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        portfolio = np.array(env.history["portfolio_values"])
        prices    = np.array(env.history["prices"])
        actions   = np.array(env.history["actions"])
        returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-8)

        total_ret = (portfolio[-1] - portfolio[0]) / portfolio[0]
        bh_ret    = (prices[-1] - prices[0]) / prices[0]
        sharpe    = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        max_dd    = np.max(
            (np.maximum.accumulate(portfolio) - portfolio)
            / (np.maximum.accumulate(portfolio) + 1e-8)
        )

        results.append({
            "seed"      : seed,
            "return"    : total_ret,
            "bh_return" : bh_ret,
            "sharpe"    : sharpe,
            "max_dd"    : max_dd,
            "n_trades"  : env.history["n_trades"],
            "hold_pct"  : (actions == 0).mean(),
            "buy_pct"   : (actions == 1).mean(),
            "sell_pct"  : (actions == 2).mean(),
        })

    df = pd.DataFrame(results)

    print("\n" + "="*50)
    print("  RÉSULTATS MULTI-SEEDS")
    print("="*50)
    print(f"  Ticker  : {ticker}")
    print(f"  Période : {start_date} → {end_date}")
    print(f"  Return  : {df['return'].mean():+.2%} ± {df['return'].std():.2%}")
    print(f"  B&H     : {df['bh_return'].mean():+.2%}")
    print(f"  Alpha   : {(df['return'] - df['bh_return']).mean():+.2%}")
    print(f"  Sharpe  : {df['sharpe'].mean():.3f} ± {df['sharpe'].std():.3f}")
    print(f"  Max DD  : {df['max_dd'].mean():.2%}")
    print(f"  Trades  : {df['n_trades'].mean():.0f}")
    print("="*50)
    print(f"\n  Actions (moyenne) :")
    print(f"  Hold : {df['hold_pct'].mean():.1%}")
    print(f"  Buy  : {df['buy_pct'].mean():.1%}")
    print(f"  Sell : {df['sell_pct'].mean():.1%}")

    # Render dernier épisode
    env.render()

    return df


if __name__ == "__main__":

    # ✅ Change juste les dates ici SANS réentraîner
    evaluate(
        model_path  = "models/ppo_aapl.zip",
        ticker      = "AAPL",
        start_date  = "2019-01-01",
        end_date    = "2023-01-01",
    )