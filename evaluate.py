# evaluate.py

from stable_baselines3 import PPO
from data_loader import load_data, load_multi_ticker_data, DataConfig
from environment import TradingEnv, EnvConfig
import numpy as np
import pandas as pd


def evaluate_one(
    model_path : str,
    test_data  : pd.DataFrame,
    cfg_env    : EnvConfig,
    label      : str,
    seeds      : list = [42, 123, 456, 789, 1337]
) -> dict:
    """Évalue un modèle sur un test set."""

    model   = PPO.load(model_path)
    results = []

    for seed in seeds:
        env    = TradingEnv(data=test_data, cfg=cfg_env)
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

        results.append({
            "seed"     : seed,
            "return"   : (portfolio[-1] - portfolio[0]) / portfolio[0],
            "bh"       : (prices[-1] - prices[0]) / prices[0],
            "sharpe"   : (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252),
            "max_dd"   : np.max(
                (np.maximum.accumulate(portfolio) - portfolio)
                / (np.maximum.accumulate(portfolio) + 1e-8)
            ),
            "n_trades" : env.history["n_trades"],
            "hold_pct" : (actions == 0).mean(),
            "buy_pct"  : (actions == 1).mean(),
            "sell_pct" : (actions == 2).mean(),
        })

    df = pd.DataFrame(results)

    return {
        "label"    : label,
        "return"   : df["return"].mean(),
        "return_std": df["return"].std(),
        "bh"       : df["bh"].mean(),
        "alpha"    : (df["return"] - df["bh"]).mean(),
        "sharpe"   : df["sharpe"].mean(),
        "sharpe_std": df["sharpe"].std(),
        "max_dd"   : df["max_dd"].mean(),
        "n_trades" : df["n_trades"].mean(),
        "hold_pct" : df["hold_pct"].mean(),
        "buy_pct"  : df["buy_pct"].mean(),
        "sell_pct" : df["sell_pct"].mean(),
    }


def compare_models(ticker="AAPL"):
    """
    Compare le modèle single vs multi ticker
    sur le MÊME test set (AAPL)
    pour une comparaison équitable.
    """

    cfg_env = EnvConfig()

    # ── Test set commun : AAPL ──────────────────────────
    cfg_data = DataConfig(
        ticker     = ticker,
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    _, _, test_aapl, _ = load_data(cfg_data)

    print(f"\n{'='*60}")
    print(f"  COMPARAISON Single vs Multi — Test set {ticker}")
    print(f"{'='*60}\n")

    # ── Évalue les 2 modèles ───────────────────────────
    results = []

    # Single ticker
    r_single = evaluate_one(
        model_path = "models/ppo_single/best_model.zip",
        test_data  = test_aapl,
        cfg_env    = cfg_env,
        label      = "Single (AAPL)"
    )
    results.append(r_single)

    # Multi ticker (testé sur AAPL quand même)
    r_multi = evaluate_one(
        model_path = "models/ppo_multi/best_model.zip",
        test_data  = test_aapl,
        cfg_env    = cfg_env,
        label      = "Multi (5 tickers)"
    )
    results.append(r_multi)

    # ── Tableau comparatif ─────────────────────────────
    print(f"{'Modèle':<20} {'Return':>10} {'B&H':>10} "
          f"{'Alpha':>10} {'Sharpe':>10} {'MaxDD':>10} "
          f"{'Trades':>8}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['label']:<20} "
            f"{r['return']:>+10.2%} "
            f"{r['bh']:>+10.2%} "
            f"{r['alpha']:>+10.2%} "
            f"{r['sharpe']:>10.3f} "
            f"{r['max_dd']:>10.2%} "
            f"{r['n_trades']:>8.0f}"
        )

    print("-" * 80)

    # ── Actions ───────────────────────────────────────
    print(f"\n{'Modèle':<20} {'Hold':>10} {'Buy':>10} {'Sell':>10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['label']:<20} "
            f"{r['hold_pct']:>10.1%} "
            f"{r['buy_pct']:>10.1%} "
            f"{r['sell_pct']:>10.1%}"
        )

    # ── Verdict ───────────────────────────────────────
    print(f"\n📊 Verdict :")
    if r_multi['sharpe'] > r_single['sharpe']:
        print(f"  ✅ Multi-ticker GAGNE en Sharpe "
              f"({r_multi['sharpe']:.3f} vs {r_single['sharpe']:.3f})")
    else:
        print(f"  ✅ Single-ticker GAGNE en Sharpe "
              f"({r_single['sharpe']:.3f} vs {r_multi['sharpe']:.3f})")

    if r_multi['alpha'] > r_single['alpha']:
        print(f"  ✅ Multi-ticker GAGNE en Alpha "
              f"({r_multi['alpha']:+.2%} vs {r_single['alpha']:+.2%})")
    else:
        print(f"  ✅ Single-ticker GAGNE en Alpha "
              f"({r_single['alpha']:+.2%} vs {r_multi['alpha']:+.2%})")

    return results


def check_overfitting_both():
    """Vérifie l'overfitting des 2 modèles."""

    cfg_env  = EnvConfig()
    cfg_data = DataConfig(
        ticker     = "AAPL",
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    train, val, test, _ = load_data(cfg_data)

    print(f"\n{'='*65}")
    print(f"  DÉTECTION OVERFITTING — Single vs Multi")
    print(f"{'='*65}")

    for model_name, model_path in [
        ("Single", "models/ppo_single/best_model.zip"),
        ("Multi",  "models/ppo_multi/best_model.zip"),
    ]:
        print(f"\n── {model_name} ──────────────────────────────────")
        print(f"{'Split':<10} {'Return':>10} {'B&H':>10} "
              f"{'Alpha':>10} {'Sharpe':>10}")
        print("-" * 50)

        for split_name, split_data in [
            ("Train", train),
            ("Val",   val),
            ("Test",  test),
        ]:
            r = evaluate_one(
                model_path = model_path,
                test_data  = split_data,
                cfg_env    = cfg_env,
                label      = split_name,
                seeds      = [42]   # 1 seed suffit pour overfit check
            )
            print(
                f"{split_name:<10} "
                f"{r['return']:>+10.2%} "
                f"{r['bh']:>+10.2%} "
                f"{r['alpha']:>+10.2%} "
                f"{r['sharpe']:>10.3f}"
            )


if __name__ == "__main__":

    # 1. Compare les 2 modèles sur AAPL
    compare_models(ticker="AAPL")

    # 2. Vérifie l'overfitting des 2
    check_overfitting_both()