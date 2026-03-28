# evaluate.py
from data_loader import DataConfig, load_data, FEATURES
from environment import TradingEnv, EnvConfig
from stable_baselines3 import PPO
import numpy as np
import pandas as pd


# ============================================================
# ÉVALUATION D'UN MODÈLE SUR UN SPLIT
# ============================================================
def evaluate_one(
    model_path : str,
    test_data  : pd.DataFrame,
    cfg_env    : EnvConfig,
    label      : str  = "",
    seeds      : list = [42]
) -> dict:
    """Évalue un modèle sur un dataset donné."""

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
        "label"      : label,
        "return"     : df["return"].mean(),
        "return_std" : df["return"].std(),
        "bh"         : df["bh"].mean(),
        "alpha"      : (df["return"] - df["bh"]).mean(),
        "sharpe"     : df["sharpe"].mean(),
        "sharpe_std" : df["sharpe"].std(),
        "max_dd"     : df["max_dd"].mean(),
        "n_trades"   : df["n_trades"].mean(),
        "hold_pct"   : df["hold_pct"].mean(),
        "buy_pct"    : df["buy_pct"].mean(),
        "sell_pct"   : df["sell_pct"].mean(),
    }


# ============================================================
# DÉTECTION OVERFITTING
# ============================================================
def check_overfitting(
    model_path : str,
    label      : str  = "",
    ticker     : str  = "AAPL",
    start_date : str  = "2018-01-01",
    end_date   : str  = "2023-01-01",
) -> dict:
    """
    Évalue le modèle sur Train / Val / Test
    pour détecter l'overfitting.
    """

    # ✅ load_data retourne 4 valeurs (sans gmm)
    cfg_data = DataConfig(
        ticker     = ticker,
        start_date = start_date,
        end_date   = end_date,
    )
    train, val, test, scaler = load_data(cfg_data)

    cfg_env  = EnvConfig()
    results  = {}

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
            seeds      = [42]
        )
        results[split_name] = r

    # ── Affichage ──────────────────────────────────────
    print("\n" + "="*65)
    print(f"  DÉTECTION OVERFITTING — {label}")
    print("="*65)
    print(f"{'Split':<10} {'Return':>10} {'B&H':>10} "
          f"{'Alpha':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-"*65)

    for split_name, m in results.items():
        print(
            f"{split_name:<10} "
            f"{m['return']:>+10.2%} "
            f"{m['bh']:>+10.2%} "
            f"{m['alpha']:>+10.2%} "
            f"{m['sharpe']:>10.3f} "
            f"{m['max_dd']:>10.2%}"
        )

    print("="*65)

    # ── Diagnostic ─────────────────────────────────────
    train_sharpe = results['Train']['sharpe']
    test_sharpe  = results['Test']['sharpe']
    gap          = train_sharpe - test_sharpe

    print(f"\n📊 Diagnostic {label} :")
    if gap > 1.0:
        print(f"  🔴 OVERFITTING — gap={gap:.2f}")
        print(f"     Train={train_sharpe:.2f} >> Test={test_sharpe:.2f}")
    elif gap > 0.5:
        print(f"  🟡 Légère sur-adaptation — gap={gap:.2f}")
        print(f"     Train={train_sharpe:.2f} > Test={test_sharpe:.2f}")
    else:
        print(f"  ✅ Pas d'overfitting — gap={gap:.2f}")
        print(f"     Train={train_sharpe:.2f} ≈ Test={test_sharpe:.2f}")

    return results


# ============================================================
# COMPARAISON SINGLE VS MULTI
# ============================================================
def compare_models(
    ticker     : str = "AAPL",
    start_date : str = "2018-01-01",
    end_date   : str = "2023-01-01",
    seeds      : list = [42, 123, 456, 789, 1337]
):
    """Compare single vs multi ticker sur le même test set."""

    cfg_data = DataConfig(
        ticker     = ticker,
        start_date = start_date,
        end_date   = end_date,
    )
    # ✅ 4 valeurs
    _, _, test_data, scaler = load_data(cfg_data)

    cfg_env = EnvConfig()

    print(f"\n{'='*70}")
    print(f"  COMPARAISON Single vs Multi — Test set {ticker}")
    print(f"{'='*70}\n")

    results = []

    for label, model_path in [
        ("Single (AAPL)",    "models/ppo_single/best_model.zip"),
        ("Multi (5 tickers)", "models/ppo_multi/best_model.zip"),
    ]:
        r = evaluate_one(
            model_path = model_path,
            test_data  = test_data,
            cfg_env    = cfg_env,
            label      = label,
            seeds      = seeds
        )
        results.append(r)

    # ── Tableau ────────────────────────────────────────
    print(f"{'Modèle':<22} {'Return':>10} {'±':>6} "
          f"{'B&H':>10} {'Alpha':>10} "
          f"{'Sharpe':>10} {'MaxDD':>10} {'Trades':>8}")
    print("-"*90)

    for r in results:
        print(
            f"{r['label']:<22} "
            f"{r['return']:>+10.2%} "
            f"{r['return_std']:>6.2%} "
            f"{r['bh']:>+10.2%} "
            f"{r['alpha']:>+10.2%} "
            f"{r['sharpe']:>10.3f} "
            f"{r['max_dd']:>10.2%} "
            f"{r['n_trades']:>8.0f}"
        )

    print("-"*90)

    # ── Actions ────────────────────────────────────────
    print(f"\n{'Modèle':<22} {'Hold':>10} {'Buy':>10} {'Sell':>10}")
    print("-"*55)
    for r in results:
        print(
            f"{r['label']:<22} "
            f"{r['hold_pct']:>10.1%} "
            f"{r['buy_pct']:>10.1%} "
            f"{r['sell_pct']:>10.1%}"
        )

    # ── Verdict ────────────────────────────────────────
    r_single = results[0]
    r_multi  = results[1]

    print(f"\n📊 Verdict :")
    winner_sharpe = "Multi" if r_multi['sharpe'] > r_single['sharpe'] else "Single"
    winner_alpha  = "Multi" if r_multi['alpha']  > r_single['alpha']  else "Single"
    winner_dd     = "Multi" if r_multi['max_dd'] < r_single['max_dd'] else "Single"

    print(f"  Sharpe   → {winner_sharpe} gagne "
          f"({r_multi['sharpe']:.3f} vs {r_single['sharpe']:.3f})")
    print(f"  Alpha    → {winner_alpha} gagne "
          f"({r_multi['alpha']:+.2%} vs {r_single['alpha']:+.2%})")
    print(f"  Max DD   → {winner_dd} gagne "
          f"({r_multi['max_dd']:.2%} vs {r_single['max_dd']:.2%})")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    TICKER     = "AAPL"
    START_DATE = "2018-01-01"
    END_DATE   = "2023-01-01"

    # ── 1. Overfitting single ───────────────────────────
    check_overfitting(
        model_path = "models/ppo_single/best_model.zip",
        label      = "Single (AAPL)",
        ticker     = TICKER,
        start_date = START_DATE,
        end_date   = END_DATE,
    )

    # ── 2. Overfitting multi ────────────────────────────
    check_overfitting(
        model_path = "models/ppo_multi/best_model.zip",
        label      = "Multi (5 tickers)",
        ticker     = TICKER,
        start_date = START_DATE,
        end_date   = END_DATE,
    )

    # ── 3. Comparaison finale ───────────────────────────
    compare_models(
        ticker     = TICKER,
        start_date = START_DATE,
        end_date   = END_DATE,
        seeds      = [42, 123, 456, 789, 1337]
    )