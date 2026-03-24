# Évalue sur les 3 splits pour détecter l'overfitting


from data_loader import DataConfig, load_data
from environment import TradingEnv, EnvConfig
from stable_baselines3 import PPO
import numpy as np

def check_overfitting(model_path, ticker, start_date, end_date):

    cfg_data = DataConfig(
        ticker     = ticker,
        start_date = start_date,
        end_date   = end_date,
    )
    train, val, test, _, _ = load_data(cfg_data)

    results = {}

    for split_name, split_data in [
        ("Train", train),
        ("Val",   val),
        ("Test",  test)
    ]:
        cfg_env = EnvConfig()
        env     = TradingEnv(data=split_data, cfg=cfg_env)
        model   = PPO.load(model_path)

        obs, _  = env.reset(seed=42)
        done    = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        portfolio = np.array(env.history["portfolio_values"])
        prices    = np.array(env.history["prices"])
        returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-8)

        results[split_name] = {
            "return" : (portfolio[-1] - portfolio[0]) / portfolio[0],
            "bh"     : (prices[-1] - prices[0]) / prices[0],
            "sharpe" : (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252),
            "max_dd" : np.max(
                (np.maximum.accumulate(portfolio) - portfolio)
                / (np.maximum.accumulate(portfolio) + 1e-8)
            ),
        }

    # Affichage
    print("\n" + "="*55)
    print("  DÉTECTION OVERFITTING")
    print("="*55)
    print(f"{'Split':<10} {'Return':>10} {'B&H':>10} "
          f"{'Alpha':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-"*55)

    for split, m in results.items():
        alpha = m['return'] - m['bh']
        print(
            f"{split:<10} "
            f"{m['return']:>+10.2%} "
            f"{m['bh']:>+10.2%} "
            f"{alpha:>+10.2%} "
            f"{m['sharpe']:>10.3f} "
            f"{m['max_dd']:>10.2%}"
        )

    print("="*55)

    # Diagnostic automatique
    train_sharpe = results['Train']['sharpe']
    test_sharpe  = results['Test']['sharpe']
    gap          = train_sharpe - test_sharpe

    print("\n📊 Diagnostic :")
    if gap > 1.0:
        print("  🔴 OVERFITTING détecté")
        print(f"     Sharpe Train={train_sharpe:.2f} >> Test={test_sharpe:.2f}")
        print("  → Réduis total_timesteps ou augmente ent_coef")
    elif gap > 0.5:
        print("  🟡 Légère sur-adaptation")
        print(f"     Sharpe Train={train_sharpe:.2f} > Test={test_sharpe:.2f}")
        print("  → Résultats acceptables")
    else:
        print("  ✅ Pas d'overfitting détecté")
        print(f"     Sharpe Train={train_sharpe:.2f} ≈ Test={test_sharpe:.2f}")

    return results


if __name__ == "__main__":
    check_overfitting(
        model_path = "models/ppo_aapl.zip",
        ticker     = "AAPL",
        start_date = "2019-01-01",
        end_date   = "2023-01-01",
    )