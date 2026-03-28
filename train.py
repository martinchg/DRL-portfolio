# train.py
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_loader import load_data, load_multi_ticker_data, DataConfig, FEATURES
from environment import TradingEnv, EnvConfig


# ============================================================
# SEEDS
# ============================================================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)
    print(f"🌱 Seeds fixés à {seed}")


# ============================================================
# CONFIG TRAINING
# ============================================================
@dataclass
class TrainConfig:

    algo             : str   = "PPO"

    # Hyperparams — bon compromis vitesse/généralisation
    learning_rate    : float = 3e-4    # Remonté à la valeur standard
    n_steps          : int   = 1024    # Compromis entre 512 et 2048
    batch_size       : int   = 128     # Compromis entre 64 et 256
    n_epochs         : int   = 7       # Compromis entre 5 et 10
    gamma            : float = 0.99
    gae_lambda       : float = 0.95
    clip_range       : float = 0.2
    ent_coef         : float = 0.02    # Léger bonus exploration

    # Training — plus de steps pour mieux généraliser
    total_timesteps  : int   = 500_000

    n_envs           : int   = 4

    # Sauvegarde
    save_dir         : str   = "models/"
    log_dir          : str   = "logs/"
    model_name       : str   = "ppo_trading"


# ============================================================
# CALLBACK CUSTOM
# ============================================================
class FinancialMetricsCallback(BaseCallback):

    def __init__(
        self,
        eval_env  : TradingEnv,
        eval_freq : int = 10_000,
        verbose   : int = 1
    ):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq

        self.metrics_history = {
            "timesteps"    : [],
            "mean_return"  : [],
            "sharpe"       : [],
            "max_drawdown" : [],
            "n_trades"     : [],
            "vs_buy_hold"  : [],
        }

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq != 0:
            return True

        obs, _ = self.eval_env.reset()
        done   = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated

        portfolio = np.array(self.eval_env.history["portfolio_values"])
        prices    = np.array(self.eval_env.history["prices"])

        if len(portfolio) < 2:
            return True

        returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-8)
        sharpe    = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        peak      = np.maximum.accumulate(portfolio)
        max_dd    = np.max((peak - portfolio) / (peak + 1e-8))
        total_ret = (portfolio[-1] - portfolio[0]) / portfolio[0]
        bh_ret    = (prices[-1]    - prices[0])    / prices[0]

        self.metrics_history["timesteps"].append(self.num_timesteps)
        self.metrics_history["mean_return"].append(total_ret)
        self.metrics_history["sharpe"].append(sharpe)
        self.metrics_history["max_drawdown"].append(max_dd)
        self.metrics_history["n_trades"].append(
            self.eval_env.history["n_trades"]
        )
        self.metrics_history["vs_buy_hold"].append(total_ret - bh_ret)

        if self.verbose:
            print(
                f"\n📊 Step {self.num_timesteps:,} | "
                f"Return: {total_ret:+.2%} | "
                f"Sharpe: {sharpe:.2f} | "
                f"DD: {max_dd:.2%} | "
                f"vs B&H: {total_ret - bh_ret:+.2%} | "
                f"Trades: {self.eval_env.history['n_trades']}"
            )

        return True


# ============================================================
# TRAINING
# ============================================================
def train(
    train_data : pd.DataFrame,
    val_data   : pd.DataFrame,
    cfg_env    : EnvConfig   = EnvConfig(),
    cfg_train  : TrainConfig = TrainConfig(),
) -> Tuple[PPO, VecNormalize, dict]:

    os.makedirs(cfg_train.save_dir, exist_ok=True)
    os.makedirs(cfg_train.log_dir,  exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Training {cfg_train.algo} — {cfg_train.model_name}")
    print(f"{'='*55}\n")

    # ---- Train env ----
    def make_train_env():
        env = TradingEnv(data=train_data, cfg=cfg_env)
        env = Monitor(env, cfg_train.log_dir)
        return env

    train_env = DummyVecEnv(
        [make_train_env for _ in range(cfg_train.n_envs)]
    )
    train_env = VecNormalize(
        train_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0
    )

    # ---- Eval env SB3 ----
    def make_eval_env():
        env = TradingEnv(data=val_data, cfg=cfg_env)
        env = Monitor(env, cfg_train.log_dir)
        return env

    eval_env_sb3 = DummyVecEnv([make_eval_env])
    eval_env_sb3 = VecNormalize(
        eval_env_sb3,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        training    = False
    )

    # ---- Eval env custom ----
    eval_env_custom = TradingEnv(
        data        = val_data,
        cfg         = cfg_env,
        render_mode = "human"
    )

    # ---- Modèle ----
    # Réseau [128,128] → compromis expressivité/overfitting
    policy_kwargs = dict(
        net_arch = [dict(pi=[128, 64], vf=[128, 64])]
    )

    if cfg_train.algo == "PPO":
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = cfg_train.learning_rate,
            n_steps         = cfg_train.n_steps,
            batch_size      = cfg_train.batch_size,
            n_epochs        = cfg_train.n_epochs,
            gamma           = cfg_train.gamma,
            gae_lambda      = cfg_train.gae_lambda,
            clip_range      = cfg_train.clip_range,
            ent_coef        = cfg_train.ent_coef,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = cfg_train.log_dir,
            seed            = 42,
            verbose         = 0
        )
    else:
        model = A2C(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = cfg_train.learning_rate,
            gamma           = cfg_train.gamma,
            gae_lambda      = cfg_train.gae_lambda,
            ent_coef        = cfg_train.ent_coef,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = cfg_train.log_dir,
            seed            = 42,
            verbose         = 0
        )

    # ---- Callbacks ----
    financial_cb = FinancialMetricsCallback(
        eval_env  = eval_env_custom,
        eval_freq = 10_000,
        verbose   = 1
    )

    eval_cb = EvalCallback(
        eval_env             = eval_env_sb3,
        best_model_save_path = cfg_train.save_dir,
        log_path             = cfg_train.log_dir,
        eval_freq            = 10_000,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 0
    )

    callbacks = CallbackList([financial_cb, eval_cb])

    # ---- Launch ----
    print(f"   Algo            : {cfg_train.algo}")
    print(f"   Total steps     : {cfg_train.total_timesteps:,}")
    print(f"   Envs parallèles : {cfg_train.n_envs}")
    print(f"   Learning rate   : {cfg_train.learning_rate}")
    print(f"   ent_coef        : {cfg_train.ent_coef}")
    print(f"   Net arch        : [128, 64]")
    print(f"   n_steps         : {cfg_train.n_steps}")
    print(f"   batch_size      : {cfg_train.batch_size}\n")

    model.learn(
        total_timesteps = cfg_train.total_timesteps,
        callback        = callbacks,
        progress_bar    = True
    )

    # ---- Sauvegarde ----
    model_path = os.path.join(cfg_train.save_dir, cfg_train.model_name)
    model.save(model_path)
    train_env.save(
        os.path.join(cfg_train.save_dir, "vec_normalize.pkl")
    )
    print(f"\n✅ Sauvegardé : {model_path}.zip")

    return model, train_env, financial_cb.metrics_history


# ============================================================
# PLOT
# ============================================================
def plot_training_curves(metrics_history: dict, title: str = ""):

    if not metrics_history["timesteps"]:
        print("⚠️  Pas assez de données pour les courbes")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Training Curves — {title}", fontsize=13)

    steps = metrics_history["timesteps"]

    axes[0].plot(steps, metrics_history["mean_return"],
                 color='#2ecc71', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[0].set_title("Return Total")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, metrics_history["sharpe"],
                 color='#3498db', linewidth=2)
    axes[1].axhline(1, color='green', linestyle='--', linewidth=1)
    axes[1].axhline(0, color='gray',  linestyle=':', linewidth=1)
    axes[1].set_title("Sharpe Ratio")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, metrics_history["vs_buy_hold"],
                 color='#e67e22', linewidth=2)
    axes[2].fill_between(
        steps, metrics_history["vs_buy_hold"], 0,
        where=[v > 0 for v in metrics_history["vs_buy_hold"]],
        color='#2ecc71', alpha=0.2
    )
    axes[2].fill_between(
        steps, metrics_history["vs_buy_hold"], 0,
        where=[v <= 0 for v in metrics_history["vs_buy_hold"]],
        color='#e74c3c', alpha=0.2
    )
    axes[2].axhline(0, color='white', linewidth=0.8)
    axes[2].set_title("Alpha vs Buy & Hold")
    axes[2].set_xlabel("Timesteps")
    axes[2].grid(alpha=0.3)

    os.makedirs("logs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f"logs/training_curves_{title}.png",
        dpi=150, bbox_inches='tight'
    )
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    set_all_seeds(42)

    cfg_env = EnvConfig(
        initial_capital  = 10_000.0,
        transaction_cost = 0.001,
        window_size      = 10,
        max_drawdown_pct = 0.25,
        reward_scaling   = 100.0
    )

    # ── Single Ticker ──────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING 1/2 — Single Ticker (AAPL)")
    print("="*55)

    cfg_data_single = DataConfig(
        ticker     = "AAPL",
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    train_single, val_single, test_single, scaler = load_data(
        cfg_data_single
    )

    cfg_train_single = TrainConfig(
        total_timesteps = 500_000,
        model_name      = "ppo_single",
        save_dir        = "models/ppo_single/",
        log_dir         = "logs/single/",
    )

    model_single, _, metrics_single = train(
        train_data = train_single,
        val_data   = val_single,
        cfg_env    = cfg_env,
        cfg_train  = cfg_train_single,
    )
    plot_training_curves(metrics_single, title="Single_AAPL")

    # ── Multi Ticker ───────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING 2/2 — Multi Ticker")
    print("="*55)

    cfg_data_multi = DataConfig(
        tickers    = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"],
        start_date = "2018-01-01",
        end_date   = "2023-01-01",
    )
    train_multi, val_multi, _ = load_multi_ticker_data(cfg_data_multi)

    cfg_train_multi = TrainConfig(
        total_timesteps = 500_000,
        model_name      = "ppo_multi",
        save_dir        = "models/ppo_multi/",
        log_dir         = "logs/multi/",
    )

    model_multi, _, metrics_multi = train(
        train_data = train_multi,
        val_data   = val_multi,
        cfg_env    = cfg_env,
        cfg_train  = cfg_train_multi,
    )
    plot_training_curves(metrics_multi, title="Multi_5tickers")

    # ── Résumé ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  ✅ TRAINING TERMINÉ")
    print("="*55)
    print("  ├── models/ppo_single/best_model.zip")
    print("  └── models/ppo_multi/best_model.zip")
    print("  → python evaluate.py")
    print("="*55)