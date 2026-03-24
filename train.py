# train.py
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_loader import load_data, DataConfig
from environment import TradingEnv, EnvConfig


# ============================================================
# CONFIG TRAINING
# ============================================================
@dataclass
class TrainConfig:

    # Algorithme
    algo             : str   = "PPO"       # "PPO" ou "A2C"

    # PPO Hyperparams
    learning_rate : float = 1e-4  # Vitesse d'apprentissage
                                  # Trop grand → instable
                                  # Trop petit → lent

    n_steps       : int   = 1024  # Steps collectés avant update
                                  # Plus grand = plus stable mais + lent

    batch_size    : int   = 128    # Taille des mini-batchs
                                  # Plus petit = + de bruit = + d'exploration

    n_epochs      : int   = 10    # Combien de fois on réutilise les données
                                  # PPO permet de les réutiliser (≠ DQN)

    gamma         : float = 0.99  # Importance du futur
                                  # 0.99 → l'agent pense à ~100 steps
                                  # 0.9  → l'agent pense à ~10 steps

    gae_lambda    : float = 0.95  # Biais/variance tradeoff des avantages
                                  # 1.0 = pas de biais mais + de variance
                                  # 0.0 = biais mais variance nulle

    clip_range    : float = 0.2   # Le "garde-fou" PPO
                                  # Change max de 20% par update

    ent_coef      : float = 0.05 # Bonus d'exploration
                                  # Force l'agent à ne pas trop vite
                                  # se spécialiser sur une seule action

     # Training
    total_timesteps  : int   = 200_000     # Steps totaux
    n_envs           : int   = 4           # Environnements parallèles

    # Sauvegarde
    save_dir         : str   = "models/"
    log_dir          : str   = "logs/"
    model_name       : str   = "ppo_trading"


# ============================================================
# CALLBACK CUSTOM : LOG MÉTRIQUES FINANCIÈRES
# ============================================================
class FinancialMetricsCallback(BaseCallback):
    """
    Callback qui log les métriques financières pendant le training :
    - Portfolio value
    - Sharpe Ratio
    - Drawdown
    - Nb trades
    """

    def __init__(self, eval_env: TradingEnv, eval_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env   = eval_env
        self.eval_freq  = eval_freq

        # Historique des métriques
        self.metrics_history = {
            "timesteps"       : [],
            "mean_return"     : [],
            "sharpe"          : [],
            "max_drawdown"    : [],
            "n_trades"        : [],
            "vs_buy_hold"     : [],
        }

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq != 0:
            return True

        # Run un épisode complet sur eval_env
        obs, _   = self.eval_env.reset()
        done     = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated

        # Récupère les métriques
        portfolio = np.array(self.eval_env.history["portfolio_values"])
        prices    = np.array(self.eval_env.history["prices"])

        if len(portfolio) < 2:
            return True

        returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-8)
        sharpe    = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        peak      = np.maximum.accumulate(portfolio)
        max_dd    = np.max((peak - portfolio) / (peak + 1e-8))

        total_ret = (portfolio[-1] - portfolio[0])  / portfolio[0]
        bh_ret    = (prices[-1]    - prices[0])     / prices[0]
        vs_bh     = total_ret - bh_ret

        n_trades  = self.eval_env.history["n_trades"]

        # Stocke
        self.metrics_history["timesteps"].append(self.num_timesteps)
        self.metrics_history["mean_return"].append(total_ret)
        self.metrics_history["sharpe"].append(sharpe)
        self.metrics_history["max_drawdown"].append(max_dd)
        self.metrics_history["n_trades"].append(n_trades)
        self.metrics_history["vs_buy_hold"].append(vs_bh)

        if self.verbose:
            print(
                f"\n📊 Step {self.num_timesteps:,} | "
                f"Return: {total_ret:+.2%} | "
                f"Sharpe: {sharpe:.2f} | "
                f"DD: {max_dd:.2%} | "
                f"vs B&H: {vs_bh:+.2%} | "
                f"Trades: {n_trades}"
            )

        return True


# ============================================================
# FONCTION DE TRAINING PRINCIPALE
# ============================================================
# ============================================================
# FONCTION DE TRAINING PRINCIPALE — VERSION CORRIGÉE
# ============================================================
def train(
    train_data : pd.DataFrame,
    val_data   : pd.DataFrame,
    cfg_env    : EnvConfig    = EnvConfig(),
    cfg_train  : TrainConfig  = TrainConfig(),
) -> PPO:

    os.makedirs(cfg_train.save_dir, exist_ok=True)
    os.makedirs(cfg_train.log_dir,  exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  DRL Portfolio — Training {cfg_train.algo}")
    print(f"{'='*55}\n")

    # ---- 1. Train env ----
    def make_train_env():
        env = TradingEnv(data=train_data, cfg=cfg_env)
        env = Monitor(env, cfg_train.log_dir)
        return env

    train_env = DummyVecEnv([make_train_env for _ in range(cfg_train.n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0
    )

    # ---- 2. Eval env (doit être wrappé pareil que train_env) ----
    def make_eval_env():
        env = TradingEnv(data=val_data, cfg=cfg_env)
        env = Monitor(env, cfg_train.log_dir)
        return env

    # ✅ FIX : VecNormalize sur l'eval env aussi
    eval_env_sb3 = DummyVecEnv([make_eval_env])
    eval_env_sb3 = VecNormalize(
        eval_env_sb3,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        training    = False   # ← Important : pas de update des stats sur eval
    )

    # Eval env séparé pour nos métriques custom (pas de VecNormalize)
    eval_env_custom = TradingEnv(
        data        = val_data,
        cfg         = cfg_env,
        render_mode = "human"
    )

    # ---- 3. Modèle PPO ----
    policy_kwargs = dict(
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
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
            verbose         = 0
        )

    # ---- 4. Callbacks ----

    # a) Métriques financières custom
    financial_cb = FinancialMetricsCallback(
        eval_env  = eval_env_custom,
        eval_freq = 10_000,
        verbose   = 1
    )

    # b) ✅ EvalCallback avec eval_env wrappé correctement
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

    # ---- 5. Training ----
    print(f"🚀 Début du training...")
    print(f"   Algo            : {cfg_train.algo}")
    print(f"   Total steps     : {cfg_train.total_timesteps:,}")
    print(f"   Envs parallèles : {cfg_train.n_envs}")
    print(f"   Learning rate   : {cfg_train.learning_rate}")
    print(f"   Net arch        : [64, 64]\n")

    model.learn(
        total_timesteps = cfg_train.total_timesteps,
        callback        = callbacks,
        progress_bar    = True
    )

    # ---- 6. Sauvegarde ----
    model_path = os.path.join(cfg_train.save_dir, cfg_train.model_name)
    model.save(model_path)
    train_env.save(os.path.join(cfg_train.save_dir, "vec_normalize.pkl"))

    print(f"\n✅ Modèle sauvegardé : {model_path}")

    return model, train_env, financial_cb.metrics_history

# ============================================================
# PLOT TRAINING CURVES
# ============================================================
def plot_training_curves(metrics_history: dict):
    """Visualise l'évolution des métriques pendant le training."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Training Curves — DRL Portfolio", fontsize=13)

    steps = metrics_history["timesteps"]

    # Return vs Buy & Hold
    axes[0].plot(steps, metrics_history["mean_return"],
                 color='#2ecc71', label='Agent Return', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[0].set_title("Return Total")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Sharpe
    axes[1].plot(steps, metrics_history["sharpe"],
                 color='#3498db', label='Sharpe Ratio', linewidth=2)
    axes[1].axhline(1, color='green', linestyle='--',
                    linewidth=1, label='Sharpe = 1')
    axes[1].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[1].set_title("Sharpe Ratio")
    axes[1].set_ylabel("Sharpe")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # vs Buy & Hold
    axes[2].plot(steps, metrics_history["vs_buy_hold"],
                 color='#e67e22', label='Alpha vs B&H', linewidth=2)
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
    axes[2].set_ylabel("Alpha")
    axes[2].set_xlabel("Timesteps")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # 1. Données
    cfg_data = DataConfig(ticker="AAPL")
    train_data, val_data, test_data, scaler, gmm = load_data(cfg_data)

    # 2. Config environnement
    cfg_env = EnvConfig(
        initial_capital  = 10_000.0,
        transaction_cost = 0.001,
        window_size      = 10,
        max_drawdown_pct = 0.25,
        reward_scaling   = 100.0
    )

    # 3. Config training
    cfg_train = TrainConfig(
        algo            = "PPO",
        total_timesteps = 200_000,
        n_envs          = 4,
        learning_rate   = 1e-4,
        ent_coef        = 0.10,
        model_name      = "ppo_aapl"
    )

    # 4. Training
    model, train_env, metrics = train(
        train_data = train_data,
        val_data   = val_data,
        cfg_env    = cfg_env,
        cfg_train  = cfg_train,
    )

    # 5. Courbes de training
    plot_training_curves(metrics)
