# train.py
import os
import random
from copy import deepcopy
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

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

    # ── Hyperparamètres PPO ───────────────────────────────────────────
    # [TECHNIQUE] Vitesse d'apprentissage du réseau de neurones.
    # 1e-4 = apprentissage lent mais stable  |  3e-4 = standard
    # 1e-3 = apprentissage rapide mais instable (peut diverger)
    learning_rate    : float = 3e-4

    # [TECHNIQUE] Nombre de steps collectés avant chaque mise à jour du réseau.
    # Avec 4 envs parallèles, chaque update utilise 4 × n_steps = 4096 transitions.
    # 512  = updates fréquentes (plus réactif, plus bruité)
    # 1024 = par défaut
    # 2048 = updates rares (plus stable, mais apprend moins vite)
    n_steps          : int   = 1024

    # [TECHNIQUE] Taille des mini-batches lors des mises à jour.
    # Doit diviser exactement n_steps × n_envs (4096).
    # Valeurs valides : 64, 128, 256, 512
    batch_size       : int   = 128

    # [TECHNIQUE] Nombre de fois qu'on réutilise les mêmes transitions pour mettre à jour.
    # Bas (3-4) = moins d'overfitting sur les données récentes
    # Élevé (7-10) = plus d'optimisation par batch, risque de surfit
    n_epochs         : int   = 4

    # [GESTIONNAIRE] Horizon de planification de l'agent.
    # gamma = 0.99 → l'agent "voit" environ 1/(1-0.99) = 100 steps dans le futur
    # gamma = 0.95 → horizon ~20 steps (court-terme)
    # gamma = 0.999 → horizon ~1000 steps (très long-terme)
    # Pour le trading journalier, 0.99 (~100 jours) est un bon compromis.
    gamma            : float = 0.99

    # [TECHNIQUE] GAE lambda — équilibre biais/variance dans l'estimation des avantages.
    # 0.9 → plus biaisé, moins de variance (apprentissage stable)
    # 1.0 → non biaisé, haute variance (Monte Carlo pur)
    # 0.95 = valeur standard recommandée dans la littérature PPO
    gae_lambda       : float = 0.95

    # [TECHNIQUE] Amplitude maximale d'une mise à jour de politique (PPO clipping).
    # 0.1 = très conservatif (updates petits, apprentissage lent)
    # 0.2 = valeur standard PPO
    # Réduit à 0.15 pour éviter les changements de politique trop brusques
    clip_range       : float = 0.15

    # [GESTIONNAIRE] Bonus d'exploration : encourage l'agent à essayer les 3 actions.
    # 0.0  = pas d'exploration forcée (l'agent exploite toujours)
    # 0.01 = très faible (biais fort vers l'exploitation)
    # 0.05 = équilibre exploration/exploitation
    # 0.1  = beaucoup d'exploration (utile si l'agent bloque sur Hold ou Buy)
    # Si l'agent fait trop de Hold → augmenter | Si trop de trades → diminuer
    ent_coef         : float = 0.05

    # ── Durée d'entraînement ─────────────────────────────────────────
    # [GESTIONNAIRE] Nombre total de steps d'interaction avec l'environnement.
    # 500_000 = ~15 min sur CPU, résultats moyens
    # 800_000 = par défaut
    # 1_500_000 = meilleurs résultats mais ~30-40 min sur CPU
    # Augmenter si les courbes d'entraînement ne convergent pas encore.
    total_timesteps  : int   = 800_000

    # [TECHNIQUE] Nombre d'environnements parallèles (accélère la collecte de données).
    # Augmenter si vous avez plusieurs CPU disponibles.
    n_envs           : int   = 4

    # ── Reproductibilité ──────────────────────────────────────────────
    # [TECHNIQUE] Seed passé à SB3 (réseau, envs, tirages d'actions).
    # Le faire varier mesure la sensibilité de l'entraînement au hasard.
    seed             : int   = 42

    # ── Sauvegarde ────────────────────────────────────────────────────
    save_dir         : str   = "models/"
    log_dir          : str   = "logs/"
    model_name       : str   = "ppo_trading"


# ============================================================
# CALLBACK CUSTOM
# ============================================================
class FinancialMetricsCallback(BaseCallback):
    """
    Callback qui évalue les métriques financières pendant l'entraînement.

    Corrections vs version précédente :
    1. Les observations sont normalisées avec les stats VecNormalize du train_env
       avant d'être passées à model.predict() — sinon le modèle reçoit des obs
       brutes alors qu'il a été entraîné sur des obs normalisées (bug critique).
    2. La moyenne est faite sur n_eval_episodes épisodes pour réduire le bruit
       (un seul épisode avec départ aléatoire = Sharpe très instable).
    """

    def __init__(
        self,
        eval_env       : TradingEnv,
        eval_freq      : int = 10_000,
        n_eval_episodes: int = 5,
        eval_env_vec   : VecNormalize = None,
        verbose        : int = 1
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env_vec    = eval_env_vec

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

        # Récupère l'env d'entraînement (VecNormalize) pour normaliser les obs
        train_vec_env = self.model.get_env()

        # Synchronise les stats de normalisation vers l'env d'évaluation SB3
        # (EvalCallback reçoit des obs normalisées cohérentes avec le train)
        if self.eval_env_vec is not None:
            sync_envs_normalization(train_vec_env, self.eval_env_vec)

        all_returns = []
        all_sharpes = []
        all_dds     = []
        all_trades  = []
        all_alphas  = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done   = False

            while not done:
                # ✅ Normalise l'obs avec les stats du VecNormalize d'entraînement
                # Sans ça, le modèle reçoit des obs brutes alors qu'il attend
                # des obs normalisées → prédictions incohérentes
                obs_norm = train_vec_env.normalize_obs(
                    np.array([obs], dtype=np.float32)
                )[0]

                action, _ = self.model.predict(obs_norm, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(int(action))
                done = terminated or truncated

            portfolio = np.array(self.eval_env.history["portfolio_values"])
            prices    = np.array(self.eval_env.history["prices"])

            if len(portfolio) < 2:
                continue

            returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-8)
            sharpe    = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
            peak      = np.maximum.accumulate(portfolio)
            max_dd    = np.max((peak - portfolio) / (peak + 1e-8))
            total_ret = (portfolio[-1] - portfolio[0]) / portfolio[0]
            bh_ret    = (prices[-1]    - prices[0])    / prices[0]

            all_returns.append(total_ret)
            all_sharpes.append(sharpe)
            all_dds.append(max_dd)
            all_trades.append(self.eval_env.history["n_trades"])
            all_alphas.append(total_ret - bh_ret)

        if not all_returns:
            return True

        mean_return = float(np.mean(all_returns))
        mean_sharpe = float(np.mean(all_sharpes))
        mean_dd     = float(np.mean(all_dds))
        mean_trades = float(np.mean(all_trades))
        mean_alpha  = float(np.mean(all_alphas))

        self.metrics_history["timesteps"].append(self.num_timesteps)
        self.metrics_history["mean_return"].append(mean_return)
        self.metrics_history["sharpe"].append(mean_sharpe)
        self.metrics_history["max_drawdown"].append(mean_dd)
        self.metrics_history["n_trades"].append(mean_trades)
        self.metrics_history["vs_buy_hold"].append(mean_alpha)

        if self.verbose:
            print(
                f"\n📊 Step {self.num_timesteps:,} | "
                f"Return: {mean_return:+.2%} | "
                f"Sharpe: {mean_sharpe:.2f} | "
                f"DD: {mean_dd:.2%} | "
                f"vs B&H: {mean_alpha:+.2%} | "
                f"Trades: {mean_trades:.1f}"
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
    # Réseau réduit [64, 64] : 5 features × 10 steps = 52 inputs
    # [128, 64] était surdimensionné → mémorisation facile du train set
    policy_kwargs = dict(
        net_arch = dict(pi=[64, 64], vf=[64, 64])
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
            seed            = cfg_train.seed,
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
            seed            = cfg_train.seed,
            verbose         = 0
        )

    # ---- Callbacks ----
    financial_cb = FinancialMetricsCallback(
        eval_env     = eval_env_custom,
        eval_freq    = 10_000,
        eval_env_vec = eval_env_sb3,
        verbose      = 1
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
    print(f"   Net arch        : [64, 64]")
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
    """
    3 graphiques indépendants — chacun mesure une dimension différente :

    1. Alpha vs B&H   → est-ce que l'agent SURPERFORME le marché ?
                         vert = oui / rouge = non
                         C'est la métrique principale à optimiser.

    2. Return absolu  → combien l'agent a-t-il gagné (ou perdu) en absolu ?
                         La ligne 0% sépare gain/perte réelle.
                         Indépendant du marché — complémentaire à l'alpha.

    3. Max Drawdown   → quelle est la perte maximale depuis un pic ?
                         0% = aucune perte depuis le pic (idéal)
                         25%+ = l'agent a perdu jusqu'à 25% de sa valeur max
                         Mesure le RISQUE, pas la performance.
    """

    if not metrics_history["timesteps"]:
        print("⚠️  Pas assez de données pour les courbes")
        return

    steps   = metrics_history["timesteps"]
    alpha   = metrics_history["vs_buy_hold"]
    ret     = metrics_history["mean_return"]
    max_dd  = metrics_history["max_drawdown"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Training Curves — {title}", fontsize=13, fontweight='bold')

    # ── Subplot 1 : Alpha vs B&H ────────────────────────────────
    ax = axes[0]
    ax.plot(steps, alpha, color='#e67e22', linewidth=2)
    ax.fill_between(steps, alpha, 0,
                    where=[v > 0 for v in alpha],
                    color='#2ecc71', alpha=0.25, label='Bat le B&H')
    ax.fill_between(steps, alpha, 0,
                    where=[v <= 0 for v in alpha],
                    color='#e74c3c', alpha=0.25, label='Sous-performe le B&H')
    ax.axhline(0, color='white', linewidth=1, linestyle='--')
    ax.set_title("① Alpha vs Buy & Hold  (positif = bat le marché)", fontsize=10)
    ax.set_ylabel("Alpha")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ── Subplot 2 : Return absolu ───────────────────────────────
    ax = axes[1]
    ax.plot(steps, ret, color='#3498db', linewidth=2)
    ax.fill_between(steps, ret, 0,
                    where=[v > 0 for v in ret],
                    color='#3498db', alpha=0.15, label='Gain absolu')
    ax.fill_between(steps, ret, 0,
                    where=[v <= 0 for v in ret],
                    color='#e74c3c', alpha=0.15, label='Perte absolue')
    ax.axhline(0, color='gray', linewidth=1, linestyle=':')
    ax.set_title("② Return Total  (indépendant du marché)", fontsize=10)
    ax.set_ylabel("Return")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ── Subplot 3 : Max Drawdown ────────────────────────────────
    ax = axes[2]
    ax.plot(steps, max_dd, color='#e74c3c', linewidth=2)
    ax.fill_between(steps, max_dd, 0, color='#e74c3c', alpha=0.2)
    ax.axhline(0.25, color='orange', linewidth=1, linestyle='--',
               label='Seuil arrêt épisode (25%)')
    ax.axhline(0.10, color='yellow', linewidth=1, linestyle=':',
               label='Alerte drawdown (10%)')
    ax.invert_yaxis()   # 0% en haut (meilleur) → valeurs élevées vers le bas
    ax.set_title("③ Max Drawdown  (0% = pas de perte depuis pic — plus bas = mieux)",
                 fontsize=10)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Timesteps")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

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

    # Dates par défaut de DataConfig (2010→2023) : plusieurs cycles de marché,
    # cf. recommandations documentées dans data_loader.py
    cfg_data_single = DataConfig(ticker="AAPL")
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
        tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA"],
    )
    train_multi, val_multi, _ = load_multi_ticker_data(cfg_data_multi)

    cfg_train_multi = TrainConfig(
        total_timesteps = 800_000,
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