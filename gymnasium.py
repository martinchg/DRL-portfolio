import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


# ============================================================
# CONFIG DE L'ENVIRONNEMENT
# ============================================================
@dataclass
class EnvConfig:
    # Portfolio
    initial_capital:   float = 10_000.0   # Capital de départ ($)
    transaction_cost:  float = 0.001      # 0.1% par trade (frais réalistes)
    
    # Observation
    window_size:       int   = 10         # Nombre de jours que l'agent "voit"
    
    # Reward
    reward_scaling:    float = 100.0      # Scaling du reward pour stabiliser PPO
    
    # Risk Management
    max_drawdown_pct:  float = 0.25       # Stop si drawdown > 25%
    
    # Actions
    # 0 = Hold | 1 = Buy (long 100%) | 2 = Sell (flat)
    n_actions:         int   = 3


FEATURES = [
    'log_returns',
    'volatility',
    'rsi',
    'dist_to_sma',
    'market_regime',
]


# ============================================================
# ENVIRONNEMENT DE TRADING
# ============================================================
class TradingEnv(gym.Env):
    """
    Environnement de trading single-asset pour DRL.
    
    Observation Space :
        Fenêtre glissante de `window_size` jours × 5 features
        + 2 variables de portefeuille (position, unrealized_pnl)
        
    Action Space :
        Discret(3) → 0: Hold | 1: Buy | 2: Sell
        
    Reward :
        Sharpe-like reward : rendement ajusté par la volatilité récente
        Pénalité sur les frais de transaction
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        data:       pd.DataFrame,
        cfg:        EnvConfig = EnvConfig(),
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.cfg         = cfg
        self.render_mode = render_mode
        
        # ---- Données ----
        self._validate_data(data)
        self.data        = data.reset_index(drop=True)
        self.prices      = self.data['price'].values.astype(np.float32)
        self.features    = self.data[FEATURES].values.astype(np.float32)
        self.n_steps     = len(self.data)
        
        # ---- Spaces ----
        # Observation : (window_size × n_features) + portfolio_state
        n_obs = cfg.window_size * len(FEATURES) + 2  # +2 : position + pnl
        
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (n_obs,),
            dtype = np.float32
        )
        
        self.action_space = spaces.Discrete(cfg.n_actions)
        
        # ---- Historique pour render ----
        self._reset_history()
        
        # ---- State interne ----
        self._current_step   = 0
        self._position       = 0        # 0 = flat | 1 = long
        self._entry_price    = 0.0
        self._portfolio_value = cfg.initial_capital
        self._peak_value     = cfg.initial_capital
        self._cash           = cfg.initial_capital
        self._shares         = 0.0
        
        print(f"✅ TradingEnv initialisé")
        print(f"   Steps disponibles : {self.n_steps - cfg.window_size}")
        print(f"   Observation shape : {self.observation_space.shape}")
        print(f"   Actions           : {{0: Hold, 1: Buy, 2: Sell}}")
    
    
    # ============================================================
    # RESET
    # ============================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        
        super().reset(seed=seed)
        
        # On commence après window_size pour avoir assez d'historique
        self._current_step    = self.cfg.window_size
        self._position        = 0
        self._entry_price     = 0.0
        self._cash            = self.cfg.initial_capital
        self._shares          = 0.0
        self._portfolio_value = self.cfg.initial_capital
        self._peak_value      = self.cfg.initial_capital
        
        self._reset_history()
        
        obs  = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    
    # ============================================================
    # STEP
    # ============================================================
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        
        assert self.action_space.contains(action), f"Action invalide : {action}"
        
        current_price = self.prices[self._current_step]
        prev_value    = self._portfolio_value
        
        # --- Exécution de l'action ---
        transaction_cost = self._execute_action(action, current_price)
        
        # --- Mise à jour de la valeur du portfolio ---
        self._portfolio_value = self._cash + self._shares * current_price
        
        # --- Mise à jour du peak (pour drawdown) ---
        self._peak_value = max(self._peak_value, self._portfolio_value)
        
        # --- Calcul du reward ---
        reward = self._compute_reward(
            prev_value       = prev_value,
            current_value    = self._portfolio_value,
            transaction_cost = transaction_cost
        )
        
        # --- Logging ---
        self._update_history(action, current_price, reward)
        
        # --- Avance dans le temps ---
        self._current_step += 1
        
        # --- Conditions de fin d'épisode ---
        terminated = self._is_terminated()
        truncated  = self._current_step >= self.n_steps - 1
        
        obs  = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    
    # ============================================================
    # EXÉCUTION DES ACTIONS
    # ============================================================
    def _execute_action(self, action: int, price: float) -> float:
        """
        Exécute l'action et retourne le coût de transaction.
        
        0 = Hold → ne rien faire
        1 = Buy  → investir tout le cash disponible
        2 = Sell → liquider toute la position
        """
        cost = 0.0
        
        # BUY : on est flat et on veut acheter
        if action == 1 and self._position == 0:
            # Nombre de shares achetables (frais inclus)
            cost        = self._cash * self.cfg.transaction_cost
            investable  = self._cash - cost
            self._shares    = investable / price
            self._cash      = 0.0
            self._position  = 1
            self._entry_price = price
        
        # SELL : on est long et on veut vendre
        elif action == 2 and self._position == 1:
            proceeds    = self._shares * price
            cost        = proceeds * self.cfg.transaction_cost
            self._cash      = proceeds - cost
            self._shares    = 0.0
            self._position  = 0
            self._entry_price = 0.0
        
        # HOLD ou action redondante → rien
        return cost
    
    
    # ============================================================
    # REWARD FUNCTION
    # ============================================================
    def _compute_reward(
        self,
        prev_value:       float,
        current_value:    float,
        transaction_cost: float
    ) -> float:
        """
        Reward = Sharpe-like reward
        
        Formule :
            r_t = (V_t - V_{t-1}) / V_{t-1}   ← rendement du portfolio
            reward = r_t / σ_recent             ← ajusté par la volatilité
            - pénalité_transaction              ← décourage le overtrading
        
        Pourquoi ce reward ?
        - Encourage les rendements RÉGULIERS (pas juste les gros gains)
        - Pénalise les pertes brutales
        - Décourage le trading excessif (frais)
        """
        # Rendement brut du portefeuille
        portfolio_return = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Volatilité récente sur la fenêtre d'observation
        recent_vol = self._get_recent_volatility()
        
        # Sharpe-like : rendement / risque
        if recent_vol > 1e-8:
            sharpe_reward = portfolio_return / recent_vol
        else:
            sharpe_reward = portfolio_return
        
        # Pénalité de transaction (normalisée)
        tx_penalty = (
            transaction_cost / self.cfg.initial_capital
        ) * self.cfg.reward_scaling
        
        # Pénalité drawdown progressif
        drawdown = (self._peak_value - current_value) / (self._peak_value + 1e-8)
        drawdown_penalty = drawdown * 0.1  # Légère pénalité continue
        
        reward = (
            sharpe_reward   * self.cfg.reward_scaling
            - tx_penalty
            - drawdown_penalty
        )
        
        return float(np.clip(reward, -10.0, 10.0))  # Clipping pour PPO
    
    
    # ============================================================
    # OBSERVATION
    # ============================================================
    def _get_observation(self) -> np.ndarray:
        """
        Construit le vecteur d'observation :
        [features_window (flattened)] + [position, unrealized_pnl]
        
        window_size × n_features + 2 valeurs portfolio
        """
        # Fenêtre glissante des features
        start = self._current_step - self.cfg.window_size
        end   = self._current_step
        window = self.features[start:end].flatten()   # shape: (window×features,)
        
        # État du portfolio normalisé
        current_price   = self.prices[self._current_step]
        unrealized_pnl  = 0.0
        
        if self._position == 1 and self._entry_price > 0:
            unrealized_pnl = (
                (current_price - self._entry_price) / self._entry_price
            )
        
        portfolio_state = np.array([
            float(self._position),   # 0 ou 1
            unrealized_pnl           # PnL latent normalisé
        ], dtype=np.float32)
        
        obs = np.concatenate([window, portfolio_state])
        
        return obs.astype(np.float32)
    
    
    # ============================================================
    # CONDITIONS DE FIN
    # ============================================================
    def _is_terminated(self) -> bool:
        """
        Episode terminé si :
        1. Drawdown dépasse le seuil max
        2. Portfolio value quasi nulle (ruine)
        """
        # Drawdown max atteint
        drawdown = (
            (self._peak_value - self._portfolio_value)
            / (self._peak_value + 1e-8)
        )
        if drawdown > self.cfg.max_drawdown_pct:
            print(f"   ⚠️  Episode terminé : Drawdown {drawdown:.1%} > "
                  f"{self.cfg.max_drawdown_pct:.1%}")
            return True
        
        # Quasi ruine
        if self._portfolio_value < self.cfg.initial_capital * 0.05:
            print("   ⚠️  Episode terminé : Quasi-ruine")
            return True
        
        return False
    
    
    # ============================================================
    # HELPERS
    # ============================================================
    def _get_recent_volatility(self) -> float:
        """Volatilité des log-returns sur la fenêtre courante."""
        start = self._current_step - self.cfg.window_size
        end   = self._current_step
        # log_returns = index 0 dans FEATURES
        returns = self.features[start:end, 0]
        vol = float(np.std(returns))
        return max(vol, 1e-8)
    
    
    def _get_info(self) -> Dict:
        """Métriques de l'épisode courant."""
        current_price = self.prices[self._current_step]
        
        drawdown = (
            (self._peak_value - self._portfolio_value)
            / (self._peak_value + 1e-8)
        )
        total_return = (
            (self._portfolio_value - self.cfg.initial_capital)
            / self.cfg.initial_capital
        )
        
        return {
            "step"            : self._current_step,
            "portfolio_value" : round(self._portfolio_value, 2),
            "total_return"    : round(total_return, 4),
            "drawdown"        : round(drawdown, 4),
            "position"        : self._position,
            "current_price"   : round(current_price, 2),
            "n_trades"        : self.history['n_trades'],
        }
    
    
    def _reset_history(self):
        """Reset les historiques pour render."""
        self.history = {
            "portfolio_values" : [],
            "prices"           : [],
            "actions"          : [],
            "rewards"          : [],
            "n_trades"         : 0,
        }
    
    
    def _update_history(self, action: int, price: float, reward: float):
        self.history["portfolio_values"].append(self._portfolio_value)
        self.history["prices"].append(price)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        if action in [1, 2]:
            self.history["n_trades"] += 1
    
    
    # ============================================================
    # VALIDATION
    # ============================================================
    @staticmethod
    def _validate_data(data: pd.DataFrame):
        required = FEATURES + ['price']
        missing  = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")
        if data.isnull().any().any():
            raise ValueError("Le DataFrame contient des NaN. Applique dropna().")
    
    
    # ============================================================
    # RENDER
    # ============================================================
    def render(self):
        if self.render_mode == "human":
            self._render_human()
    
    
    def _render_human(self):
        """Visualisation complète de l'épisode."""
        
        if len(self.history["portfolio_values"]) < 2:
            return
        
        portfolio = np.array(self.history["portfolio_values"])
        prices    = np.array(self.history["prices"])
        actions   = np.array(self.history["actions"])
        rewards   = np.array(self.history["rewards"])
        
        # Buy & Hold benchmark
        bh_return = prices / prices[0] * self.cfg.initial_capital
        
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)
        
        # --- Subplot 1 : Portfolio vs Buy & Hold ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(portfolio,  label='Agent DRL',    color='#2ecc71', linewidth=2)
        ax1.plot(bh_return,  label='Buy & Hold',   color='#3498db',
                 linewidth=1.5, linestyle='--', alpha=0.8)
        ax1.axhline(
            self.cfg.initial_capital,
            color='gray', linestyle=':', linewidth=1
        )
        
        # Markers buy/sell
        buy_idx  = np.where(actions == 1)[0]
        sell_idx = np.where(actions == 2)[0]
        ax1.scatter(buy_idx,  portfolio[buy_idx],
                    marker='^', color='green', s=80,
                    zorder=5, label='Buy')
        ax1.scatter(sell_idx, portfolio[sell_idx],
                    marker='v', color='red',   s=80,
                    zorder=5, label='Sell')
        
        final_return = (portfolio[-1] - self.cfg.initial_capital) \
                       / self.cfg.initial_capital * 100
        bh_final     = (bh_return[-1]  - self.cfg.initial_capital) \
                       / self.cfg.initial_capital * 100
        
        ax1.set_title(
            f'Portfolio : {final_return:+.1f}% | '
            f'Buy & Hold : {bh_final:+.1f}% | '
            f'Trades : {self.history["n_trades"]}',
            fontsize=11
        )
        ax1.set_ylabel('Valeur ($)')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # --- Subplot 2 : Reward ---
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(
            range(len(rewards)), rewards,
            color=np.where(rewards > 0, '#27ae60', '#e74c3c'),
            alpha=0.7, width=1.0
        )
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Reward par Step')
        ax2.set_ylabel('Reward')
        ax2.grid(alpha=0.3)
        
        # --- Subplot 3 : Actions ---
        ax3 = fig.add_subplot(gs[2])
        action_colors = {0: '#95a5a6', 1: '#2ecc71', 2: '#e74c3c'}
        bar_colors    = [action_colors[a] for a in actions]
        ax3.bar(range(len(actions)), actions,
                color=bar_colors, alpha=0.8, width=1.0)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Hold', 'Buy', 'Sell'])
        ax3.set_title('Actions de l\'Agent')
        ax3.set_xlabel('Steps')
        ax3.grid(alpha=0.3, axis='x')
        
        plt.suptitle('DRL Trading Agent — Episode Summary', 
                     fontsize=13, fontweight='bold')
        plt.show()
        
        # Stats console
        self._print_episode_stats(portfolio, bh_return)
    
    
    def _print_episode_stats(self, portfolio: np.ndarray, bh: np.ndarray):
        """Affiche les métriques clés de l'épisode."""
        
        returns      = np.diff(portfolio) / portfolio[:-1]
        sharpe       = (
            np.mean(returns) / (np.std(returns) + 1e-8)
        ) * np.sqrt(252)
        
        peak         = np.maximum.accumulate(portfolio)
        drawdowns    = (peak - portfolio) / (peak + 1e-8)
        max_dd       = np.max(drawdowns)
        
        total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
        bh_return    = (bh[-1] - bh[0]) / bh[0]
        
        print("\n" + "="*45)
        print("  EPISODE STATS")
        print("="*45)
        print(f"  Portfolio Final  : ${portfolio[-1]:,.2f}")
        print(f"  Total Return     : {total_return:+.2%}")
        print(f"  Buy & Hold       : {bh_return:+.2%}")
        print(f"  Sharpe Ratio     : {sharpe:.3f}")
        print(f"  Max Drawdown     : {max_dd:.2%}")
        print(f"  Nb Trades        : {self.history['n_trades']}")
        print("="*45 + "\n")


# ============================================================
# TEST DE L'ENVIRONNEMENT
# ============================================================
if __name__ == "__main__":
    
    # Import du data loader (fichier précédent)
    from feature_engineering import load_data, DataConfig
    
    # 1. Chargement des données
    cfg_data = DataConfig(ticker="AAPL")
    train, val, test, scaler, gmm = load_data(cfg_data)
    
    # 2. Init environnement
    cfg_env = EnvConfig(
        initial_capital  = 10_000.0,
        transaction_cost = 0.001,
        window_size      = 10,
        max_drawdown_pct = 0.25,
    )
    
    env = TradingEnv(data=train, cfg=cfg_env, render_mode="human")
    
    # 3. Vérification Gymnasium (IMPORTANT avant SB3)
    from gymnasium.utils.env_checker import check_env
    print("🔍 Vérification de l'environnement...")
    check_env(env, warn=True)
    print("✅ Environnement valide !\n")
    
    # 4. Episode aléatoire pour tester
    print("🎲 Episode avec actions aléatoires...")
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    done         = False
    
    while not done:
        action           = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward    += reward
        done             = terminated or truncated
    
    print(f"Reward total : {total_reward:.2f}")
    env.render()