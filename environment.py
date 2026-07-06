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
    # ── Paramètres financiers ─────────────────────────────────────────
    # [GESTIONNAIRE] Capital initial de simulation.
    # N'affecte pas l'apprentissage (les returns sont normalisés),
    # mais change les valeurs absolues affichées dans les graphes.
    initial_capital:   float = 10_000.0

    # [GESTIONNAIRE] Coût de transaction par trade (aller simple).
    # 0.001 = 0.1% (courtier discount) | 0.002 = 0.2% (réaliste avec spread)
    # 0.005 = 0.5% (marché moins liquide)
    # Augmenter ce paramètre pénalise davantage l'overtrading.
    transaction_cost:  float = 0.002

    # ── Paramètres d'observation ──────────────────────────────────────
    # [GESTIONNAIRE] Nombre de jours historiques visibles par l'agent.
    # 5  = vue très court-terme (day trading)
    # 10 = par défaut — équilibre signal/bruit
    # 20 = vue moyen-terme, capte mieux les tendances mais augmente la taille des obs
    # ⚠️  Changer ce paramètre change la taille de l'observation space → réentraîner
    window_size:       int   = 10

    # ── Paramètres du reward ─────────────────────────────────────────
    # [TECHNIQUE] Facteur multiplicatif appliqué à l'alpha avant de le passer à PPO.
    # PPO fonctionne mieux avec des rewards dans [-5, 5].
    # Sur des returns journaliers (~±1%), ×100 donne un signal dans [-1, 1] avant clipping.
    # Ne pas toucher sauf si les rewards sont trop petits (<0.01) ou trop grands (>10).
    reward_scaling:    float = 100.0

    # ── Gestion du risque ────────────────────────────────────────────
    # [GESTIONNAIRE] Drawdown maximum toléré avant de terminer l'épisode.
    # 0.10 = 10% (très strict, force l'agent à couper rapidement ses pertes)
    # 0.25 = 25% (par défaut, réaliste pour un actif volatil)
    # 0.50 = 50% (permissif, laisse l'agent traverser des baisses profondes)
    # Un seuil plus bas force l'agent à apprendre à gérer le risque,
    # mais raccourcit les épisodes → moins d'expérience par episode.
    max_drawdown_pct:  float = 0.25

    # ── Actions ──────────────────────────────────────────────────────
    # 0 = Hold  → ne rien faire (reste dans la position courante)
    # 1 = Long  → aller long 100% (ou couvrir le short et aller long)
    # 2 = Flat  → fermer toute position (cash)
    # 3 = Short → vendre à découvert 100% (ou fermer le long et shorter)
    #
    # Le short permet de PROFITER des baisses de marché.
    # Sans short, l'agent ne peut que "éviter" les baisses (rester flat).
    # Avec short, l'agent peut gagner de l'argent pendant les baisses.
    #
    # [GESTIONNAIRE] Mettre n_actions = 3 pour désactiver le short (= ancien mode)
    # ⚠️  Changer ce paramètre nécessite de réentraîner le modèle
    n_actions:         int   = 4

    # ── Features observées ───────────────────────────────────────────
    # None = FEATURES de base (obs de taille 52). Tuple de noms pour une
    # liste custom, ex. base + régime → 72 (cf. data_loader.REGIME_FEATURES).
    # ⚠️  'log_returns' doit rester en position 0 (_get_recent_volatility).
    # ⚠️  Changer la liste change l'observation space → réentraîner.
    features:          Optional[Tuple[str, ...]] = None

    # ── Position continue (Acte 5) ───────────────────────────────────
    # [GESTIONNAIRE] True → action = poids cible w ∈ [-1, 1] (Box) au lieu
    # des 4 actions discrètes. Même comptabilité cash/shares (le short
    # fractionnaire est un nombre de titres négatif), frais sur le notionnel
    # échangé |Δshares|·P. L'observation garde sa taille : le slot position
    # passe de {-1, 0, 1} à [-1, 1].
    # ⚠️  Change l'action space → modèles discrets incompatibles.
    continuous:        bool  = False

    # [GESTIONNAIRE] λ ≥ 0 : aversion au risque dépendante du régime.
    # Ajoute -λ·σ̂·|w|·reward_scaling à la récompense (σ̂ = vol des rendements
    # sur la fenêtre d'observation) : porter de l'exposition coûte d'autant
    # plus que le marché est nerveux → incite à dérisquer AVANT le
    # kill-switch. 0 = désactivé (récompense historique inchangée).
    risk_aversion:     float = 0.0


FEATURES = [
    'log_returns',
    'volatility',
    'rsi',
    'macd_norm',
    'momentum_5',
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
        self.features_list = list(cfg.features) if cfg.features else list(FEATURES)

        # ---- Données ----
        self._validate_data(data, self.features_list)
        self.data        = data.reset_index(drop=True)
        self.prices      = self.data['price'].values.astype(np.float32)
        self.features    = self.data[self.features_list].values.astype(np.float32)
        self.n_steps     = len(self.data)

        # ---- Spaces ----
        # Observation : (window_size × n_features) + portfolio_state
        n_obs = cfg.window_size * len(self.features_list) + 2  # +2 : position + pnl
        
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (n_obs,),
            dtype = np.float32
        )
        
        if cfg.continuous:
            # Poids cible w ∈ [-1, 1] : -1 = short 100 %, 0 = flat, 1 = long 100 %
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(cfg.n_actions)
        
        # ---- Segments (multi-ticker) ----
        # Si la donnée contient une colonne segment_id, on divise les épisodes
        # par segment pour éviter de croiser les frontières entre tickers.
        if 'segment_id' in self.data.columns:
            self._segments = []
            for sid in sorted(self.data['segment_id'].unique()):
                idxs = self.data.index[self.data['segment_id'] == sid].tolist()
                seg_start = idxs[0] + cfg.window_size
                seg_end   = idxs[-1] - 1
                if seg_end - seg_start > 50:          # segment assez long
                    self._segments.append((seg_start, seg_end))
        else:
            self._segments = None

        # ---- Historique pour render ----
        self._reset_history()

        # ---- State interne ----
        self._current_step   = 0
        self._seg_end        = self.n_steps - 1  # fin d'épisode courante
        self._position       = 0        # -1 = short | 0 = flat | 1 = long
        self._entry_price    = 0.0
        self._portfolio_value = cfg.initial_capital
        self._peak_value     = cfg.initial_capital
        self._cash           = cfg.initial_capital
        self._shares         = 0.0     # positif si long, négatif si short

        print(f"✅ TradingEnv initialisé")
        print(f"   Steps disponibles : {self.n_steps - cfg.window_size}")
        print(f"   Observation shape : {self.observation_space.shape}")
        if cfg.continuous:
            print(f"   Actions           : poids continu w ∈ [-1, 1]"
                  + (f" | aversion risque λ={cfg.risk_aversion}"
                     if cfg.risk_aversion > 0 else ""))
        elif cfg.n_actions == 4:
            print(f"   Actions           : {{0: Hold, 1: Long, 2: Flat, 3: Short}}")
        else:
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

        # options={"random_start": False} → départ déterministe au début des
        # données, épisode sur TOUT le split. Utilisé par evaluate.py pour des
        # métriques reproductibles. Défaut True (entraînement : départs variés).
        random_start = True
        if options is not None:
            random_start = options.get("random_start", True)

        # Point de départ — par segment si multi-ticker
        if self._segments:
            if random_start:
                seg_idx = int(self.np_random.integers(0, len(self._segments)))
            else:
                seg_idx = 0
            seg_start, seg_end = self._segments[seg_idx]
            self._seg_end = seg_end
            if random_start:
                buffer = min(100, (seg_end - seg_start) // 2)
                self._current_step = int(self.np_random.integers(seg_start, seg_end - buffer))
            else:
                self._current_step = seg_start
        else:
            self._seg_end = self.n_steps - 1
            max_start = self.n_steps - self.cfg.window_size - 100
            if random_start and max_start > self.cfg.window_size:
                self._current_step = int(self.np_random.integers(self.cfg.window_size, max_start))
            else:
                # Données trop courtes pour un départ aléatoire → départ fixe
                self._current_step = self.cfg.window_size
        
        self._position        = 0
        self._cash            = self.cfg.initial_capital
        self._shares          = 0.0
        self._portfolio_value = self.cfg.initial_capital
        self._peak_value      = self.cfg.initial_capital
        self._entry_price     = 0.0

        self._reset_history()
        return self._get_observation(), self._get_info()
        
    
    # ============================================================
    # STEP
    # ============================================================
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:

        prev_price    = self.prices[self._current_step - 1]
        current_price = self.prices[self._current_step]
        prev_value    = self._portfolio_value

        # --- Exécution de l'action ---
        if self.cfg.continuous:
            # Le poids cible est clippé (PPO gaussien peut sortir de la boîte)
            w_target = float(np.clip(
                np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
            transaction_cost = self._rebalance_to(w_target, current_price)
            action = w_target                     # loggé en float dans history
        else:
            assert self.action_space.contains(action), f"Action invalide : {action}"
            transaction_cost = self._execute_action(action, current_price)

        # --- Mise à jour de la valeur du portfolio ---
        self._portfolio_value = self._cash + self._shares * current_price

        # --- Mise à jour du peak (pour drawdown) ---
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # --- Calcul du reward ---
        reward = self._compute_reward(
            prev_value       = prev_value,
            current_value    = self._portfolio_value,
            transaction_cost = transaction_cost,
            prev_price       = prev_price,
            current_price    = current_price,
        )
        
        # --- Logging ---
        self._update_history(action, current_price, reward)
        
        # --- Avance dans le temps ---
        self._current_step += 1
        
        # --- Conditions de fin d'épisode ---
        terminated = self._is_terminated()
        truncated  = self._current_step >= self._seg_end
        
        obs  = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    
    # ============================================================
    # EXÉCUTION DES ACTIONS
    # ============================================================
    def _execute_action(self, action: int, price: float) -> float:
        """
        Exécute l'action et retourne le coût de transaction total.

        ─── Actions (n_actions = 4) ────────────────────────────────────
        0 = HOLD  → ne rien faire, reste dans la position courante
        1 = LONG  → aller long 100%
                     • si flat  → achète des actions
                     • si short → couvre le short PUIS achète (2 trades)
                     • si long  → no-op (déjà là)
        2 = FLAT  → fermer toute position, aller en cash
                     • si long  → vend les actions
                     • si short → couvre le short (rachète les actions empruntées)
                     • si flat  → no-op
        3 = SHORT → vendre à découvert 100%
                     • si flat  → ouvre un short
                     • si long  → vend les actions PUIS ouvre un short (2 trades)
                     • si short → no-op (déjà là)

        ─── Mécanique du short ─────────────────────────────────────────
        Emprunter N actions et les vendre au prix P0 :
          _shares = -N  (négatif = dette de shares)
          _cash  += N × P0  (reçoit les proceeds de la vente à découvert)

        Portfolio value = _cash + _shares × P_current
                        = (initial + N×P0) + (-N) × P_current
                        = initial + N × (P0 - P_current)
        → Profit si P_current < P0 ✅ (prix a baissé)
        → Perte  si P_current > P0 ✅ (prix a monté)

        ─── Compatibilité mode 3 actions ───────────────────────────────
        Si n_actions = 3, actions 1=Buy, 2=Sell fonctionnent identiquement
        (action 3 ne sera jamais choisie par l'agent).
        """
        cost = 0.0

        # ── ACTION 0 : HOLD ─────────────────────────────────────────
        if action == 0:
            return 0.0

        # ── ACTION 1 : GO LONG ──────────────────────────────────────
        elif action == 1:
            if self._position == 1:
                return 0.0  # Déjà long, no-op

            # Étape 1 : couvrir le short si nécessaire
            if self._position == -1:
                cover_cost       = abs(self._shares) * price * self.cfg.transaction_cost
                # Racheter les shares empruntées (rembourse la dette)
                self._cash      += self._shares * price   # _shares < 0 → soustrait du cash
                self._cash      -= cover_cost
                self._shares     = 0.0
                self._position   = 0
                self._entry_price = 0.0
                cost            += cover_cost

            # Étape 2 : acheter (si on a du cash)
            if self._cash > 0 and self._position == 0:
                buy_cost         = self._cash * self.cfg.transaction_cost
                investable       = self._cash - buy_cost
                self._shares     = investable / price
                self._cash       = 0.0
                self._position   = 1
                self._entry_price = price
                cost            += buy_cost

        # ── ACTION 2 : GO FLAT (SELL ou COVER) ──────────────────────
        elif action == 2:
            if self._position == 0:
                return 0.0  # Déjà flat, no-op

            if self._position == 1:
                # Vendre les actions (long → flat)
                proceeds         = self._shares * price
                sell_cost        = proceeds * self.cfg.transaction_cost
                self._cash       = proceeds - sell_cost
                self._shares     = 0.0
                self._position   = 0
                self._entry_price = 0.0
                cost            += sell_cost

            elif self._position == -1:
                # Couvrir le short (racheter les shares empruntées)
                cover_cost       = abs(self._shares) * price * self.cfg.transaction_cost
                self._cash      += self._shares * price   # _shares < 0
                self._cash      -= cover_cost
                self._shares     = 0.0
                self._position   = 0
                self._entry_price = 0.0
                cost            += cover_cost

        # ── ACTION 3 : GO SHORT ─────────────────────────────────────
        elif action == 3:
            if self._position == -1:
                return 0.0  # Déjà short, no-op

            # Étape 1 : fermer le long si nécessaire
            if self._position == 1:
                proceeds         = self._shares * price
                sell_cost        = proceeds * self.cfg.transaction_cost
                self._cash       = proceeds - sell_cost
                self._shares     = 0.0
                self._position   = 0
                self._entry_price = 0.0
                cost            += sell_cost

            # Étape 2 : ouvrir le short
            if self._cash > 0 and self._position == 0:
                short_cost       = self._cash * self.cfg.transaction_cost
                investable       = self._cash - short_cost
                n_shares         = investable / price
                self._shares     = -n_shares               # négatif = short
                self._cash      -= short_cost              # frais déduits du cash (symétrie avec le long)
                self._cash      += n_shares * price        # reçoit les proceeds
                self._position   = -1
                self._entry_price = price
                cost            += short_cost

        return cost


    # ============================================================
    # REBALANCEMENT CONTINU (Acte 5)
    # ============================================================
    def _rebalance_to(self, w_target: float, price: float) -> float:
        """
        Amène le portefeuille au poids cible w ∈ [-1, 1].

        Même comptabilité que le discret : V = cash + shares × P, shares < 0
        pour un short fractionnaire. On échange Δ = w·V/P − shares titres,
        frais = |Δ|·P·tc débités du cash. Le poids réalisé après frais dévie
        du poids cible d'un epsilon (le notionnel cible est calculé avant
        frais) — approximation documentée, négligeable à tc = 0.1 %.
        """
        prev_w = float(self._position)
        self._delta_w = abs(w_target - prev_w)

        # Bande de non-rebalancement (0.5 % de poids) : sans elle, le jitter
        # gaussien de PPO et la dérive du poids induite par les frais eux-mêmes
        # déclencheraient un micro-trade payant à CHAQUE pas (hémorragie de
        # frais). En-dessous du seuil : on ne touche à rien.
        if self._delta_w < 0.005:
            self._delta_w = 0.0
            return 0.0

        value = self._cash + self._shares * price
        target_shares = w_target * value / price
        delta = target_shares - self._shares
        trade_notional = abs(delta) * price

        if trade_notional < 1e-12:
            return 0.0

        cost = trade_notional * self.cfg.transaction_cost
        self._cash  -= delta * price + cost
        self._shares = target_shares

        # entry_price au changement de signe : réfère le PnL latent de l'obs
        if np.sign(w_target) != np.sign(prev_w):
            self._entry_price = price if abs(w_target) > 1e-9 else 0.0
        self._position = w_target
        return cost


    # ============================================================
    # REWARD FUNCTION
    # ============================================================
    def _compute_reward(
        self,
        prev_value:       float,
        current_value:    float,
        transaction_cost: float,
        prev_price:       float,
        current_price:    float,
    ) -> float:
        """
        Reward = Alpha vs Buy & Hold par step

        Formule :
            portfolio_return_t = (V_t - V_{t-1}) / V_{t-1}
            bh_return_t        = (P_t - P_{t-1}) / P_{t-1}
            alpha_t            = portfolio_return_t - bh_return_t

        Pourquoi ce reward ?
        - Signal direct pour BATTRE le marché, pas juste monter avec lui
        - Être FLAT quand le marché baisse → alpha > 0 (récompense)
        - Être FLAT quand le marché monte → alpha < 0 (pénalité)
        - Être LONG track le marché      → alpha ≈ 0 (neutre)
        - L'agent apprend à TIMER le marché, pas à juste hold
        - Élimine le biais "toujours Buy" de la reward Sharpe-like précédente
        """
        # Rendement du portefeuille sur ce step
        portfolio_return = (current_value - prev_value) / (prev_value + 1e-8)

        # Rendement du marché sur ce step (référence Buy & Hold)
        bh_step_return = (current_price - prev_price) / (prev_price + 1e-8)

        # Alpha : l'agent fait-il mieux que le marché à ce step ?
        alpha = portfolio_return - bh_step_return

        # Pénalité de transaction (décourage l'overtrading)
        tx_penalty = (
            transaction_cost / self.cfg.initial_capital
        ) * self.cfg.reward_scaling

        # Pénalité drawdown légère (gestion du risque)
        drawdown = (self._peak_value - current_value) / (self._peak_value + 1e-8)
        drawdown_penalty = drawdown * 0.05

        # Aversion au risque dépendante du régime (Acte 5, bras B) :
        # porter |w| coûte proportionnellement à la nervosité du marché →
        # paie le dérisquage AVANT que le kill-switch ne s'en charge.
        risk_penalty = 0.0
        if self.cfg.risk_aversion > 0.0:
            risk_penalty = (
                self.cfg.risk_aversion
                * self._get_recent_volatility()
                * abs(float(self._position))
                * self.cfg.reward_scaling
            )

        reward = (
            alpha * self.cfg.reward_scaling
            - tx_penalty
            - drawdown_penalty
            - risk_penalty
        )

        return float(np.clip(reward, -10.0, 10.0))
    
    
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
        current_price  = self.prices[self._current_step]
        unrealized_pnl = 0.0

        # Formule signée unifiée : sign(w)·(P/P_entry − 1) — identique à
        # l'ancien if/elif pour w ∈ {-1, 1}, et valide pour w fractionnaire.
        if self._entry_price > 0 and abs(float(self._position)) > 1e-9:
            unrealized_pnl = (
                np.sign(float(self._position))
                * (current_price - self._entry_price) / self._entry_price
            )

        portfolio_state = np.array([
            float(self._position),   # -1 (short) | 0 (flat) | 1 (long)
            unrealized_pnl           # PnL latent normalisé (positif = en gain)
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
            "turnover"         : 0.0,   # Σ|Δw| (mode continu)
        }
        self._delta_w = 0.0


    def _update_history(self, action, price: float, reward: float):
        self.history["portfolio_values"].append(self._portfolio_value)
        self.history["prices"].append(price)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        if self.cfg.continuous:
            self.history["turnover"] += self._delta_w
            if self._delta_w > 0.01:          # rebalancement significatif
                self.history["n_trades"] += 1
        elif action in [1, 2, 3]:
            self.history["n_trades"] += 1
    
    
    # ============================================================
    # VALIDATION
    # ============================================================
    @staticmethod
    def _validate_data(data: pd.DataFrame, features_list=None):
        required = (features_list if features_list is not None else FEATURES) + ['price']
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
        
        # Markers long/flat/short
        long_idx  = np.where(actions == 1)[0]
        flat_idx  = np.where(actions == 2)[0]
        short_idx = np.where(actions == 3)[0]
        ax1.scatter(long_idx,  portfolio[long_idx],
                    marker='^', color='green',  s=80,
                    zorder=5, label='Go Long')
        ax1.scatter(flat_idx,  portfolio[flat_idx],
                    marker='s', color='gray',   s=60,
                    zorder=5, label='Go Flat')
        ax1.scatter(short_idx, portfolio[short_idx],
                    marker='v', color='orange', s=80,
                    zorder=5, label='Go Short')
        
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
        action_colors = {0: '#95a5a6', 1: '#2ecc71', 2: '#3498db', 3: '#e74c3c'}
        bar_colors    = [action_colors[a] for a in actions]
        ax3.bar(range(len(actions)), actions,
                color=bar_colors, alpha=0.8, width=1.0)
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['Hold', 'Long', 'Flat', 'Short'])
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

        peak         = np.maximum.accumulate(portfolio)
        drawdowns    = (peak - portfolio) / (peak + 1e-8)
        max_dd       = np.max(drawdowns)

        total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
        bh_return    = (bh[-1] - bh[0]) / bh[0]
        alpha        = total_return - bh_return

        print("\n" + "="*45)
        print("  EPISODE STATS")
        print("="*45)
        print(f"  Portfolio Final  : ${portfolio[-1]:,.2f}")
        print(f"  Agent Return     : {total_return:+.2%}")
        print(f"  Buy & Hold       : {bh_return:+.2%}")
        print(f"  Alpha            : {alpha:+.2%}  {'✅' if alpha > 0 else '❌'}")
        print(f"  Max Drawdown     : {max_dd:.2%}")
        print(f"  Nb Trades        : {self.history['n_trades']}")
        print("="*45 + "\n")


# ============================================================
# TEST DE L'ENVIRONNEMENT
# ============================================================
if __name__ == "__main__":
    
    # Import du data loader (fichier précédent)
    from data_loader import load_data, DataConfig
    
    # 1. Chargement des données
    cfg_data = DataConfig(ticker="AAPL")
    train, val, test, scaler = load_data(cfg_data)
    
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