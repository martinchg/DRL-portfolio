# 📅 Plan d'Action Hebdomadaire — MARL + Diffusion

## Calendrier Réaliste (10 semaines)

---

## 🗓️ SEMAINE 1 : Setup Multi-Agent Infrastructure

### Lundi : Installation & Setup
```bash
# Créer une nouvelle branche
git checkout -b feature/marl-diffusion

# Installer les dépendances
pip install pettingzoo==1.24.3
pip install "ray[rllib]==2.9.0"
pip install supersuit
pip install gymnasium==0.29.1

# Créer la structure
mkdir -p envs/{tests,configs}
mkdir -p models/marl
mkdir -p scripts/training
```

**Commit goal** : "feat: setup MARL infrastructure"

---

### Mardi-Mercredi : Environnement Multi-Agent v1

**Fichier à créer** : `envs/multi_agent_trading_env.py`

Specs :
```python
from pettingzoo import ParallelEnv

class MultiAgentTradingEnv(ParallelEnv):
    metadata = {'name': 'trading-v0'}
    
    def __init__(self, data, n_agents=2):
        self.agents = ['portfolio_manager', 'market_adversary']
        
        # Agent 1 : Portfolio Manager
        self.action_spaces = {
            'portfolio_manager': Discrete(3),  # Buy/Sell/Hold
        }
        
        # Agent 2 : Market Adversary
        self.action_spaces = {
            'market_adversary': Discrete(5),  # Différents types de stress
        }
        
    def reset(self, seed=None):
        # Returns: {agent: observation}
        
    def step(self, actions):
        # Returns: observations, rewards, dones, truncated, infos
```

**Tests à écrire** :
```python
# envs/tests/test_multi_agent_env.py
def test_env_initialization()
def test_reset_returns_correct_format()
def test_step_updates_state()
def test_rewards_are_adversarial()  # Sum close to 0
```

**Commit goal** : "feat: implement multi-agent trading environment"

---

### Jeudi : Agent Adversary Logic

**Fichier** : `envs/market_adversary.py`

L'adversaire peut :
1. **Liquidity Shock** : Augmente les transaction costs temporairement
2. **Volatility Spike** : Multiplie la volatilité par 2-3x
3. **Regime Change** : Force un changement de régime GMM
4. **Correlation Break** : Modifie les corrélations entre assets
5. **Do Nothing** : Laisse le marché tranquille

```python
class MarketAdversary:
    def apply_action(self, action, market_state):
        if action == 0:  # Liquidity shock
            market_state['transaction_cost'] *= 3
        elif action == 1:  # Vol spike
            market_state['volatility'] *= 2.5
        # ...
        return market_state
```

**Commit goal** : "feat: add market adversary mechanics"

---

### Vendredi : Validation & Documentation

- [ ] Tester l'env avec agents aléatoires
- [ ] Vérifier que les rewards sont bien adversariales
- [ ] Créer un notebook de démo : `notebooks/02_marl_env_demo.ipynb`
- [ ] Écrire le README de `envs/`

**Commit goal** : "docs: add MARL environment documentation"

---

## 🗓️ SEMAINE 2 : Training MAPPO

### Lundi : Configuration Ray RLlib

**Fichier** : `configs/mappo_config.yaml`

```yaml
env: MultiAgentTradingEnv
env_config:
  ticker: AAPL
  initial_capital: 10000
  window_size: 10

train_batch_size: 4096
sgd_minibatch_size: 128
num_sgd_iter: 10

model:
  fcnet_hiddens: [256, 256]
  use_lstm: false
  
lr: 0.0003
lr_schedule:
  - [0, 0.0003]
  - [500000, 0.00001]

gamma: 0.99
lambda: 0.95
clip_param: 0.2
entropy_coef: 0.01
vf_loss_coef: 0.5

# MAPPO specific
use_critic: true
use_gae: true

# Multi-agent
policies:
  portfolio_manager:
    policy_class: PPOTorchPolicy
  market_adversary:
    policy_class: PPOTorchPolicy

policy_mapping_fn: default  # Each agent uses its own policy
```

---

### Mardi-Mercredi : Script d'entraînement MAPPO

**Fichier** : `scripts/training/train_mappo.py`

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

def train_mappo():
    ray.init()
    
    config = (
        PPOConfig()
        .environment(
            env=MultiAgentTradingEnv,
            env_config={'data': train_data}
        )
        .framework("torch")
        .training(
            train_batch_size=4096,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coef=0.01,
        )
        .multi_agent(
            policies={
                "portfolio": (None, obs_space, act_space, {}),
                "adversary": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, **kwargs: (
                "portfolio" if agent_id == "portfolio_manager" else "adversary"
            ),
        )
        .rollouts(num_rollout_workers=4)
        .callbacks(FinancialMetricsCallback)
    )
    
    algo = config.build()
    
    for i in range(100):
        result = algo.train()
        print(f"Iter {i}: reward={result['episode_reward_mean']:.2f}")
        
        if i % 10 == 0:
            algo.save(f"checkpoints/mappo_iter_{i}")
    
    ray.shutdown()
```

---

### Jeudi : Callbacks pour Métriques Financières

**Fichier** : `models/marl/callbacks.py`

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class FinancialMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Récupérer l'historique de l'épisode
        portfolio_values = episode.user_data.get('portfolio_values', [])
        
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            episode.custom_metrics['sharpe_ratio'] = sharpe
            episode.custom_metrics['total_return'] = (
                (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            )
            
            # Max Drawdown
            peak = np.maximum.accumulate(portfolio_values)
            dd = (peak - portfolio_values) / peak
            episode.custom_metrics['max_drawdown'] = np.max(dd)
```

---

### Vendredi : Premier Run & Debugging

- [ ] Lancer le training sur 50k steps
- [ ] Vérifier les logs TensorBoard
- [ ] Comparer avec le PPO simple (baseline)

**Objectif** : Avoir un modèle MAPPO fonctionnel, même si pas optimal.

**Commit goal** : "feat: working MAPPO training pipeline"

---

## 🗓️ SEMAINE 3 : Optimisation MARL

### Lundi-Mardi : Reward Shaping

Tester différentes reward functions :

```python
# Version 1 : Simple adversarial
reward_pm = sharpe_ratio
reward_adv = -sharpe_ratio

# Version 2 : Balanced
reward_pm = sharpe_ratio - 0.1 * transaction_costs
reward_adv = -sharpe_ratio + 0.1 * market_stability_bonus

# Version 3 : Risk-adjusted
reward_pm = sharpe_ratio - 0.2 * max_drawdown
reward_adv = -sharpe_ratio + 0.1 * diversity_bonus
```

Faire une ablation study : quelle reward fonctionne le mieux ?

---

### Mercredi : Hyperparameter Tuning

Utiliser Ray Tune :

```python
from ray import tune

config = {
    "lr": tune.grid_search([1e-4, 3e-4, 1e-3]),
    "entropy_coef": tune.grid_search([0.0, 0.01, 0.05]),
    "clip_param": tune.grid_search([0.1, 0.2, 0.3]),
}

analysis = tune.run(
    "PPO",
    config=config,
    stop={"timesteps_total": 500000},
    num_samples=3,
)

best_config = analysis.get_best_config(metric="episode_reward_mean")
```

---

### Jeudi-Vendredi : Alternative MADDPG

Pour des actions continues :

```python
# envs/continuous_trading_env.py
action_space = Box(
    low=np.array([-1.0] * n_assets),
    high=np.array([1.0] * n_assets),
)
# -1 = short max, 0 = neutral, +1 = long max
```

Entraîner MADDPG et comparer avec MAPPO.

**Commit goal** : "feat: add MADDPG alternative implementation"

---

## 🗓️ SEMAINE 4 : Setup Diffusion Models

### Lundi : Théorie & Setup

- [ ] Lire le paper TimeGrad
- [ ] Comprendre le forward/reverse process
- [ ] Installer Hugging Face Diffusers

```bash
pip install diffusers==0.25.0
pip install pytorch-lightning==2.1.0
pip install einops
```

---

### Mardi-Mercredi : DDPM pour Séries Temporelles

**Fichier** : `diffusion/ddpm_time_series.py`

```python
import torch
import torch.nn as nn
from diffusers import DDPMScheduler

class UNet1D(nn.Module):
    """U-Net pour séries temporelles 1D"""
    def __init__(self, in_channels=5, out_channels=5, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder (downsampling)
        self.enc1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 256, 3, padding=1),
        )
        
        # Decoder (upsampling)
        self.dec3 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.Conv1d(64, out_channels, 3, padding=1)
        
    def forward(self, x, t):
        # x: [batch, channels, seq_len]
        # t: [batch] timestep
        
        t_emb = self.time_mlp(t.unsqueeze(-1))  # [batch, time_emb_dim]
        
        # Encoder
        e1 = F.gelu(self.enc1(x))
        e2 = F.gelu(self.enc2(e1))
        e3 = F.gelu(self.enc3(e2))
        
        # Bottleneck (inject time embedding ici si besoin)
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = F.gelu(self.dec3(b))
        d2 = F.gelu(self.dec2(d3 + e2))  # Skip connection
        out = self.dec1(d2 + e1)  # Skip connection
        
        return out


class TimeSeriesDDPM(nn.Module):
    def __init__(self, num_train_timesteps=1000):
        super().__init__()
        
        self.unet = UNet1D(in_channels=5, out_channels=5)
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
        )
        
    def forward(self, x0, noise=None):
        """
        Training forward pass
        x0: clean data [batch, channels, seq_len]
        """
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=x0.device
        ).long()
        
        # Add noise
        if noise is None:
            noise = torch.randn_like(x0)
        
        noisy_x = self.noise_scheduler.add_noise(x0, noise, t)
        
        # Predict noise
        noise_pred = self.unet(noisy_x, t)
        
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(self, batch_size=1, seq_len=252, device='cpu'):
        """Generate samples via reverse diffusion"""
        
        # Start from pure noise
        shape = (batch_size, 5, seq_len)  # 5 features
        x = torch.randn(shape, device=device)
        
        # Reverse process
        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x, t_batch)
            
            # Remove noise (one step)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        return x
```

---

### Jeudi : Data Preprocessing pour Diffusion

**Fichier** : `diffusion/data_preprocessing.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=252, features=['log_returns', 'volatility', 'rsi', 'dist_to_sma', 'market_regime']):
        """
        data: DataFrame avec les features
        seq_len: Longueur des séquences (ex: 252 jours = 1 an)
        """
        self.data = data[features].values
        self.seq_len = seq_len
        
        # Normalisation (important pour la diffusion)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Retourne une fenêtre de seq_len
        x = self.data[idx:idx + self.seq_len]
        
        # Format: [seq_len, features] → [features, seq_len]
        x = torch.FloatTensor(x).transpose(0, 1)
        
        return x
```

---

### Vendredi : Premier entraînement Diffusion (test)

```python
# scripts/training/train_diffusion.py

from torch.utils.data import DataLoader

# Load data
train_data, _, _, _, _ = load_data(DataConfig(ticker="AAPL"))
dataset = TimeSeriesDataset(train_data, seq_len=252)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = TimeSeriesDDPM(num_train_timesteps=1000).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in loader:
        batch = batch.to(device)
        
        noise_pred, noise = model(batch)
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
    
    # Sample quelques trajectoires pour visualiser
    if epoch % 10 == 0:
        samples = model.sample(batch_size=5, seq_len=252, device=device)
        plot_samples(samples)

torch.save(model.state_dict(), 'checkpoints/diffusion_model.pth')
```

**Commit goal** : "feat: working diffusion model for time series"

---

## 🗓️ SEMAINE 5 : Entraînement & Validation Diffusion

### Lundi-Mardi : Full Training Run

- [ ] Entraîner sur 100+ epochs
- [ ] Sauvegarder des checkpoints régulièrement
- [ ] Logger avec Weights & Biases ou TensorBoard

---

### Mercredi : Validation des Samples

**Fichier** : `diffusion/validation.py`

```python
def validate_diffusion_samples(real_data, synthetic_data):
    """
    Compare les distributions des données réelles vs synthétiques
    """
    metrics = {}
    
    # 1. KL Divergence
    from scipy.stats import entropy
    real_hist, _ = np.histogram(real_data, bins=50, density=True)
    synth_hist, _ = np.histogram(synthetic_data, bins=50, density=True)
    metrics['kl_divergence'] = entropy(real_hist + 1e-10, synth_hist + 1e-10)
    
    # 2. Autocorrelation
    from statsmodels.tsa.stattools import acf
    real_acf = acf(real_data, nlags=20)
    synth_acf = acf(synthetic_data, nlags=20)
    metrics['acf_distance'] = np.mean((real_acf - synth_acf)**2)
    
    # 3. Volatility Clustering (GARCH effect)
    real_vol = pd.Series(real_data).rolling(20).std()
    synth_vol = pd.Series(synthetic_data).rolling(20).std()
    metrics['vol_correlation'] = np.corrcoef(
        real_vol[20:], synth_vol[20:]
    )[0, 1]
    
    # 4. Moments
    metrics['mean_diff'] = abs(np.mean(real_data) - np.mean(synthetic_data))
    metrics['std_diff'] = abs(np.std(real_data) - np.std(synthetic_data))
    metrics['skew_diff'] = abs(
        pd.Series(real_data).skew() - pd.Series(synthetic_data).skew()
    )
    metrics['kurt_diff'] = abs(
        pd.Series(real_data).kurtosis() - pd.Series(synthetic_data).kurtosis()
    )
    
    return metrics
```

Faire un notebook avec des visualisations comparatives.

---

### Jeudi : Génération de 10k Trajectoires

```python
# scripts/generate_synthetic_data.py

model = TimeSeriesDDPM.load('checkpoints/diffusion_model.pth')
model.eval()

all_trajectories = []

for i in tqdm(range(10000)):
    traj = model.sample(batch_size=1, seq_len=252, device='cuda')
    all_trajectories.append(traj.cpu().numpy())

all_trajectories = np.concatenate(all_trajectories, axis=0)

# Sauvegarder
np.savez_compressed(
    'data/synthetic_trajectories.npz',
    trajectories=all_trajectories,
    metadata={'model': 'DDPM', 'n_samples': 10000, 'seq_len': 252}
)
```

---

### Vendredi : Rapport de Validation

Créer un rapport PDF avec :
- Distribution plots (real vs synthetic)
- ACF comparison
- Statistical tests (KS test, etc.)
- Visual inspection de 100 trajectoires

**Commit goal** : "feat: generate and validate 10k synthetic trajectories"

---

## 🗓️ SEMAINE 6 : Intégration Diffusion → MARL

### Lundi : Data Loader Hybride

**Fichier** : `data/hybrid_data_loader.py`

```python
class HybridDataLoader:
    def __init__(self, real_data, synthetic_data, mix_ratio=0.5):
        """
        mix_ratio: 0.5 = 50% real, 50% synthetic
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.mix_ratio = mix_ratio
    
    def sample_episode(self):
        if np.random.rand() < self.mix_ratio:
            # Sample from real data
            idx = np.random.randint(0, len(self.real_data) - 252)
            return self.real_data.iloc[idx:idx+252]
        else:
            # Sample from synthetic
            idx = np.random.randint(0, len(self.synthetic_data))
            return self.synthetic_data[idx]
```

---

### Mardi-Mercredi : Curriculum Learning

```python
# scripts/training/train_marl_with_diffusion.py

curriculum_schedule = [
    (0, 50000, 0.0),      # Phase 1: 100% real
    (50000, 150000, 0.3),  # Phase 2: 30% synthetic
    (150000, 300000, 0.5), # Phase 3: 50/50
    (300000, 500000, 0.7), # Phase 4: 70% synthetic
]

for phase_start, phase_end, synth_ratio in curriculum_schedule:
    loader = HybridDataLoader(real, synthetic, mix_ratio=synth_ratio)
    
    # Update env config
    config = config.environment(env_config={'data_loader': loader})
    
    # Train for (phase_end - phase_start) steps
    algo.train(num_steps=phase_end - phase_start)
```

---

### Jeudi-Vendredi : Comparaison des 3 Versions

Entraîner en parallèle :
1. **Baseline** : Real data only
2. **Augmented** : 50/50
3. **Full Synthetic** : 100% diffusion

Évaluer sur le test set (2024, données réelles out-of-sample).

**Métriques** :
- Sharpe Ratio
- Max Drawdown
- CVaR (95%)
- Sortino Ratio
- Calmar Ratio

**Commit goal** : "feat: complete MARL + Diffusion integration"

---

## 🗓️ SEMAINE 7 : Features Avancées

### Lundi-Mardi : Microstructure

**Fichier** : `features/microstructure.py`

```python
def compute_kyle_lambda(data, window=20):
    """Price impact coefficient"""
    returns = data['close'].pct_change()
    signed_volume = data['volume'] * np.sign(returns)
    
    kyle_lambda = []
    for i in range(window, len(data)):
        X = signed_volume[i-window:i].values.reshape(-1, 1)
        y = returns[i-window:i].values
        
        model = LinearRegression().fit(X, y)
        kyle_lambda.append(model.coef_[0])
    
    return pd.Series(kyle_lambda, index=data.index[window:])

def compute_amihud_illiquidity(data):
    """Amihud (2002) illiquidity measure"""
    returns = data['close'].pct_change().abs()
    dollar_volume = data['volume'] * data['close']
    
    return returns / (dollar_volume + 1e-10)

def compute_roll_spread(data):
    """Roll (1984) bid-ask spread estimator"""
    price_changes = data['close'].diff()
    covariance = price_changes.rolling(20).cov(price_changes.shift(1))
    
    spread = 2 * np.sqrt(-covariance.clip(upper=0))
    return spread
```

---

### Mercredi : Sentiment Analysis (bonus)

Si tu as accès à des news :

```python
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def get_sentiment_score(news_text):
    result = sentiment_analyzer(news_text)[0]
    
    # Convert to score: -1 (negative) to +1 (positive)
    if result['label'] == 'positive':
        return result['score']
    elif result['label'] == 'negative':
        return -result['score']
    else:
        return 0.0

# Example
news = "Tesla stock surges 15% on record deliveries"
score = get_sentiment_score(news)  # → +0.95
```

---

### Jeudi-Vendredi : Feature Engineering Pipeline

Intégrer toutes les nouvelles features :

```python
# features/pipeline.py

class FeaturePipeline:
    def __init__(self):
        self.features = [
            'log_returns',
            'volatility',
            'rsi',
            'dist_to_sma',
            'market_regime',
            'kyle_lambda',         # NEW
            'amihud_illiquidity',  # NEW
            'roll_spread',         # NEW
            'sentiment_score',     # NEW (optionnel)
        ]
    
    def transform(self, data):
        df = data.copy()
        
        # Existing features
        df = compute_base_features(df)
        
        # Microstructure
        df['kyle_lambda'] = compute_kyle_lambda(df)
        df['amihud_illiquidity'] = compute_amihud_illiquidity(df)
        df['roll_spread'] = compute_roll_spread(df)
        
        # Normalisation
        scaler = RobustScaler()
        df[self.features] = scaler.fit_transform(df[self.features])
        
        return df
```

**Commit goal** : "feat: add microstructure and sentiment features"

---

## 🗓️ SEMAINE 8 : Backtesting & Risk Analysis

### Lundi-Mardi : Framework de Backtesting

**Fichier** : `backtest/backtester.py`

```python
class Backtester:
    def __init__(self, agent, data, initial_capital=10000):
        self.agent = agent
        self.data = data
        self.initial_capital = initial_capital
        
    def run_walk_forward(self, train_window=252, test_window=63):
        """
        Walk-forward analysis : retrain tous les 3 mois
        """
        results = []
        
        for i in range(0, len(self.data) - train_window - test_window, test_window):
            # Train
            train_data = self.data.iloc[i:i+train_window]
            self.agent.retrain(train_data)
            
            # Test
            test_data = self.data.iloc[i+train_window:i+train_window+test_window]
            metrics = self.evaluate(test_data)
            
            results.append({
                'period': i,
                'sharpe': metrics['sharpe'],
                'return': metrics['total_return'],
                'mdd': metrics['max_drawdown'],
            })
        
        return pd.DataFrame(results)
    
    def evaluate(self, data):
        env = TradingEnv(data=data)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = self.agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        return self.compute_metrics(env.history)
    
    def compute_metrics(self, history):
        portfolio = np.array(history['portfolio_values'])
        
        returns = np.diff(portfolio) / portfolio[:-1]
        
        # Sharpe
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Sortino (downside deviation only)
        downside_returns = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside_returns) + 1e-8) * np.sqrt(252)
        
        # Max Drawdown
        peak = np.maximum.accumulate(portfolio)
        dd = (peak - portfolio) / peak
        max_dd = np.max(dd)
        
        # Calmar Ratio
        total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
        calmar = total_return / (max_dd + 1e-8)
        
        # CVaR (Conditional Value at Risk)
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)])
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'cvar_95': cvar_95,
        }
```

---

### Mercredi : Stress Testing

```python
# backtest/stress_testing.py

class StressTester:
    def __init__(self, agent):
        self.agent = agent
        
    def test_crisis_periods(self):
        """Test sur les crises historiques"""
        
        crises = {
            'COVID-19': ('2020-02-15', '2020-04-15'),
            'Financial Crisis': ('2008-09-01', '2009-03-01'),
            'Dot-com Bubble': ('2000-03-01', '2002-10-01'),
        }
        
        results = {}
        
        for name, (start, end) in crises.items():
            data = download_data(ticker='SPY', start=start, end=end)
            metrics = backtester.evaluate(data)
            
            results[name] = {
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_drawdown'],
                'cvar': metrics['cvar_95'],
            }
        
        return pd.DataFrame(results).T
    
    def monte_carlo_simulation(self, n_simulations=1000):
        """
        Simule 1000 scenarios futurs via le modèle de diffusion
        """
        all_returns = []
        
        for _ in range(n_simulations):
            # Générer un scenario
            scenario = diffusion_model.sample()
            
            # Évaluer l'agent dessus
            metrics = backtester.evaluate(scenario)
            all_returns.append(metrics['total_return'])
        
        # Statistiques
        return {
            'mean_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'var_95': np.percentile(all_returns, 5),
            'cvar_95': np.mean([r for r in all_returns if r <= np.percentile(all_returns, 5)]),
            'prob_positive': np.mean(np.array(all_returns) > 0),
        }
```

---

### Jeudi-Vendredi : Rapport Automatisé

**Fichier** : `backtest/report_generator.py`

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(results, output_path='reports/backtest_report.pdf'):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("DRL Portfolio — Backtest Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary Table
    summary_data = [
        ['Metric', 'Value'],
        ['Sharpe Ratio', f"{results['sharpe']:.2f}"],
        ['Sortino Ratio', f"{results['sortino']:.2f}"],
        ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
        ['CVaR (95%)', f"{results['cvar_95']:.2%}"],
        ['Total Return', f"{results['total_return']:.2%}"],
    ]
    
    table = Table(summary_data)
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Equity Curve (save as PNG first)
    plt.figure(figsize=(10, 6))
    plt.plot(results['portfolio_values'])
    plt.title('Portfolio Value Over Time')
    plt.savefig('temp_equity_curve.png')
    plt.close()
    
    img = Image('temp_equity_curve.png', width=400, height=240)
    story.append(img)
    
    doc.build(story)
```

**Commit goal** : "feat: complete backtesting framework with stress testing"

---

## 🗓️ SEMAINE 9 : Code Quality & Documentation

### Lundi-Mardi : Refactoring

- [ ] Appliquer Black formatter
- [ ] Type hints partout (mypy)
- [ ] Docstrings (Google style)
- [ ] Supprimer le code mort

```bash
# Install tools
pip install black isort mypy pytest-cov

# Format
black .
isort .

# Type check
mypy envs/ models/ diffusion/

# Tests
pytest --cov=. --cov-report=html
```

---

### Mercredi : Tests Unitaires

Atteindre 70%+ de coverage :

```python
# tests/test_marl_env.py
# tests/test_diffusion_model.py
# tests/test_backtester.py
# tests/test_features.py
```

---

### Jeudi-Vendredi : Documentation Sphinx

```bash
cd docs/
sphinx-quickstart
sphinx-apidoc -o . ../envs ../models ../diffusion
make html
```

Déployer sur GitHub Pages :
```bash
git checkout gh-pages
cp -r docs/_build/html/* .
git add .
git commit -m "docs: update documentation"
git push
```

**Commit goal** : "docs: complete Sphinx documentation"

---

## 🗓️ SEMAINE 10 : Présentation & Polish

### Lundi-Mardi : README Killer

Sections :
1. 🎯 **Problem Statement**
2. 💡 **Solution (MARL + Diffusion)**
3. 📊 **Results** (table avec comparaisons)
4. 🏗️ **Architecture** (diagram)
5. 🚀 **Quick Start** (3 commandes pour run)
6. 📚 **Documentation** (lien vers Sphinx)
7. 🎥 **Demo** (vidéo YouTube)
8. 📄 **Citation** (BibTeX si tu publies)

---

### Mercredi : Vidéo Démo

Scénario (2-3 min) :
1. Introduction (10s) : "Multi-Agent RL + Diffusion for Trading"
2. Problème (20s) : Overfitting, market dynamics
3. Solution (30s) : Adversarial training, synthetic data
4. Demo (60s) : Show dashboard, performance graphs
5. Results (30s) : Table de comparaison
6. Call to action (10s) : "Check GitHub for code"

---

### Jeudi : Article Medium

Titre : "I Built a Multi-Agent Trading System with Diffusion Models — Here's What I Learned"

Sections :
1. Why traditional RL fails in finance
2. Multi-Agent approach (adversarial learning)
3. Diffusion models for data augmentation
4. Implementation details
5. Results & lessons learned

---

### Vendredi : Préparation Stage

- [ ] CV updated
- [ ] Pitch 1 page prêt
- [ ] GitHub impeccable (README, badges, doc)
- [ ] LinkedIn post sur le projet

---

## ✅ Checklist Finale

### Must-Have
- [ ] Multi-Agent env fonctionnel
- [ ] MAPPO training pipeline
- [ ] Diffusion model entraîné (10k trajectoires)
- [ ] Pipeline intégré Diffusion → MARL
- [ ] Backtesting avec CVaR, MDD, Sortino
- [ ] Tests (coverage > 70%)
- [ ] Documentation Sphinx
- [ ] README avec résultats

### Nice-to-Have
- [ ] MADDPG alternative
- [ ] Sentiment analysis
- [ ] Stress testing sur crises
- [ ] Rapport PDF automatisé
- [ ] Dashboard Streamlit upgraded
- [ ] CI/CD (GitHub Actions)
- [ ] Démo vidéo

### Bonus (Si Temps)
- [ ] Paper trading en temps réel
- [ ] API REST pour le modèle
- [ ] Dockerize le projet
- [ ] Blog post technique

---

## 🎯 Success Criteria

Pour le stage :
1. ✅ **Sharpe Ratio** : > 1.5
2. ✅ **Max Drawdown** : < 15%
3. ✅ **CVaR (95%)** : < 4%
4. ✅ **Alpha vs B&H** : > 3%
5. ✅ **Code Quality** : Tests, docs, clean

---

Bonne chance pour ton stage ! 🚀💼