# 🚀 DRL Portfolio → MARL + Diffusion Models
## Roadmap de Transition (8-10 semaines)

---

## 📋 Vue d'Ensemble

**Objectif Final** : Passer d'un agent PPO simple à un système multi-agent robuste entraîné sur des données synthétiques générées par diffusion.

**Stack Technique** :
- **MARL** : PettingZoo + Ray RLlib (MAPPO/MADDPG)
- **Diffusion** : TimeGrad / Denoising Diffusion (DDPM)
- **Évaluation** : Backtesting avec métriques risk-adjusted

---

## 🎯 Phase 1 : Multi-Agent RL Foundation (Semaines 1-3)

### Semaine 1 : Setup & Architecture Multi-Agent

#### Objectifs
- [ ] Restructurer le code pour supporter plusieurs agents
- [ ] Définir les rôles des agents (Portfolio Manager vs Market Adversary)
- [ ] Créer un environnement PettingZoo compatible

#### Tâches Concrètes

**1.1 - Installation des dépendances**
```bash
pip install pettingzoo==1.24.3
pip install ray[rllib]==2.9.0
pip install supersuit  # Wrappers pour PettingZoo
```

**1.2 - Créer `envs/multi_agent_trading_env.py`**

Structure de l'environnement :
- **Agent 1 (Portfolio Manager)** : 
  - Actions : Buy/Sell/Hold sur N assets
  - Observation : Features de marché + état du portfolio
  - Reward : Sharpe Ratio - Pénalité de transaction
  
- **Agent 2 (Market Adversary)** :
  - Actions : Simuler des chocs de liquidité, créer de la volatilité
  - Observation : État du marché global
  - Reward : Inverse du Sharpe de l'Agent 1 (adversarial)

**1.3 - Fichiers à créer**
```
envs/
├── __init__.py
├── multi_agent_trading_env.py    # Environnement PettingZoo
├── market_adversary.py            # Logique de l'adversaire
└── rewards.py                     # Reward shaping pour les 2 agents
```

#### Livrables
- ✅ Environnement multi-agent fonctionnel
- ✅ Tests unitaires avec `check_env()` de PettingZoo
- ✅ Notebook de démo avec agents aléatoires

---

### Semaine 2 : Implémentation MAPPO

#### Objectifs
- [ ] Entraîner 2 agents avec MAPPO (Multi-Agent PPO)
- [ ] Comparer les performances avec l'agent simple (baseline)

#### Tâches Concrètes

**2.1 - Créer `train_mappo.py`**
```python
# Structure du script
- Config Ray Tune
- Définir les policies pour chaque agent
- Shared critic network (clé de MAPPO)
- Callbacks pour logging (Sharpe, MDD, Trades)
```

**2.2 - Hyperparamètres MAPPO**
```yaml
algo: MAPPO
lr_schedule: [[0, 3e-4], [500_000, 1e-5]]  # Decay
entropy_coef: 0.01  # Exploration
vf_loss_coef: 0.5
clip_param: 0.2
shared_critic: True  # Important pour la coordination
```

**2.3 - Métriques à tracker**
- Sharpe Ratio (Agent 1 vs Baseline)
- Max Drawdown
- Nombre de trades
- Correlation entre les agents (Nash Equilibrium check)

#### Livrables
- ✅ Script d'entraînement MAPPO fonctionnel
- ✅ Comparaison graphique PPO simple vs MAPPO
- ✅ Logs TensorBoard avec métriques financières

---

### Semaine 3 : Raffinement Multi-Agent

#### Objectifs
- [ ] Tester MADDPG (continuous actions)
- [ ] Implémenter un 3ème agent (Market Maker)
- [ ] Optimiser les reward functions

#### Tâches Concrètes

**3.1 - Alternative MADDPG**
Pour des actions continues (ex : % d'allocation), implémenter MADDPG :
```python
# envs/continuous_trading_env.py
action_space = Box(low=-1.0, high=1.0, shape=(n_assets,))
# -1 = short, 0 = hold, +1 = long
```

**3.2 - Market Maker Agent (optionnel)**
- **Rôle** : Fournir de la liquidité, créer du spread
- **Interaction** : Ajoute des frais dynamiques pour l'Agent 1

**3.3 - Reward Shaping Avancé**
```python
# Reward pour Agent 1 (Portfolio Manager)
reward = (
    alpha * sharpe_ratio 
    - beta * transaction_costs 
    - gamma * max_drawdown_penalty
)

# Reward pour Agent 2 (Adversary)
reward = (
    delta * (-sharpe_ratio_agent1)  # Rendre la vie difficile
    + epsilon * diversity_bonus      # Ne pas juste crasher le marché
)
```

#### Livrables
- ✅ Comparaison MAPPO vs MADDPG
- ✅ Ablation study : impact de chaque composante du reward
- ✅ Dashboard interactif Streamlit pour comparer les agents

---

## 🌊 Phase 2 : Diffusion Models pour la Génération de Données (Semaines 4-6)

### Semaine 4 : Théorie & Setup Diffusion

#### Objectifs
- [ ] Comprendre DDPM (Denoising Diffusion Probabilistic Models)
- [ ] Adapter pour les séries temporelles financières
- [ ] Setup du pipeline de données

#### Tâches Concrètes

**4.1 - Étude théorique**
Papers à lire :
- **TimeGrad** (2021) - Autoregressive Denoising for Time Series
- **Diffusion Models for Time Series** (2023)
- **Score-based Generative Models** (Song et al.)

**4.2 - Installation**
```bash
pip install diffusers==0.25.0  # Hugging Face
pip install pytorch-lightning==2.1.0
pip install einops  # Pour les reshapes
```

**4.3 - Créer `diffusion/`**
```
diffusion/
├── __init__.py
├── ddpm_time_series.py       # Modèle de diffusion
├── noise_scheduler.py        # Beta schedule
├── data_preprocessing.py     # Normalisation, windowing
└── sampling.py               # Génération de trajectoires
```

#### Livrables
- ✅ Notebook de démo DDPM sur données synthétiques (sine wave)
- ✅ Comprendre forward/reverse process visuellement
- ✅ Données financières formatées pour la diffusion

---

### Semaine 5 : Entraînement du Modèle de Diffusion

#### Objectifs
- [ ] Entraîner DDPM sur tes données historiques (AAPL, etc.)
- [ ] Valider la qualité des samples générés
- [ ] Créer un dataset de 10k trajectoires synthétiques

#### Tâches Concrètes

**5.1 - Architecture du modèle**
```python
# diffusion/ddpm_time_series.py

class TimeSeriesDDPM(nn.Module):
    """
    U-Net 1D pour séries temporelles
    Input : [batch, seq_len, features]
    Output : Bruit prédit
    """
    def __init__(self):
        # Encoder : Conv1D + Downsampling
        # Bottleneck : Transformer layers
        # Decoder : TransposeConv1D + Upsampling
```

**5.2 - Training Loop**
```python
# Hyperparams
T = 1000  # Timesteps de diffusion
beta_schedule = "linear"  # ou "cosine"
lr = 1e-4
batch_size = 64
epochs = 100

# Loss : MSE entre bruit ajouté et bruit prédit
loss = F.mse_loss(noise_pred, noise_true)
```

**5.3 - Validation des Samples**
Métriques à calculer :
- **Distribution des returns** : KL divergence avec données réelles
- **Autocorrélation** : Doit matcher les données réelles
- **Volatility clustering** : Présence de régimes GARCH-like
- **Maximum Likelihood** : Score sur un hold-out set

#### Livrables
- ✅ Modèle de diffusion entraîné (checkpoint .pth)
- ✅ 10,000 trajectoires synthétiques (`.npz` ou `.h5`)
- ✅ Rapport de validation (distributions, stats descriptives)

---

### Semaine 6 : Pipeline Intégré Diffusion → MARL

#### Objectifs
- [ ] Connecter la diffusion au training MARL
- [ ] Curriculum learning : données réelles → synthétiques
- [ ] Benchmark : MARL sur données réelles vs synthétiques

#### Tâches Concrètes

**6.1 - Data Generator Wrapper**
```python
# data/synthetic_data_loader.py

class DiffusionDataGenerator:
    def __init__(self, diffusion_model, n_scenarios=1000):
        self.model = diffusion_model
        self.n_scenarios = n_scenarios
    
    def generate_episode(self):
        # Sample une trajectoire depuis le modèle de diffusion
        trajectory = self.model.sample(seq_len=252)  # 1 an
        return trajectory
```

**6.2 - Curriculum Training**
```python
# Phase 1 : 70% données réelles, 30% synthétiques
# Phase 2 : 50/50
# Phase 3 : 30% réelles, 70% synthétiques

# Bénéfice : Évite le "mode collapse" de la diffusion
```

**6.3 - Comparaison**
Entraîner 3 versions de MAPPO :
1. **Baseline** : Données historiques uniquement
2. **Augmented** : 50/50 réel/synthétique
3. **Full Synthetic** : 100% diffusion

Comparer sur un test set de données réelles (2024).

#### Livrables
- ✅ Pipeline automatisé : Diffusion → DataLoader → MARL
- ✅ Graphiques de comparaison (Sharpe, MDD, CVaR)
- ✅ Analyse de robustesse (stress testing)

---

## 🔬 Phase 3 : Features Avancées & Déploiement (Semaines 7-8)

### Semaine 7 : Features Quantitatives

#### Objectifs
- [ ] Ajouter des features de microstructure
- [ ] Intégrer du sentiment analysis (optionnel)
- [ ] Normalisation robuste

#### Tâches Concrètes

**7.1 - Microstructure Features**
```python
# features/microstructure.py

def compute_bid_ask_spread(data):
    # Simulé si pas de données réelles
    spread = data['high'] - data['low']
    return spread / data['close']

def compute_volume_profile(data, window=20):
    # Volume moyen par tranche de prix
    return data['volume'].rolling(window).mean()

def compute_kyle_lambda(data):
    # Price impact (régression returns ~ volume)
    return ols(returns, signed_volume).coef
```

**7.2 - Sentiment Analysis (bonus)**
```python
# Si tu as accès à des news
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis", 
                          model="ProsusAI/finbert")

score = sentiment_model("Tesla stock surges 10%")
# {'label': 'positive', 'score': 0.95}
```

**7.3 - Normalisation Robuste**
```python
# features/normalization.py

class RollingZScore:
    """
    Normalisation glissante pour éviter le look-ahead bias
    """
    def __init__(self, window=252):
        self.window = window
    
    def transform(self, x):
        mu = x.rolling(self.window).mean()
        sigma = x.rolling(self.window).std()
        return (x - mu) / (sigma + 1e-8)
```

#### Livrables
- ✅ 10+ features nouvelles ajoutées
- ✅ Feature importance analysis (SHAP values)
- ✅ Comparaison performances avec/sans nouvelles features

---

### Semaine 8 : Backtesting & Risk Metrics

#### Objectifs
- [ ] Pipeline de backtesting professionnel
- [ ] Calcul de CVaR, MDD, Sortino
- [ ] Stress testing sur événements historiques

#### Tâches Concrètes

**8.1 - Créer `backtest/backtester.py`**
```python
class Backtester:
    def __init__(self, agent, data):
        self.agent = agent
        self.data = data
        
    def run(self):
        # Walk-forward analysis
        # Retrain tous les 3 mois
        
    def compute_metrics(self):
        return {
            'sharpe': self.sharpe_ratio(),
            'sortino': self.sortino_ratio(),
            'calmar': self.calmar_ratio(),
            'max_drawdown': self.mdd(),
            'cvar_95': self.cvar(alpha=0.05),
            'var_95': self.var(alpha=0.05),
        }
```

**8.2 - Stress Testing**
```python
# Tester sur des crises historiques
crisis_periods = {
    'covid': ('2020-02-15', '2020-04-15'),
    'gfc': ('2008-09-01', '2009-03-01'),
    'dotcom': ('2000-03-01', '2002-10-01'),
}

for name, period in crisis_periods.items():
    metrics = backtester.run(period)
    print(f"{name}: Sharpe={metrics['sharpe']:.2f}, MDD={metrics['max_drawdown']:.2%}")
```

**8.3 - Rapport Final**
Générer un PDF avec :
- Equity curves (Agent vs Buy&Hold vs Benchmarks)
- Drawdown chart
- Rolling Sharpe (12 mois)
- Trade analysis (win rate, avg profit/loss)
- Feature attributions

#### Livrables
- ✅ Framework de backtesting complet
- ✅ Rapport PDF automatisé (ReportLab ou LaTeX)
- ✅ Dashboard interactif (Streamlit upgraded)

---

## 📊 Phase 4 : Documentation & Portfolio (Semaines 9-10)

### Semaine 9 : Packaging & Clean Code

#### Objectifs
- [ ] Refactoring complet du code
- [ ] Documentation Sphinx
- [ ] Tests unitaires (coverage > 70%)

#### Tâches Concrètes

**9.1 - Structure finale du repo**
```
DRL-portfolio/
├── README.md                    # Updated avec MARL + Diffusion
├── requirements.txt             # Pinned versions
├── setup.py                     # pip install -e .
├── configs/
│   ├── mappo_config.yaml
│   ├── diffusion_config.yaml
│   └── backtest_config.yaml
├── envs/
│   ├── multi_agent_trading_env.py
│   └── market_adversary.py
├── diffusion/
│   ├── ddpm_time_series.py
│   └── sampling.py
├── models/
│   ├── actor_critic.py
│   └── unet_1d.py
├── features/
│   ├── microstructure.py
│   └── normalization.py
├── backtest/
│   ├── backtester.py
│   └── metrics.py
├── scripts/
│   ├── train_mappo.py
│   ├── train_diffusion.py
│   └── run_backtest.py
├── tests/
│   ├── test_env.py
│   ├── test_diffusion.py
│   └── test_backtest.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_diffusion_demo.ipynb
│   ├── 03_marl_training.ipynb
│   └── 04_results_analysis.ipynb
└── docs/
    ├── conf.py                 # Sphinx config
    ├── architecture.md
    └── api_reference.rst
```

**9.2 - Tests unitaires**
```python
# tests/test_env.py
import pytest
from envs.multi_agent_trading_env import MultiAgentTradingEnv

def test_env_reset():
    env = MultiAgentTradingEnv(data=sample_data)
    obs = env.reset()
    assert 'portfolio_manager' in obs
    assert 'market_adversary' in obs

def test_env_step():
    env = MultiAgentTradingEnv(data=sample_data)
    env.reset()
    actions = {'portfolio_manager': 1, 'market_adversary': 0}
    obs, rewards, dones, infos = env.step(actions)
    assert not dones['__all__']
```

**9.3 - Documentation Sphinx**
```bash
cd docs/
sphinx-quickstart
sphinx-apidoc -o . ../
make html
```

#### Livrables
- ✅ Code 100% type-hinted (mypy compatible)
- ✅ Test coverage > 70%
- ✅ Documentation Sphinx déployée (GitHub Pages)

---

### Semaine 10 : Présentation & GitHub Polish

#### Objectifs
- [ ] README killer avec démo vidéo
- [ ] Article Medium/Blog technique
- [ ] Préparer le pitch pour stage

#### Tâches Concrètes

**10.1 - README Killer**
```markdown
# 🚀 DRL Portfolio — Multi-Agent RL + Diffusion Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]
[![Tests](https://github.com/martinchg/DRL-portfolio/workflows/tests/badge.svg)]
[![License](https://img.shields.io/badge/License-MIT-green.svg)]

## 🎯 Key Features
- ✅ Multi-Agent PPO (MAPPO) with adversarial training
- ✅ DDPM-based scenario generation (10k+ synthetic trajectories)
- ✅ Professional backtesting framework (CVaR, MDD, Sortino)
- ✅ Microstructure features + sentiment analysis

## 📊 Results
| Model | Sharpe | Max DD | CVaR (95%) | Alpha vs B&H |
|-------|--------|--------|------------|--------------|
| PPO Baseline | 1.2 | -18% | -5.2% | +2.1% |
| MAPPO + Diffusion | **1.8** | **-12%** | **-3.1%** | **+7.4%** |

[📹 Demo Video](link) | [📖 Documentation](link) | [📝 Blog Post](link)
```

**10.2 - Démo Vidéo (2-3 min)**
- Montrer l'environnement multi-agent en action
- Visualiser la diffusion générant des trajectoires
- Dashboard interactif avec résultats

**10.3 - Article Technique**
Publier sur Medium/Dev.to :
- **Titre** : "Building a Multi-Agent Trading System with Diffusion Models"
- **Sections** :
  1. Why MARL for finance?
  2. Diffusion models for robust data generation
  3. Results & lessons learned
  4. Code snippets & architecture

**10.4 - Pitch Stage (1 page)**
```
Problématique : Les modèles RL classiques overfittent sur les données historiques.

Solution : 
1. Multi-Agent RL → Robustesse via adversarial training
2. Diffusion Models → Génération de scénarios inédits
3. Risk-adjusted metrics → CVaR, Stress Testing

Résultats :
- Sharpe +50% vs baseline
- Max Drawdown -33%
- Robuste sur 3 crises historiques (backtested)

Tech Stack : PyTorch, Ray RLlib, PettingZoo, Diffusers
```

#### Livrables
- ✅ README avec badges, GIFs, résultats
- ✅ Vidéo démo sur YouTube
- ✅ Article Medium avec 1000+ vues
- ✅ CV updated avec ce projet en highlight

---

## 📈 Métriques de Succès

### Pour le Stage en Banque

Ce qu'ils veulent voir :
1. ✅ **Risk Management** : MDD < 15%, CVaR bien calculé
2. ✅ **Robustesse** : Performances stables sur plusieurs régimes de marché
3. ✅ **Innovation** : MARL + Diffusion = différenciant
4. ✅ **Code Quality** : Tests, docs, architecture propre
5. ✅ **Business Impact** : Alpha positif vs benchmark

### KPIs Techniques

| Métrique | Target | Ton Projet |
|----------|--------|------------|
| Sharpe Ratio | > 1.5 | ? |
| Max Drawdown | < 15% | ? |
| CVaR (95%) | < 4% | ? |
| Win Rate | > 55% | ? |
| Calmar Ratio | > 1.0 | ? |

---

## 🛠️ Outils & Ressources

### Frameworks
- **MARL** : Ray RLlib, PettingZoo, SMAC
- **Diffusion** : Hugging Face Diffusers, PyTorch Lightning
- **Backtesting** : Backtrader, Zipline, VectorBT

### Papers à Lire
1. **TimeGrad** (Salesforce, 2021)
2. **MAPPO** (Yu et al., 2022)
3. **Score-based Generative Models** (Song et al., 2021)
4. **Deep Hedging** (Buehler et al., 2019)

### Datasets Additionnels
- **Alpha Vantage** : API gratuite pour stocks
- **Yahoo Finance** : yfinance (ton actuel)
- **Quandl** : Données macro
- **Twitter Sentiment** : Scraper ou API

---

## ⚠️ Pièges à Éviter

### Technique
- ❌ **Look-ahead bias** : Normalisation sur tout le dataset
- ❌ **Mode collapse** : Diffusion génère toujours les mêmes trajectoires
- ❌ **Overfitting** : Agent apprend les patterns du générateur

### Projet
- ❌ **Scope creep** : Reste focus, pas besoin de 15 features dès le début
- ❌ **Manque de validation** : Toujours backtest sur données out-of-sample
- ❌ **Documentation négligée** : README = 1ère impression

---

## 🎓 Checklist Finale

### Code
- [ ] Multi-Agent env fonctionnel (PettingZoo)
- [ ] MAPPO training script avec callbacks
- [ ] Modèle de diffusion entraîné et validé
- [ ] 10k+ trajectoires synthétiques générées
- [ ] Pipeline intégré Diffusion → MARL
- [ ] Features microstructure ajoutées
- [ ] Backtesting framework avec CVaR, MDD
- [ ] Tests unitaires (coverage > 70%)
- [ ] Documentation Sphinx

### Présentation
- [ ] README avec démo visuelle
- [ ] Vidéo démo (2-3 min)
- [ ] Article technique publié
- [ ] Résultats benchmarkés vs baselines
- [ ] Pitch 1 page pour stage

### Bonus (Si Temps)
- [ ] Déploiement en temps réel (paper trading)
- [ ] Dashboard web (Streamlit Cloud)
- [ ] API REST pour le modèle
- [ ] Intégration CI/CD (GitHub Actions)

---

## 📞 Questions ?

Si tu bloques sur une étape :
1. Regarde les notebooks dans `notebooks/`
2. Lis la doc dans `docs/`
3. Check les issues GitHub (ou pose-en une nouvelle)
4. DM moi sur LinkedIn (si tu veux)

Bon courage pour ton stage ! 🚀💼

---

**Auteur** : Martin Chassaing  
**Contact** : [GitHub](https://github.com/martinchg) | [LinkedIn](...)  
