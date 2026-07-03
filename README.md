# DRL portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

## Overview
**DRL portfolio** is an experimental project applying **Deep Reinforcement Learning (DRL)** to financial markets. The goal is to train an autonomous agent capable of making trading decisions (Long/Flat/Short/Hold) by maximizing a risk-adjusted reward function (alpha vs Buy & Hold, with transaction-cost and drawdown penalties).

Unlike traditional "black box" approaches, this project leverages **Financial Feature Engineering** based on stochastic modeling concepts (Volatility regimes, Mean Reversion, Momentum) to guide the agent's learning process.

## Key Features
* **Custom Trading Environment:** Built on top of `Gymnasium` to simulate realistic market conditions (latency, transaction fees).
* **Stochastic Feature Engineering:** Inputs include Log-returns, Rolling Volatility (GARCH proxy), and Ornstein-Uhlenbeck mean reversion signals.
* **State-of-the-Art RL Algorithms:** Implementation of **PPO** (Proximal Policy Optimization) and **A2C** using `Stable-Baselines3`.
* **Backtesting Engine:** Deterministic full-split evaluation + random-window robustness checks, with risk-adjusted metrics (Sharpe, Sortino, Calmar, CVaR 95%, Max Drawdown) and a 25% max-drawdown kill-switch.
* **Anti-Overfitting Design:** Chronological train/val/test splits (2010→2023), scaler fitted on train only, multi-ticker training with per-ticker episode segments, model selection on a clean validation signal.
* **Walk-Forward Validation:** 5 rolling retrainings (anchored expanding window), each tested on a fully out-of-sample year (2018→2022) across 5 assets — 25 out-of-sample year×asset cells.
* **Test Suite:** 67 pytest tests covering the data pipeline, trading mechanics (exact fee accounting), Gymnasium compliance, real short PPO training runs, evaluation, walk-forward and the Streamlit dashboard.

## Tech Stack
* **Core:** Python
* **ML & RL:** Stable-Baselines3, PyTorch, Gymnasium
* **Data & Analysis:** Pandas, NumPy, yfinance, TA-Lib
* **Visualization:** Matplotlib

## Results

Test set = last 15% of 2010→2023 (Feb 2021 → Dec 2022, unseen during training — bull 2021 + bear 2022).
Deterministic full-split evaluation, observations normalized with the training `VecNormalize` stats, 0.1% fees per trade. If the 25% drawdown kill-switch fires, the strategy is held in cash until the end of the window (comparable horizons).

| Model | Return | Alpha vs B&H | Max DD | Sharpe | Sortino | CVaR 95% | Robustness* |
|---|---|---|---|---|---|---|---|
| PPO Single (AAPL) | +15.5% | +19.2% | 26.2% | 0.44 | 0.57 | -3.4% | 5/5 |
| **PPO Multi (5 tickers)** | **+23.8%** | **+27.5%** | 25.2% | **0.57** | **0.77** | -3.7% | **5/5** |
| Buy & Hold AAPL (ref.) | -3.7% | — | — | — | — | — | — |

*\*Number of random test sub-windows (out of 5) where the agent beats Buy & Hold.*

**Cross-ticker generalization** — the multi-ticker agent evaluated on each ticker's unseen test split achieves a **positive alpha on 5/5 assets** (AAPL +27.5%, TSLA +28.6%, MSFT +14.7%, GOOGL +12.7%, SPY +2.7%), mean **+17.2%**.

**Walk-forward validation (the honest picture)** — 5 rolling retrainings tested on out-of-sample years 2018→2022 × 5 assets reveal a **regime-dependent defensive profile**: strongly positive alpha in down years (2018: +9.2%, 2022: +37.6%) and positive *absolute* returns in 4/5 years, but the agent lags Buy & Hold in strong bull years (2020: stopped in the COVID crash, it misses the violent rebound). Median cell alpha: -2.0%, 48% of cells positive. The single-split result above must be read through this lens: 2021-2022 is exactly the regime where a defensive agent shines.

**Seed robustness & a falsified fix (Act 3)** — retraining across 5 seeds shows the single-asset alpha is a lottery (AAPL: -11% to +44%, positive 2/5) while the **cross-asset mean holds: +14.5% ± 6.5, positive for 5/5 seeds** — that is the defensible claim. Adding regime features to the observation (dist-to-1y-high, 200-day trend; obs 52→72) was then tested against this noise band and **failed consistently** (walk-forward: 2021 and 2022 both degraded): seeing the regime is not enough — the *reward* must incentivize using it. Next lever: continuous position sizing with a regime-aware risk penalty.

Known limitations: regime-dependent performance (see walk-forward above), alpha mostly carried by the stop rule, Max Drawdown bounded by the kill-switch (~25%), absolute Sharpe below 1. Reproduce with `python train.py`, `python evaluate.py`, `python walk_forward.py`, `python seed_robustness.py`, `python regime_experiment.py`; run the test suite with `python -m pytest`.

## Documentation & Dashboards

* 📄 **[Rapport pédagogique (PDF)](reports/rapport_drl_portfolio.pdf)** — full methodology in French: MDP formulation, features ↔ econometrics (stationarity, GARCH, look-ahead bias), reward ↔ Jensen's alpha, risk metrics ↔ portfolio theory, the complete timeline of bugs found and fixed, and an honest overfitting analysis. Source: [rapport_drl_portfolio.tex](reports/rapport_drl_portfolio.tex) (compile with `tectonic` or Overleaf).
* 📊 **[Live results dashboard](https://martinchg.github.io/DRL-portfolio/)** — interactive Plotly charts, the 7-step research timeline, and an honest "what works / what doesn't" reading. No installation needed (source: [docs/index.html](docs/index.html), served by GitHub Pages).
* 🖥️ **Interactive dashboard** — `streamlit run dashboard.py` (data/feature exploration, model evaluation with proper observation normalization).
* 🔁 Regenerate result figures + metrics + static dashboard: `python reports/build_assets.py` ; regenerate the conceptual/pedagogical figures (MDP loop, PPO clipping, Ornstein-Uhlenbeck, efficient frontier…): `python reports/build_concept_figures.py`.

## Motivation
This project bridges the gap between **Quantitative Finance** (Stochastic Calculus, Portfolio Theory) and **Modern AI** (Deep Learning). It was developed to explore how model-free agents can adapt to non-stationary market environments where traditional parametric models often fail.

## Disclaimer
This software is for **educational and research purposes only**. It is not financial advice. Do not use this code to trade real money. The author assumes no responsibility for any financial losses.

---
*Project developed by **Martin Chassaing** – Engineering Student at IMT Atlantique & Economics Student at Université Paris Dauphine.*
