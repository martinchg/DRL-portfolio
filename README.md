# DRL portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

## Overview
**DRL portfolio** is an experimental project applying **Deep Reinforcement Learning (DRL)** to financial markets. The goal is to train an autonomous agent capable of making trading decisions (Buy/Sell/Hold) by maximizing a risk-adjusted reward function (Sharpe Ratio).

Unlike traditional "black box" approaches, this project leverages **Financial Feature Engineering** based on stochastic modeling concepts (Volatility regimes, Mean Reversion, Momentum) to guide the agent's learning process.

## Key Features
* **Custom Trading Environment:** Built on top of `Gymnasium` to simulate realistic market conditions (latency, transaction fees).
* **Stochastic Feature Engineering:** Inputs include Log-returns, Rolling Volatility (GARCH proxy), and Ornstein-Uhlenbeck mean reversion signals.
* **State-of-the-Art RL Algorithms:** Implementation of **PPO** (Proximal Policy Optimization) and **A2C** using `Stable-Baselines3`.
* **Backtesting Engine:** Robust evaluation framework comparing the agent against "Buy & Hold" and traditional strategies.

## Tech Stack
* **Core:** Python
* **ML & RL:** Stable-Baselines3, PyTorch, Gymnasium
* **Data & Analysis:** Pandas, NumPy, yfinance, TA-Lib
* **Visualization:** Matplotlib

## Motivation
This project bridges the gap between **Quantitative Finance** (Stochastic Calculus, Portfolio Theory) and **Modern AI** (Deep Learning). It was developed to explore how model-free agents can adapt to non-stationary market environments where traditional parametric models often fail.

## Disclaimer
This software is for **educational and research purposes only**. It is not financial advice. Do not use this code to trade real money. The author assumes no responsibility for any financial losses.

---
*Project developed by **Martin Chassaing** – Engineering Student at IMT Atlantique & Economics Student at Université Paris Dauphine.*