# evaluate.py
import io
import os
import json
import contextlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_loader import load_data, DataConfig
from environment import TradingEnv, EnvConfig


# ============================================================
# CONFIG D'ÉVALUATION
# ============================================================
# ⚠️  Doit rester alignée sur la config d'ENTRAÎNEMENT (train.py __main__).
# Évaluer avec d'autres frais/fenêtre que ceux vus à l'entraînement
# fausse les métriques (ex : tc=0.002 à l'éval vs 0.001 au train).
EVAL_ENV_CFG = EnvConfig(
    initial_capital  = 10_000.0,
    transaction_cost = 0.001,
    window_size      = 10,
    max_drawdown_pct = 0.25,
    reward_scaling   = 100.0,
)

# Dates : source de vérité unique = DataConfig (data_loader.py)
DATES = DataConfig()


# ============================================================
# HELPERS INTERNES
# ============================================================

def _make_env(data, cfg):
    """Crée un TradingEnv silencieux (sans spam de print)."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = TradingEnv(data=data, cfg=cfg)
    return env


def _metrics_from_series(portfolio: np.ndarray, prices: np.ndarray) -> dict:
    """
    Métriques risk-adjusted calculées sur la trajectoire d'un épisode.

    - sharpe  : mean/std des returns journaliers, annualisé √252
    - sortino : comme le Sharpe mais ne pénalise que la volatilité BAISSIÈRE
    - calmar  : return annualisé / max drawdown (rendement par unité de pire perte)
    - cvar_95 : moyenne des 5 % pires returns journaliers (queue de distribution)
    """
    returns = np.diff(portfolio) / (portfolio[:-1] + 1e-8)
    n_days  = len(portfolio)

    total_ret = (portfolio[-1] - portfolio[0]) / portfolio[0]
    bh_ret    = (prices[-1] - prices[0]) / prices[0]

    peak   = np.maximum.accumulate(portfolio)
    max_dd = float(np.max((peak - portfolio) / (peak + 1e-8)))

    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

    downside = returns[returns < 0]
    if len(downside) > 1:
        sortino = float(np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(252))
    else:
        sortino = float("nan")

    ann_ret = (1.0 + total_ret) ** (252.0 / max(n_days, 1)) - 1.0
    calmar  = float(ann_ret / max_dd) if max_dd > 1e-8 else float("nan")

    if len(returns) >= 20:
        var_95  = np.quantile(returns, 0.05)
        cvar_95 = float(np.mean(returns[returns <= var_95]))
    else:
        cvar_95 = float("nan")

    return {
        "return"  : total_ret,
        "bh"      : bh_ret,
        "alpha"   : total_ret - bh_ret,
        "beat_bh" : total_ret > bh_ret,
        "max_dd"  : max_dd,
        "sharpe"  : sharpe,
        "sortino" : sortino,
        "calmar"  : calmar,
        "cvar_95" : cvar_95,
    }


def _run_episode(model, env, seed, vec_normalize=None, random_start=True,
                 freeze_after_stop=False):
    """
    Lance un épisode complet. Retourne les métriques brutes.

    vec_normalize : VecNormalize chargé depuis vec_normalize.pkl.
    Si fourni, les observations sont normalisées avant model.predict()
    — indispensable car le modèle a été entraîné sur des obs normalisées.

    random_start=False : départ déterministe au début des données,
    l'épisode couvre TOUT le split (métriques reproductibles).

    freeze_after_stop : si le kill-switch drawdown stoppe l'épisode,
    la stratégie est considérée liquidée en cash (valeur gelée) jusqu'à
    la fin de la fenêtre. Sans ça, chaque épisode stoppé se termine PAR
    CONSTRUCTION à son point bas et se compare à un B&H d'horizon
    différent → alphas incomparables entre épisodes/modèles.
    """
    terminated = False
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed, options={"random_start": random_start})
        done   = False
        while not done:
            if vec_normalize is not None:
                obs_input = vec_normalize.normalize_obs(
                    np.array([obs], dtype=np.float32)
                )[0]
            else:
                obs_input = obs
            action, _ = model.predict(obs_input, deterministic=True)
            # Discret : SB3 renvoie un array 0-d → cast int.
            # Continu : le vecteur de poids passe tel quel (clippé par l'env).
            step_action = action if env.cfg.continuous else int(action)
            obs, _, terminated, truncated, _ = env.step(step_action)
            done = terminated or truncated

    portfolio = np.array(env.history["portfolio_values"], dtype=float)
    prices    = np.array(env.history["prices"], dtype=float)
    actions   = np.array(env.history["actions"])

    if terminated and freeze_after_stop:
        # _current_step pointe déjà sur le jour suivant le stop.
        # Borne exclusive _seg_end : un épisode qui va au bout logge son
        # dernier prix à seg_end - 1 → même point final pour tous les épisodes.
        rest = env.prices[env._current_step: env._seg_end].astype(float)
        if len(rest) > 0:
            portfolio = np.concatenate([portfolio, np.full(len(rest), portfolio[-1])])
            prices    = np.concatenate([prices, rest])

    metrics = _metrics_from_series(portfolio, prices)

    # Distribution des décisions — MÊMES clés dans les deux modes pour que
    # l'aval (frames, dashboard, JSON) reste compatible. En continu :
    # long/flat/short lus sur le poids (seuil ±0.2), hold = pas de
    # rebalancement significatif (|Δw| < 0.01).
    if env.cfg.continuous and len(actions):
        w = actions.astype(float)
        dw = np.abs(np.diff(w, prepend=w[0]))
        action_stats = {
            "hold_pct"  : float((dw < 0.01).mean()),
            "long_pct"  : float((w > 0.2).mean()),
            "flat_pct"  : float((np.abs(w) <= 0.2).mean()),
            "short_pct" : float((w < -0.2).mean()),
        }
    else:
        action_stats = {
            "hold_pct"  : (actions == 0).mean(),
            "long_pct"  : (actions == 1).mean(),
            "flat_pct"  : (actions == 2).mean(),
            "short_pct" : (actions == 3).mean(),
        }

    metrics.update({
        "n_trades"         : env.history["n_trades"],
        **action_stats,
        "n_steps"          : len(portfolio),
        "terminated_early" : bool(terminated),   # stoppé par drawdown max
    })
    return metrics


def load_model_and_norm(model_path, data, cfg_env):
    """Charge le modèle PPO + son VecNormalize (si présent à côté du .zip)."""
    model = PPO.load(model_path)

    vec_normalize      = None
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        dummy_env     = DummyVecEnv([lambda: _make_env(data, cfg_env)])
        vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
        vec_normalize.training    = False
        vec_normalize.norm_reward = False

    return model, vec_normalize


# ============================================================
# API D'ÉVALUATION
# ============================================================

def evaluate_full(model_path, data, cfg_env=EVAL_ENV_CFG):
    """
    Évaluation DÉTERMINISTE sur le split complet (départ fixe, politique
    deterministic). C'est la métrique de référence : reproductible,
    couvre toute la période, comparable entre modèles.

    Si le kill-switch drawdown stoppe l'épisode avant la fin, la stratégie
    est considérée liquidée en cash (valeur gelée) jusqu'à la fin du split.
    Ainsi return/alpha/B&H sont TOUJOURS calculés sur la même fenêtre pour
    tous les modèles — sinon chaque modèle serait comparé au B&H de sa
    propre fenêtre tronquée, ce qui rend les alphas incomparables.
    """
    model, vec_normalize = load_model_and_norm(model_path, data, cfg_env)
    env = _make_env(data, cfg_env)
    return _run_episode(model, env, seed=42,
                        vec_normalize=vec_normalize, random_start=False,
                        freeze_after_stop=True)


def evaluate_one(model_path, test_data, cfg_env=EVAL_ENV_CFG,
                 seeds=(42, 123, 456, 789, 1337)):
    """
    Robustesse : évalue sur plusieurs sous-fenêtres ALÉATOIRES du split
    (une par seed). Mesure la stabilité de la stratégie selon le point
    d'entrée — complément de evaluate_full, pas un substitut.
    """
    model, vec_normalize = load_model_and_norm(model_path, test_data, cfg_env)

    records = []
    for seed in seeds:
        env = _make_env(test_data, cfg_env)
        records.append(_run_episode(model, env, seed, vec_normalize,
                                    freeze_after_stop=True))

    df = pd.DataFrame(records)

    return {
        "return"     : df["return"].mean(),
        "return_std" : df["return"].std(),
        "bh"         : df["bh"].mean(),
        "alpha"      : df["alpha"].mean(),
        "alpha_std"  : df["alpha"].std(),
        "beat_bh"    : int(df["beat_bh"].sum()),
        "n_seeds"    : len(seeds),
        "max_dd"     : df["max_dd"].mean(),
        "sharpe"     : df["sharpe"].mean(),
        "n_trades"   : df["n_trades"].mean(),
        "hold_pct"   : df["hold_pct"].mean(),
        "long_pct"   : df["long_pct"].mean(),
        "flat_pct"   : df["flat_pct"].mean(),
        "short_pct"  : df["short_pct"].mean(),
        "n_steps"    : df["n_steps"].mean(),
    }


def _market_context(bh_return):
    """Description textuelle de la tendance du marché sur la période."""
    if bh_return > 0.15:
        return f"HAUSSIER ({bh_return:+.1%})"
    elif bh_return > 0.02:
        return f"légèrement haussier ({bh_return:+.1%})"
    elif bh_return > -0.05:
        return f"neutre ({bh_return:+.1%})"
    elif bh_return > -0.15:
        return f"légèrement baissier ({bh_return:+.1%})"
    else:
        return f"BAISSIER ({bh_return:+.1%})"


def _fmt(x, kind="pct"):
    """Formatage tolérant aux NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "   —  "
    if kind == "pct":
        return f"{x:+7.1%}"
    return f"{x:7.2f}"


# ============================================================
# RAPPORT PRINCIPAL — PERFORMANCE TEST
# ============================================================

def compare_models(ticker=None):
    """
    Rapport de performance : Single vs Multi vs Buy & Hold sur le test set.

    Section 1 — Full split (déterministe) : métrique de référence.
    Section 2 — Métriques risk-adjusted vs cibles du roadmap.
    Section 3 — Robustesse sur 5 sous-fenêtres aléatoires.
    Section 4 — Distribution des actions (diagnostic de biais).
    """
    ticker  = ticker or DATES.ticker
    cfg_env = EVAL_ENV_CFG
    cfg_data = DataConfig(ticker=ticker)

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test_data, _ = load_data(cfg_data)

    models = [
        (f"Single ({ticker})",  "models/ppo_single/best_model.zip"),
        ("Multi (5 tickers)",   "models/ppo_multi/best_model.zip"),
    ]

    full = {label: evaluate_full(path, test_data, cfg_env) for label, path in models}
    rob  = {label: evaluate_one(path, test_data, cfg_env)  for label, path in models}

    bh = list(full.values())[0]["bh"]

    # ── En-tête ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RAPPORT D'ÉVALUATION — PPO Trading Agent vs B&H")
    print(f"{'═'*70}")
    print(f"  Actif      : {ticker}")
    print(f"  Période    : {DATES.start_date} → {DATES.end_date} "
          f"(test = derniers {100 - int((DATES.train_ratio + DATES.val_ratio)*100)} %)")
    print(f"  Marché test: {_market_context(bh)}")
    print(f"{'═'*70}")

    # ── 1. Full split déterministe ─────────────────────────────
    print(f"\n  1. TEST SET COMPLET (déterministe — métrique de référence)")
    print(f"  {'─'*66}")
    print(f"  {'Modèle':<20} {'Return':>8} {'Alpha':>8} {'MaxDD':>7} "
          f"{'Sharpe':>7} {'Trades':>7}  Fin")
    print(f"  {'─'*66}")
    for label, _ in models:
        r   = full[label]
        end = "⚠️ stop DD" if r["terminated_early"] else "complet"
        print(f"  {label:<20} {r['return']:>+8.1%} {r['alpha']:>+8.1%} "
              f"{r['max_dd']:>7.1%} {r['sharpe']:>7.2f} {r['n_trades']:>7.0f}  {end}")
    print(f"  {'─'*66}")
    print(f"  {'Buy & Hold (réf.)':<20} {bh:>+8.1%} {'0.0%':>8} {'':>7} {'':>7} {'':>7}")

    # ── 2. Risk metrics vs cibles roadmap ──────────────────────
    print(f"\n  2. MÉTRIQUES RISK-ADJUSTED (test complet) — cibles roadmap.md")
    print(f"  {'─'*66}")
    print(f"  {'Modèle':<20} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} "
          f"{'CVaR95':>7} {'MaxDD':>7}")
    print(f"  {'Cible':<20} {'>1.5':>7} {'—':>8} {'>1.0':>7} {'>-4%':>7} {'<15%':>7}")
    print(f"  {'─'*66}")
    for label, _ in models:
        r = full[label]
        print(f"  {label:<20} {_fmt(r['sharpe'], 'f')} {_fmt(r['sortino'], 'f')} "
              f"{_fmt(r['calmar'], 'f')} {_fmt(r['cvar_95'])} {r['max_dd']:>7.1%}")

    # ── 3. Robustesse ──────────────────────────────────────────
    n = list(rob.values())[0]["n_seeds"]
    print(f"\n  3. ROBUSTESSE ({n} sous-fenêtres aléatoires du test set)")
    print(f"  {'─'*66}")
    print(f"  {'Modèle':<20} {'Alpha moyen':>14} {'Beat B&H':>9}")
    print(f"  {'─'*66}")
    for label, _ in models:
        r = rob[label]
        print(f"  {label:<20} {r['alpha']:>+8.1%}±{r['alpha_std']:.1%} "
              f"{r['beat_bh']}/{n:>6}")

    # ── 4. Actions ─────────────────────────────────────────────
    print(f"\n  4. ACTIONS (test complet — diagnostic biais)")
    print(f"  {'─'*66}")
    print(f"  {'Modèle':<20} {'Hold':>7} {'Long':>7} {'Flat':>7} {'Short':>7}  Note")
    print(f"  {'─'*66}")
    for label, _ in models:
        r = full[label]
        if r["hold_pct"] > 0.70:
            note = "Trop passif"
        elif r["long_pct"] > 0.55:
            note = "Biais Long"
        elif r["short_pct"] > 0.55:
            note = "Biais Short"
        else:
            note = "Équilibré"
        print(f"  {label:<20} {r['hold_pct']:>7.1%} {r['long_pct']:>7.1%} "
              f"{r['flat_pct']:>7.1%} {r['short_pct']:>7.1%}  {note}")

    # ── Verdict ────────────────────────────────────────────────
    print(f"\n  VERDICT")
    print(f"  {'─'*66}")
    labels = [label for label, _ in models]
    best_alpha = max(labels, key=lambda l: full[l]["alpha"])
    best_dd    = min(labels, key=lambda l: full[l]["max_dd"])
    print(f"  Meilleur alpha (full test) : {best_alpha} "
          f"({full[best_alpha]['alpha']:+.1%})")
    print(f"  Risque plus faible         : {best_dd} "
          f"(MaxDD {full[best_dd]['max_dd']:.1%})")
    beat_any = [l for l in labels if full[l]["beat_bh"]]
    if beat_any:
        print(f"  ✅ Bat le B&H sur le test complet : {', '.join(beat_any)}")
    else:
        print(f"  ❌ Aucun modèle ne bat le B&H sur le test complet.")
    print(f"{'═'*70}\n")

    return full, rob


# ============================================================
# DIAGNOSTIC OVERFITTING — Train / Val / Test
# ============================================================

def check_overfitting_both():
    """
    Évalue chaque modèle sur les 3 splits temporels (full split déterministe).

    Comment lire ce tableau :
    - Un bon modèle a des alphas stables entre val et test.
    - Un grand écart Train→Val/Test indique de l'overfitting (mémorisation).
    - L'alpha train du Multi est peu fiable (entraîné sur données mélangées ≠ AAPL pur).
    """
    cfg_env  = EVAL_ENV_CFG
    cfg_data = DataConfig()

    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test, _ = load_data(cfg_data)

    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC OVERFITTING — Train / Val / Test (full split, {cfg_data.ticker})")
    print(f"{'═'*70}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:<6}: {split.index[0].date()} → {split.index[-1].date()} "
              f"({len(split)} jours)")
    print(f"{'═'*70}")

    for model_name, model_path, note in [
        ("Single (AAPL)",     "models/ppo_single/best_model.zip", ""),
        ("Multi (5 tickers)", "models/ppo_multi/best_model.zip",
         "\n  ⚠️  alpha train peu comparable (entraîné sur données mélangées)"),
    ]:
        print(f"\n  ── {model_name}{note}")
        print(f"  {'Split':<7} {'Return':>9} {'B&H':>9} {'Alpha':>9} {'Sharpe':>8}  Verdict")
        print(f"  {'─'*56}")

        alphas = {}
        for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
            r = evaluate_full(model_path, split_data, cfg_env)
            alphas[split_name] = r["alpha"]

            if r["alpha"] > 0.05:
                verdict = "bat le marché ✅"
            elif r["alpha"] > -0.05:
                verdict = "proche du marché"
            else:
                verdict = "sous-performe ❌"
            if r["terminated_early"]:
                verdict += " (stop DD)"

            print(
                f"  {split_name:<7} "
                f"{r['return']:>+8.1%} "
                f"{r['bh']:>+8.1%} "
                f"{r['alpha']:>+8.1%} "
                f"{r['sharpe']:>8.2f}  "
                f"{verdict}"
            )

        gap_val_test = abs(alphas["Val"] - alphas["Test"])
        print(f"  {'─'*56}")
        print(f"  Écart Val→Test : {gap_val_test:.1%}", end="  →  ")
        if gap_val_test < 0.05:
            print("bonne généralisation ✅")
        elif gap_val_test < 0.15:
            print("généralisation acceptable")
        else:
            print("instabilité détectée ⚠️")

    print(f"\n{'═'*70}\n")


# ============================================================
# GÉNÉRALISATION CROSS-TICKER
# ============================================================

def generalization_report(model_path="models/ppo_multi/best_model.zip",
                          tickers=None, label="Multi (5 tickers)"):
    """
    Évalue un modèle sur le test set de CHAQUE ticker (full split).
    Mesure si le modèle a appris des patterns généraux ou mémorisé un actif.
    """
    tickers = tickers or DATES.tickers
    cfg_env = EVAL_ENV_CFG

    print(f"\n{'═'*70}")
    print(f"  GÉNÉRALISATION CROSS-TICKER — {label}")
    print(f"{'═'*70}")
    print(f"  {'Ticker':<8} {'Return':>9} {'B&H':>9} {'Alpha':>9} "
          f"{'MaxDD':>7} {'Sharpe':>7}  Fin")
    print(f"  {'─'*62}")

    alphas = []
    for ticker in tickers:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, test_data, _ = load_data(DataConfig(ticker=ticker))
            r = evaluate_full(model_path, test_data, cfg_env)
        except Exception as e:
            print(f"  {ticker:<8} ❌ {e}")
            continue

        alphas.append(r["alpha"])
        end = "stop DD" if r["terminated_early"] else "complet"
        print(f"  {ticker:<8} {r['return']:>+9.1%} {r['bh']:>+9.1%} "
              f"{r['alpha']:>+9.1%} {r['max_dd']:>7.1%} {r['sharpe']:>7.2f}  {end}")

    if alphas:
        print(f"  {'─'*62}")
        n_pos = sum(a > 0 for a in alphas)
        print(f"  Alpha moyen : {np.mean(alphas):+.1%} | "
              f"positif sur {n_pos}/{len(alphas)} tickers")
    print(f"{'═'*70}\n")


# ============================================================
# STRESS TESTS — les questions qu'un desk pose en premier
# ============================================================

def stress_report(model_paths=None, data=None, ticker=None,
                  fee_grid=(0.0, 0.0005, 0.001, 0.002, 0.003),
                  json_path="reports/stress_tests.json"):
    """
    Deux stress-tests d'exécution (aucun réentraînement — on stresse les
    hypothèses de coût et de risque, pas l'apprentissage) :

    1. Sensibilité aux frais : le modèle (entraîné à 0,1 %/trade) est évalué
       sous une grille de coûts par trade. Répond à « ton alpha survit-il à
       des frais réalistes (spread, slippage, emprunt de titres) ? »
    2. Ablation du kill-switch : max_drawdown_pct = 1.0 (le stop ruine à 5 %
       du capital reste actif). Répond à « quelle part de ton alpha vient de
       la règle de stop plutôt que des décisions de l'agent ? »
    """
    from dataclasses import replace

    ticker = ticker or DATES.ticker
    if model_paths is None:
        model_paths = {
            f"Single ({ticker})":  "models/ppo_single/best_model.zip",
            "Multi (5 tickers)":   "models/ppo_multi/best_model.zip",
        }
    if data is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, data, _ = load_data(DataConfig(ticker=ticker))

    results = {"ticker": ticker, "fee_grid": {}, "no_killswitch": {}}

    print(f"\n{'═'*70}")
    print(f"  STRESS TESTS — sensibilité aux frais & ablation du kill-switch")
    print(f"  (test set {ticker}, modèles entraînés avec frais 0,10 %)")
    print(f"{'═'*70}")

    # ── 1. Grille de frais ─────────────────────────────────────
    print(f"\n  1. SENSIBILITÉ AUX FRAIS (par trade, aller simple)")
    header = "  " + f"{'Modèle':<20}" + "".join(f"{100*f:>9.2f}%" for f in fee_grid)
    print(f"  {'─'*66}\n{header}   ← frais\n  {'─'*66}")

    for label, path in model_paths.items():
        row = {}
        for fee in fee_grid:
            cfg = replace(EVAL_ENV_CFG, transaction_cost=fee)
            r = evaluate_full(path, data, cfg)
            row[f"{fee:.4f}"] = {"return": r["return"], "alpha": r["alpha"],
                                 "n_trades": r["n_trades"]}
        results["fee_grid"][label] = row
        print("  " + f"{label:<20}"
              + "".join(f"{100*v['alpha']:>+9.1f}" for v in row.values())
              + "   alpha (%)")

    # ── 2. Sans kill-switch ────────────────────────────────────
    print(f"\n  2. ABLATION DU KILL-SWITCH (drawdown libre, stop ruine conservé)")
    print(f"  {'─'*66}")
    print(f"  {'Modèle':<20} {'Return':>9} {'Alpha':>9} {'MaxDD':>8}   vs avec stop 25%")
    print(f"  {'─'*66}")

    for label, path in model_paths.items():
        cfg_ns = replace(EVAL_ENV_CFG, max_drawdown_pct=1.0)
        r_ns   = evaluate_full(path, data, cfg_ns)
        r_ref  = evaluate_full(path, data, EVAL_ENV_CFG)
        results["no_killswitch"][label] = {
            "return": r_ns["return"], "alpha": r_ns["alpha"],
            "max_dd": r_ns["max_dd"],
            "alpha_with_stop": r_ref["alpha"],
            "stop_contribution": r_ref["alpha"] - r_ns["alpha"],
        }
        print(f"  {label:<20} {r_ns['return']:>+9.1%} {r_ns['alpha']:>+9.1%} "
              f"{r_ns['max_dd']:>8.1%}   alpha {r_ref['alpha']:+.1%} → "
              f"contribution du stop : {r_ref['alpha'] - r_ns['alpha']:+.1%}")

    print(f"{'═'*70}\n")

    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    compare_models()
    check_overfitting_both()
    generalization_report()
    stress_report()
