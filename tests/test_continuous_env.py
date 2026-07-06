# test_continuous_env.py — Acte 5 : position continue w ∈ [-1, 1].
#
# Même religion que le mode discret : la comptabilité se teste AU CENTIME sur
# des prix contrôlés. L'identité V = cash + shares × P doit tenir à chaque pas,
# frais exacts sur le notionnel échangé.
import numpy as np
import pytest

from environment import TradingEnv, EnvConfig
from tests.conftest import make_manual_features

TC = 0.001   # frais alignés sur la config d'évaluation


def make_env(data, **kw):
    defaults = dict(continuous=True, transaction_cost=TC, window_size=10,
                    initial_capital=10_000.0)
    defaults.update(kw)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return TradingEnv(data=data, cfg=EnvConfig(**defaults))


def test_action_space_box_et_obs_shape(flat_data):
    env = make_env(flat_data)
    from gymnasium import spaces
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    obs, _ = env.reset(seed=0, options={"random_start": False})
    assert obs.shape == (10 * 5 + 2,)          # inchangé vs discret


def test_slot_position_porte_le_poids(flat_data):
    env = make_env(flat_data)
    env.reset(seed=0, options={"random_start": False})
    obs, _, _, _, _ = env.step(np.array([0.7], dtype=np.float32))
    assert abs(obs[-2] - 0.7) < 1e-6           # avant-dernier = w


def test_comptabilite_exacte_premier_rebalance(flat_data):
    # Prix constant 100, V0 = 10 000, w = 0.5 :
    # notionnel = 5 000, frais = 5, cash = 10 000 - 5 000 - 5 = 4 995,
    # V = 4 995 + 50 × 100 = 9 995. Exact.
    env = make_env(flat_data)
    env.reset(seed=0, options={"random_start": False})
    env.step(np.array([0.5], dtype=np.float32))
    assert abs(env._shares - 50.0) < 1e-9
    assert abs(env._cash - 4995.0) < 1e-9
    assert abs(env._portfolio_value - 9995.0) < 1e-9


def test_identite_comptable_a_chaque_pas(flat_data):
    env = make_env(flat_data)
    env.reset(seed=1, options={"random_start": False})
    rng = np.random.default_rng(3)
    for _ in range(40):
        w = rng.uniform(-1, 1)
        _, _, term, trunc, _ = env.step(np.array([w], dtype=np.float32))
        price = env.prices[env._current_step - 1]
        assert abs(env._portfolio_value - (env._cash + env._shares * price)) < 1e-6
        if term or trunc:
            break


def test_rebalance_noop_zero_frais(flat_data):
    # Prix constant : re-viser le même poids n'échange aucun titre → 0 frais.
    env = make_env(flat_data)
    env.reset(seed=0, options={"random_start": False})
    env.step(np.array([0.5], dtype=np.float32))
    cash_avant, trades_avant = env._cash, env.history["n_trades"]
    env.step(np.array([0.5], dtype=np.float32))
    assert abs(env._cash - cash_avant) < 1e-9
    assert env.history["n_trades"] == trades_avant


def test_short_continu_gagne_quand_le_prix_baisse(declining_data):
    env = make_env(declining_data)
    env.reset(seed=0, options={"random_start": False})
    for _ in range(20):
        _, _, term, trunc, _ = env.step(np.array([-1.0], dtype=np.float32))
        if term or trunc:
            break
    assert env._portfolio_value > env.cfg.initial_capital   # short gagnant


def test_long_fractionnaire_suit_le_marche_a_moitie(rising_data):
    # w = 0.5 sur +3 %/jour : le rendement du portefeuille par pas doit être
    # ≈ la moitié du rendement du marché (aux frais de rebalancement près).
    env = make_env(rising_data)
    env.reset(seed=0, options={"random_start": False})
    env.step(np.array([0.5], dtype=np.float32))
    v0 = env._portfolio_value
    env.step(np.array([0.5], dtype=np.float32))
    step_ret = env._portfolio_value / v0 - 1.0
    assert 0.011 < step_ret < 0.016                          # ~1.5 % ± frais


def test_kill_switch_en_continu(declining_data):
    env = make_env(declining_data, max_drawdown_pct=0.25)
    env.reset(seed=0, options={"random_start": False})
    terminated = False
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated                                        # -3 %/j plein pot → stop


def test_reward_risk_aversion_formule_exacte(gbm_features):
    # Deux envs identiques sur données à vol RÉELLE (GBM σ≈2 %), seule λ
    # change : la différence de reward = λ·σ̂·|w|·scaling (≈ 0.16, pas un
    # epsilon dégénéré), à l'arrondi float32 près.
    env0 = make_env(gbm_features, risk_aversion=0.0)
    env1 = make_env(gbm_features, risk_aversion=0.1)
    env0.reset(seed=0, options={"random_start": False})
    env1.reset(seed=0, options={"random_start": False})
    # σ̂ capturé AVANT le step : la reward le lit avant l'avancée du temps
    sigma = env1._get_recent_volatility()
    a = np.array([0.8], dtype=np.float32)
    _, r0, _, _, _ = env0.step(a)
    _, r1, _, _, _ = env1.step(a)
    attendu = 0.1 * sigma * 0.8 * env1.cfg.reward_scaling
    assert attendu > 0.05                       # le terme a une vraie magnitude
    assert abs((r0 - r1) - attendu) < 1e-5


def test_n_trades_au_seuil_delta_w(flat_data):
    env = make_env(flat_data)
    env.reset(seed=0, options={"random_start": False})
    env.step(np.array([0.5], dtype=np.float32))       # Δw = 0.5  → trade
    env.step(np.array([0.505], dtype=np.float32))     # Δw = 0.005 → pas compté
    env.step(np.array([0.6], dtype=np.float32))       # Δw = 0.095 → trade
    assert env.history["n_trades"] == 2
    assert env.history["turnover"] == pytest.approx(0.5 + 0.005 + 0.095)


def test_action_hors_borne_clippee(flat_data):
    env = make_env(flat_data)
    env.reset(seed=0, options={"random_start": False})
    env.step(np.array([3.0], dtype=np.float32))        # PPO gaussien déborde
    assert abs(float(env._position) - 1.0) < 1e-9      # clippé à +1


def test_episode_deterministe_reproductible(gbm_features):
    data = gbm_features.copy()
    seq = np.random.default_rng(7).uniform(-1, 1, 300).astype(np.float32)

    def run():
        env = make_env(data)
        env.reset(seed=5, options={"random_start": False})
        for w in seq:
            _, _, term, trunc, _ = env.step(np.array([w]))
            if term or trunc:
                break
        return np.array(env.history["portfolio_values"])

    v1, v2 = run(), run()
    assert np.array_equal(v1, v2)


def test_run_episode_evaluate_en_continu(gbm_features):
    # Le protocole d'évaluation complet (freeze_after_stop compris) doit
    # fonctionner avec une politique continue — stub model style SB3.
    from evaluate import _run_episode

    class StubModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.3], dtype=np.float32), None

    env = make_env(gbm_features)
    m = _run_episode(StubModel(), env, seed=0, random_start=False,
                     freeze_after_stop=True)
    assert m["n_steps"] > 100
    assert m["long_pct"] == 1.0                       # w = 0.3 > 0.2 partout
    assert m["short_pct"] == 0.0
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["return"])