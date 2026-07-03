# Tests d'intégration : un VRAI entraînement PPO court sur données synthétiques,
# puis évaluation avec le pipeline evaluate.py (VecNormalize inclus).
# Aucun réseau nécessaire. Durée totale ~30-60 s (marqué slow).
import os

import numpy as np
import pytest

import evaluate as ev
from environment import EnvConfig, TradingEnv
from train import TrainConfig, plot_training_curves, set_all_seeds, train

pytestmark = pytest.mark.slow

CFG_ENV = EnvConfig(
    initial_capital=10_000.0,
    transaction_cost=0.001,
    window_size=10,
    max_drawdown_pct=0.25,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, gbm_features):
    """
    Entraîne un petit PPO (24 576 steps) sur données synthétiques.
    Assez long pour déclencher une fois les callbacks (eval_freq=10 000)
    → best_model.zip + metrics_history non vides.
    """
    set_all_seeds(42)
    root = tmp_path_factory.mktemp("train_run")

    n = len(gbm_features)
    train_data = gbm_features.iloc[: int(n * 0.7)].copy()
    val_data = gbm_features.iloc[int(n * 0.7):].copy()

    cfg_train = TrainConfig(
        total_timesteps=24_576,
        n_steps=256,
        batch_size=64,
        n_envs=2,
        save_dir=str(root / "models") + os.sep,
        log_dir=str(root / "logs") + os.sep,
        model_name="ppo_test",
    )

    model, vec_env, hist = train(train_data, val_data, CFG_ENV, cfg_train)
    return {
        "model": model,
        "hist": hist,
        "save_dir": cfg_train.save_dir,
        "val_data": val_data,
    }


# ============================================================
# TRAIN
# ============================================================
class TestTraining:

    def test_artifacts_saved(self, trained):
        d = trained["save_dir"]
        assert os.path.exists(os.path.join(d, "ppo_test.zip"))
        assert os.path.exists(os.path.join(d, "vec_normalize.pkl"))
        # EvalCallback a tourné au moins une fois → meilleur modèle sauvegardé
        assert os.path.exists(os.path.join(d, "best_model.zip"))

    def test_metrics_history_populated(self, trained):
        hist = trained["hist"]
        expected_keys = {
            "timesteps", "mean_return", "sharpe",
            "max_drawdown", "n_trades", "vs_buy_hold",
        }
        assert set(hist.keys()) == expected_keys
        assert len(hist["timesteps"]) >= 1, "Le callback financier doit avoir loggé"
        lengths = {len(v) for v in hist.values()}
        assert len(lengths) == 1, "Toutes les séries de métriques doivent être alignées"

    def test_model_predicts_valid_actions(self, trained):
        env = TradingEnv(data=trained["val_data"], cfg=CFG_ENV)
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action, _ = trained["model"].predict(obs, deterministic=True)
            assert env.action_space.contains(int(action))
            obs, _, term, trunc, _ = env.step(int(action))
            if term or trunc:
                break

    def test_plot_training_curves_writes_png(self, trained, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plot_training_curves(trained["hist"], title="pytest")
        assert (tmp_path / "logs" / "training_curves_pytest.png").exists()


# ============================================================
# EVALUATE
# ============================================================
class TestEvaluation:

    def test_run_episode_metrics_coherent(self, trained):
        env = ev._make_env(trained["val_data"], CFG_ENV)
        r = ev._run_episode(trained["model"], env, seed=3)

        assert r["alpha"] == pytest.approx(r["return"] - r["bh"], abs=1e-9)
        assert 0.0 <= r["max_dd"] <= 1.0
        total_actions = r["hold_pct"] + r["long_pct"] + r["flat_pct"] + r["short_pct"]
        assert total_actions == pytest.approx(1.0)
        assert r["n_steps"] > 0

    def test_evaluate_one_with_vecnormalize(self, trained):
        """Charge le modèle + vec_normalize.pkl depuis le disque, comme en prod."""
        model_path = os.path.join(trained["save_dir"], "ppo_test.zip")
        r = ev.evaluate_one(model_path, trained["val_data"], CFG_ENV, seeds=(0, 1, 2))

        for key in ("return", "alpha", "max_dd", "beat_bh", "n_seeds", "n_trades"):
            assert key in r
        assert r["n_seeds"] == 3
        assert 0 <= r["beat_bh"] <= 3
        assert np.isfinite(r["return"])

    def test_stress_report_structure(self, trained, tmp_path):
        """Grille de frais + ablation kill-switch : structure et cohérence."""
        mp = {"tiny": os.path.join(trained["save_dir"], "ppo_test.zip")}
        r = ev.stress_report(model_paths=mp, data=trained["val_data"],
                             fee_grid=(0.0, 0.002),
                             json_path=str(tmp_path / "stress.json"))

        assert set(r["fee_grid"]["tiny"].keys()) == {"0.0000", "0.0020"}
        for cell in r["fee_grid"]["tiny"].values():
            assert np.isfinite(cell["alpha"])
        ns = r["no_killswitch"]["tiny"]
        assert ns["stop_contribution"] == pytest.approx(
            ns["alpha_with_stop"] - ns["alpha"], abs=1e-9)
        assert (tmp_path / "stress.json").exists()

    def test_evaluation_deterministic_given_seed(self, trained):
        """Même seed + politique déterministe → mêmes métriques."""
        env1 = ev._make_env(trained["val_data"], CFG_ENV)
        env2 = ev._make_env(trained["val_data"], CFG_ENV)
        r1 = ev._run_episode(trained["model"], env1, seed=7)
        r2 = ev._run_episode(trained["model"], env2, seed=7)
        assert r1 == r2

    def test_evaluate_full_covers_whole_split(self, trained):
        """L'éval full-split est déterministe et parcourt tout le split (sauf stop DD)."""
        model_path = os.path.join(trained["save_dir"], "ppo_test.zip")
        r1 = ev.evaluate_full(model_path, trained["val_data"], CFG_ENV)
        r2 = ev.evaluate_full(model_path, trained["val_data"], CFG_ENV)

        assert r1 == r2, "Full-split doit être 100% reproductible"
        expected_steps = len(trained["val_data"]) - 1 - CFG_ENV.window_size
        if not r1["terminated_early"]:
            assert r1["n_steps"] == expected_steps

    def test_evaluate_full_bh_always_on_whole_split(self, trained):
        """
        Même en cas de stop drawdown, le B&H de référence couvre TOUT le split
        (portefeuille gelé en cash après le stop) → alphas comparables entre modèles.
        """
        model_path = os.path.join(trained["save_dir"], "ppo_test.zip")
        data = trained["val_data"]
        r = ev.evaluate_full(model_path, data, CFG_ENV)

        # Dernier prix loggé par l'env = seg_end - 1 (avant-dernier jour du split)
        p = data["price"].values
        w = CFG_ENV.window_size
        expected_bh = (p[-2] - p[w]) / p[w]
        assert r["bh"] == pytest.approx(expected_bh, rel=1e-4)

    def test_overfitting_shim_delegates(self):
        """overfitting.py ne doit plus avoir sa propre logique buggée."""
        import overfitting
        assert overfitting.check_overfitting_both is ev.check_overfitting_both
        assert overfitting.compare_models is ev.compare_models


# ============================================================
# MÉTRIQUES RISK-ADJUSTED (calculs purs, pas de modèle)
# ============================================================
class TestRiskMetrics:

    def test_flat_portfolio(self):
        """Portefeuille constant : return 0, drawdown 0, pas de division par zéro."""
        portfolio = np.full(100, 10_000.0)
        prices = np.linspace(100, 120, 100)
        m = ev._metrics_from_series(portfolio, prices)

        assert m["return"] == pytest.approx(0.0)
        assert m["max_dd"] == pytest.approx(0.0)
        assert m["alpha"] == pytest.approx(-0.2)
        assert not m["beat_bh"]

    def test_monotonic_growth(self):
        """Croissance régulière : Sharpe très élevé, MaxDD nul, Calmar NaN (pas de DD)."""
        portfolio = 10_000.0 * 1.001 ** np.arange(300)
        prices = np.full(300, 100.0)
        m = ev._metrics_from_series(portfolio, prices)

        assert m["return"] > 0
        assert m["beat_bh"]
        assert m["max_dd"] == pytest.approx(0.0, abs=1e-9)
        assert m["sharpe"] > 10
        assert np.isnan(m["calmar"])  # max_dd = 0 → non défini

    def test_drawdown_and_calmar(self):
        """Pic à 12k puis chute à 9k → MaxDD 25 %, Calmar négatif (return < 0)."""
        up = np.linspace(10_000, 12_000, 50)
        down = np.linspace(12_000, 9_000, 50)
        portfolio = np.concatenate([up, down])
        prices = np.full(100, 100.0)
        m = ev._metrics_from_series(portfolio, prices)

        assert m["max_dd"] == pytest.approx(0.25, rel=1e-6)
        assert m["calmar"] < 0
        assert m["cvar_95"] < 0

    def test_cvar_is_tail_mean(self):
        """CVaR 95 = moyenne des 5 % pires returns journaliers."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.01, 500)
        portfolio = 10_000.0 * np.cumprod(1 + np.concatenate([[0.0], returns]))
        prices = np.full(501, 100.0)
        m = ev._metrics_from_series(portfolio, prices)

        r = np.diff(portfolio) / portfolio[:-1]
        var = np.quantile(r, 0.05)
        expected = r[r <= var].mean()
        assert m["cvar_95"] == pytest.approx(expected, rel=1e-6)
        assert m["cvar_95"] < var  # la moyenne de queue est pire que le quantile

    def test_sortino_ignores_upside_volatility(self):
        """Deux trajectoires même moyenne/std baissière : la vol haussière ne pénalise pas."""
        # trajectoire avec grosses hausses et petites baisses
        r_up = np.tile([0.03, -0.001], 100)
        p_up = 10_000.0 * np.cumprod(1 + np.concatenate([[0.0], r_up]))
        prices = np.full(len(p_up), 100.0)
        m = ev._metrics_from_series(p_up, prices)

        assert m["sortino"] > m["sharpe"], (
            "La vol vient surtout du upside → Sortino doit dépasser Sharpe"
        )
