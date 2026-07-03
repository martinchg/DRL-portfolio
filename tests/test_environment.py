# Tests de TradingEnv : conformité Gymnasium, comptabilité des trades,
# reward alpha vs B&H, gestion du risque, segments multi-ticker.
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from data_loader import FEATURES as DATA_FEATURES
from environment import FEATURES as ENV_FEATURES
from environment import EnvConfig, TradingEnv

CFG = EnvConfig(
    initial_capital=10_000.0,
    transaction_cost=0.001,
    window_size=10,
    max_drawdown_pct=0.25,
)
C = CFG.initial_capital
TC = CFG.transaction_cost


def make_env(data, cfg=CFG):
    return TradingEnv(data=data, cfg=cfg)


# ============================================================
# CONFORMITÉ
# ============================================================
class TestGymnasiumCompliance:

    def test_features_lists_in_sync(self):
        """FEATURES est dupliqué dans data_loader et environment : ils doivent rester identiques."""
        assert ENV_FEATURES == DATA_FEATURES

    def test_check_env(self, gbm_features):
        env = make_env(gbm_features)
        check_env(env, skip_render_check=True)

    def test_observation_shape_and_dtype(self, gbm_features):
        env = make_env(gbm_features)
        obs, info = env.reset(seed=42)

        expected_len = CFG.window_size * len(ENV_FEATURES) + 2
        assert obs.shape == (expected_len,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        for key in ("step", "portfolio_value", "position", "drawdown", "n_trades"):
            assert key in info

    def test_reset_seed_reproducible(self, gbm_features):
        env1 = make_env(gbm_features)
        env2 = make_env(gbm_features)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)
        assert env1._current_step == env2._current_step

    def test_action_space_modes(self, gbm_features):
        assert make_env(gbm_features).action_space.n == 4
        cfg3 = EnvConfig(n_actions=3)
        assert make_env(gbm_features, cfg3).action_space.n == 3

    def test_invalid_action_rejected(self, gbm_features):
        env = make_env(gbm_features)
        env.reset(seed=0)
        with pytest.raises(AssertionError):
            env.step(7)


# ============================================================
# VALIDATION DES DONNÉES
# ============================================================
class TestDataValidation:

    def test_missing_column_raises(self, flat_data):
        bad = flat_data.drop(columns=["rsi"])
        with pytest.raises(ValueError, match="rsi"):
            TradingEnv(data=bad, cfg=CFG)

    def test_nan_raises(self, flat_data):
        bad = flat_data.copy()
        bad.loc[50, "volatility"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            TradingEnv(data=bad, cfg=CFG)


# ============================================================
# COMPTABILITÉ DES TRADES (prix constant → calculs exacts)
# ============================================================
class TestTradeAccounting:

    def test_initial_state(self, flat_data):
        env = make_env(flat_data)
        env.reset(seed=1)
        assert env._position == 0
        assert env._portfolio_value == pytest.approx(C)

    def test_go_long_pays_fee(self, flat_data):
        env = make_env(flat_data)
        env.reset(seed=1)
        _, _, _, _, info = env.step(1)

        assert env._position == 1
        assert env._cash == pytest.approx(0.0)
        assert env._shares == pytest.approx(C * (1 - TC) / 100.0)
        assert info["portfolio_value"] == pytest.approx(C * (1 - TC), rel=1e-6)

    def test_hold_changes_nothing(self, flat_data):
        env = make_env(flat_data)
        env.reset(seed=1)
        env.step(1)
        v_before = env._portfolio_value
        env.step(0)
        assert env._portfolio_value == pytest.approx(v_before)
        assert env._position == 1

    def test_long_then_flat_round_trip(self, flat_data):
        """Aller-retour à prix constant : on ne perd que les frais (2 trades)."""
        env = make_env(flat_data)
        env.reset(seed=1)
        env.step(1)
        env.step(2)

        assert env._position == 0
        assert env._shares == pytest.approx(0.0)
        assert env._portfolio_value == pytest.approx(C * (1 - TC) ** 2, rel=1e-6)

    def test_open_short_pays_fee_like_long(self, flat_data):
        """Les frais d'ouverture du short sont déduits du portefeuille (symétrie avec le long)."""
        env = make_env(flat_data)
        env.reset(seed=1)
        env.step(3)

        assert env._position == -1
        assert env._shares < 0
        assert env._portfolio_value == pytest.approx(C * (1 - TC), rel=1e-6)

    def test_short_round_trip_symmetric_with_long(self, flat_data):
        """Aller-retour short à prix constant = mêmes frais qu'un aller-retour long."""
        env = make_env(flat_data)
        env.reset(seed=1)
        env.step(3)
        env.step(2)

        assert env._position == 0
        assert env._shares == pytest.approx(0.0)
        assert env._portfolio_value == pytest.approx(C * (1 - TC) ** 2, rel=1e-6)

    def test_reversal_long_to_short(self, flat_data):
        """Long → Short = fermeture (frais) + ouverture (frais) = 3 frais au total."""
        env = make_env(flat_data)
        env.reset(seed=1)
        env.step(1)
        trades_before = env.history["n_trades"]
        env.step(3)

        assert env._position == -1
        assert env.history["n_trades"] == trades_before + 1  # 1 action = 1 trade loggé
        assert env._portfolio_value == pytest.approx(C * (1 - TC) ** 3, rel=1e-6)


# ============================================================
# DIRECTION DES POSITIONS
# ============================================================
class TestPositionDirections:

    def test_short_profits_in_downtrend(self, declining_data):
        env = make_env(declining_data)
        env.reset(seed=2)
        env.step(3)
        v0 = env._portfolio_value
        for _ in range(5):
            _, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break
        assert env._portfolio_value > v0, "Un short doit gagner quand le prix baisse"

    def test_long_profits_in_uptrend(self, rising_data):
        env = make_env(rising_data)
        env.reset(seed=2)
        env.step(1)
        v0 = env._portfolio_value
        for _ in range(5):
            env.step(0)
        assert env._portfolio_value > v0

    def test_unrealized_pnl_sign_for_short(self, declining_data):
        env = make_env(declining_data)
        env.reset(seed=2)
        env.step(3)
        obs, _, _, _, _ = env.step(0)
        # Les 2 derniers éléments de l'obs = [position, unrealized_pnl]
        assert obs[-2] == pytest.approx(-1.0)
        assert obs[-1] > 0, "PnL latent d'un short gagnant doit être positif"


# ============================================================
# REWARD = ALPHA vs BUY & HOLD
# ============================================================
class TestReward:

    def test_flat_in_downtrend_rewarded(self, declining_data):
        """Rester cash quand le marché perd 3 %/jour → alpha +3 % → reward ≈ +3."""
        env = make_env(declining_data)
        env.reset(seed=3)
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(3.0, rel=1e-2)

    def test_flat_in_uptrend_penalized(self, rising_data):
        """Rester cash quand le marché gagne 3 %/jour → alpha -3 % → reward ≈ -3."""
        env = make_env(rising_data)
        env.reset(seed=3)
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(-3.0, rel=1e-2)

    def test_long_tracks_market_reward_near_zero(self, rising_data):
        env = make_env(rising_data)
        env.reset(seed=3)
        env.step(1)
        _, reward, _, _, _ = env.step(0)  # long installé, marché +3%
        assert abs(reward) < 0.5, "Long qui suit le marché → alpha ≈ 0"

    def test_reward_clipped(self, gbm_features):
        env = make_env(gbm_features)
        env.reset(seed=4)
        done = False
        while not done:
            action = env.action_space.sample()
            _, reward, term, trunc, _ = env.step(action)
            assert -10.0 <= reward <= 10.0
            done = term or trunc


# ============================================================
# GESTION DU RISQUE / FIN D'ÉPISODE
# ============================================================
class TestEpisodeTermination:

    def test_max_drawdown_terminates(self, declining_data):
        """Long dans un marché à -3 %/jour → drawdown > 25 % → terminated."""
        env = make_env(declining_data)
        env.reset(seed=5)
        env.step(1)

        terminated, truncated = False, False
        for _ in range(30):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        assert terminated, "L'épisode doit se terminer par drawdown max"
        assert info["drawdown"] > CFG.max_drawdown_pct

    def test_truncation_at_data_end(self, flat_data):
        """Prix constant, position flat : jamais terminated → truncated à la fin des données."""
        env = make_env(flat_data)
        env.reset(seed=6)

        terminated, truncated = False, False
        for _ in range(len(flat_data)):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        assert truncated and not terminated
        assert env._current_step == env._seg_end == len(flat_data) - 1


# ============================================================
# SEGMENTS MULTI-TICKER
# ============================================================
class TestSegments:

    SEG_BOUNDS = {0: (10, 198), 1: (210, 398)}  # (start_min, seg_end) pour 2×200 jours

    def test_episode_confined_to_one_segment(self, segment_data):
        env = make_env(segment_data)
        assert env._segments is not None and len(env._segments) == 2

        for seed in range(15):
            env.reset(seed=seed)
            start, seg_end = env._current_step, env._seg_end
            ok = any(
                lo <= start and seg_end == end
                for lo, end in self.SEG_BOUNDS.values()
            )
            assert ok, f"Départ {start} / fin {seg_end} hors segment"

    def test_truncates_at_segment_boundary(self, segment_data):
        env = make_env(segment_data)
        env.reset(seed=0)
        expected_end = env._seg_end

        truncated = False
        for _ in range(len(segment_data)):
            _, _, term, truncated, _ = env.step(0)
            if term or truncated:
                break

        assert truncated
        assert env._current_step == expected_end, "L'épisode ne doit pas déborder sur l'autre ticker"

    def test_no_segments_without_column(self, flat_data):
        env = make_env(flat_data)
        assert env._segments is None


# ============================================================
# FEATURES CUSTOM (Acte 3 — observation élargie au régime)
# ============================================================
class TestCustomFeatures:

    def test_env_with_regime_features_obs_72(self):
        from conftest import make_manual_features
        data = make_manual_features(np.full(300, 100.0))
        data["dist_high_252"] = 0.0
        data["trend_200"] = 0.0

        feats = tuple(ENV_FEATURES + ["dist_high_252", "trend_200"])
        cfg = EnvConfig(transaction_cost=0.001, window_size=10, features=feats)
        env = TradingEnv(data=data, cfg=cfg)
        obs, _ = env.reset(seed=0)

        assert obs.shape == (10 * 7 + 2,)
        assert env.observation_space.shape == (72,)
        assert np.all(np.isfinite(obs))

    def test_missing_custom_feature_raises(self):
        from conftest import make_manual_features
        data = make_manual_features(np.full(300, 100.0))
        cfg = EnvConfig(features=tuple(ENV_FEATURES + ["dist_high_252"]))
        with pytest.raises(ValueError, match="dist_high_252"):
            TradingEnv(data=data, cfg=cfg)

    def test_default_unchanged(self, gbm_features):
        """Sans features custom, rien ne change (obs 52) — anciens modèles OK."""
        env = make_env(gbm_features)
        assert env.observation_space.shape == (52,)


# ============================================================
# RESET DÉTERMINISTE (évaluation full-split)
# ============================================================
class TestDeterministicReset:

    def test_starts_at_window_size(self, flat_data):
        env = make_env(flat_data)
        env.reset(seed=9, options={"random_start": False})
        assert env._current_step == CFG.window_size
        assert env._seg_end == len(flat_data) - 1

    def test_covers_full_split(self, flat_data):
        """L'épisode déterministe parcourt tout le split (aucune journée sautée)."""
        env = make_env(flat_data)
        env.reset(seed=9, options={"random_start": False})

        steps = 0
        for _ in range(len(flat_data) + 1):
            _, _, term, trunc, _ = env.step(0)
            steps += 1
            if term or trunc:
                break

        assert trunc
        assert steps == len(flat_data) - 1 - CFG.window_size

    def test_reproducible_across_seeds(self, gbm_features):
        """Sans départ aléatoire, la seed ne change rien : mêmes observations."""
        env1 = make_env(gbm_features)
        env2 = make_env(gbm_features)
        obs1, _ = env1.reset(seed=1, options={"random_start": False})
        obs2, _ = env2.reset(seed=999, options={"random_start": False})
        np.testing.assert_array_equal(obs1, obs2)

    def test_segment_data_starts_first_segment(self, segment_data):
        env = make_env(segment_data)
        env.reset(seed=3, options={"random_start": False})
        assert env._current_step == 10   # début segment 0 + window_size
        assert env._seg_end == 198

    def test_short_data_fallback_no_crash(self):
        """Données trop courtes pour un départ aléatoire → départ fixe sans crash."""
        from conftest import make_manual_features
        short = make_manual_features(np.full(115, 100.0))
        env = make_env(short)
        obs, _ = env.reset(seed=0)   # random_start=True par défaut
        assert env._current_step == CFG.window_size
        assert np.all(np.isfinite(obs))
