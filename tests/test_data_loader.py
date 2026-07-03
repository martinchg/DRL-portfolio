# Tests du pipeline de données : features, normalisation, split.
# Les tests réseau (yfinance) sont marqués @pytest.mark.network.
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from conftest import make_price_df

from data_loader import (
    FEATURES,
    REGIME_FEATURES,
    DataConfig,
    active_features,
    _build_features,
    _split,
    _scale_features,
    load_data,
    load_multi_ticker_data,
)


# ============================================================
# FEATURES
# ============================================================
class TestBuildFeatures:

    def test_columns_and_no_nan(self, price_df):
        data = _build_features(price_df, DataConfig())
        for feat in FEATURES:
            assert feat in data.columns, f"Feature manquante : {feat}"
        assert "price" in data.columns
        assert not data.isnull().any().any(), "NaN résiduels après dropna"

    def test_warmup_rows_dropped(self, price_df):
        data = _build_features(price_df, DataConfig())
        # Les fenêtres (vol 20 jours, RSI 14, momentum 5) créent des NaN au début
        assert len(data) < len(price_df)
        assert len(data) >= len(price_df) - 30

    def test_log_returns_definition(self, price_df):
        data = _build_features(price_df, DataConfig())
        p = data["price"].values
        expected = np.log(p[1:] / p[:-1])
        np.testing.assert_allclose(
            data["log_returns"].values[1:], expected, rtol=1e-10
        )

    def test_rsi_normalized_between_0_and_1(self, price_df):
        data = _build_features(price_df, DataConfig())
        assert data["rsi"].between(0.0, 1.0).all()

    def test_momentum_5_definition(self, price_df):
        data = _build_features(price_df, DataConfig())
        p = data["price"]
        expected = p.pct_change(5)
        np.testing.assert_allclose(
            data["momentum_5"].values[5:], expected.values[5:], rtol=1e-10
        )


# ============================================================
# FEATURES DE RÉGIME (Acte 3 — optionnelles)
# ============================================================
class TestRegimeFeatures:

    def test_off_by_default(self, price_df):
        """Le flag OFF ne change rien : les anciens modèles restent compatibles."""
        data = _build_features(price_df, DataConfig())
        assert "dist_high_252" not in data.columns
        assert "trend_200" not in data.columns
        assert active_features(DataConfig()) == FEATURES

    def test_rising_series_at_high_and_above_trend(self):
        """Série strictement croissante : toujours AU plus-haut, AU-DESSUS de la SMA200."""
        t = np.arange(900)
        df = pd.DataFrame(
            {"price": 100.0 * 1.001 ** t},
            index=pd.bdate_range("2015-01-01", periods=900),
        )
        data = _build_features(df, DataConfig(regime_features=True))

        cols = FEATURES + REGIME_FEATURES
        assert not data[cols].isnull().any().any()
        np.testing.assert_allclose(data["dist_high_252"].values, 0.0, atol=1e-12)
        assert (data["trend_200"] > 0).all()

    def test_declining_series_below_high_and_trend(self):
        """Série décroissante : sous le plus-haut 1 an ET sous la SMA200."""
        t = np.arange(900)
        df = pd.DataFrame(
            {"price": 100.0 * 0.999 ** t},
            index=pd.bdate_range("2015-01-01", periods=900),
        )
        data = _build_features(df, DataConfig(regime_features=True))

        assert (data["dist_high_252"] < 0).all()
        assert (data["trend_200"] < 0).all()

    def test_scaler_covers_regime_columns(self, price_df):
        """Les colonnes régime sont scalées comme les autres (médiane train ≈ 0)."""
        big = make_price_df(n=900, seed=3)
        cfg = DataConfig(regime_features=True)
        data = _build_features(big, cfg)
        train_end = int(len(data) * 0.7)
        scaled, _ = _scale_features(data, train_end, active_features(cfg))

        medians = scaled[REGIME_FEATURES].iloc[:train_end].median()
        np.testing.assert_allclose(medians.values, 0.0, atol=1e-8)


# ============================================================
# NORMALISATION
# ============================================================
class TestScaleFeatures:

    def test_scaler_fit_on_train_only(self, price_df):
        """Anti look-ahead bias : les stats du scaler doivent venir du train uniquement."""
        data = _build_features(price_df, DataConfig())
        train_end = int(len(data) * 0.7)

        scaled, scaler = _scale_features(data, train_end)

        ref = RobustScaler().fit(data[FEATURES].iloc[:train_end])
        np.testing.assert_allclose(scaler.center_, ref.center_, rtol=1e-10)
        np.testing.assert_allclose(scaler.scale_, ref.scale_, rtol=1e-10)

    def test_train_median_centered(self, price_df):
        data = _build_features(price_df, DataConfig())
        train_end = int(len(data) * 0.7)
        scaled, _ = _scale_features(data, train_end)

        medians = scaled[FEATURES].iloc[:train_end].median()
        np.testing.assert_allclose(medians.values, 0.0, atol=1e-8)

    def test_test_part_uses_train_stats(self, price_df):
        data = _build_features(price_df, DataConfig())
        train_end = int(len(data) * 0.7)
        scaled, _ = _scale_features(data, train_end)

        ref = RobustScaler().fit(data[FEATURES].iloc[:train_end])
        expected = ref.transform(data[FEATURES].iloc[train_end:])
        np.testing.assert_allclose(
            scaled[FEATURES].iloc[train_end:].values, expected, rtol=1e-8
        )

    def test_original_not_mutated(self, price_df):
        data = _build_features(price_df, DataConfig())
        before = data[FEATURES].copy()
        _scale_features(data, int(len(data) * 0.7))
        pd.testing.assert_frame_equal(data[FEATURES], before)


# ============================================================
# SPLIT
# ============================================================
class TestSplit:

    def test_proportions_and_no_loss(self, price_df):
        data = _build_features(price_df, DataConfig())
        cfg = DataConfig()
        train, val, test = _split(data, cfg)

        assert len(train) + len(val) + len(test) == len(data)
        assert len(train) == int(len(data) * cfg.train_ratio)

    def test_chronological_no_overlap(self, price_df):
        data = _build_features(price_df, DataConfig())
        train, val, test = _split(data, DataConfig())

        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]


# ============================================================
# INTÉGRATION RÉSEAU (yfinance)
# ============================================================
@pytest.mark.network
class TestLoadDataNetwork:

    def test_load_data_full_pipeline(self):
        cfg = DataConfig(ticker="AAPL", start_date="2018-01-01", end_date="2023-01-01")
        train, val, test, scaler = load_data(cfg)

        for split in (train, val, test):
            assert not split[FEATURES].isnull().any().any()
            assert "price" in split.columns
            assert len(split) > 50

        assert train.index[-1] < val.index[0] < test.index[0]
        # ~5 ans de bourse ≈ 1250 jours
        total = len(train) + len(val) + len(test)
        assert 1000 < total < 1400

    def test_load_multi_ticker(self):
        cfg = DataConfig(
            tickers=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2022-01-01",
        )
        train, val, test = load_multi_ticker_data(cfg)

        # segment_id sur TOUS les splits (épisodes confinés par ticker dans l'env,
        # y compris pour la validation utilisée par l'EvalCallback)
        for split in (train, val, test):
            assert "segment_id" in split.columns
            assert set(split["segment_id"].unique()) == {0, 1}
            assert "ticker" in split.columns
        assert not train[FEATURES].isnull().any().any()
