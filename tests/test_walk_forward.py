# Tests du walk-forward : découpe par dates, anti look-ahead par fold,
# et une exécution complète tiny sur données synthétiques (sans réseau).
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from conftest import make_price_df
from data_loader import FEATURES, DataConfig, _build_features
from walk_forward import (
    Fold,
    WalkForwardConfig,
    build_fold_datasets,
    make_folds,
    run_walk_forward,
    split_fold_data,
)

FOLD_2018 = Fold(test_year=2018, train_start="2015-01-01",
                 test_start="2018-01-01", test_end="2019-01-01")


@pytest.fixture(scope="module")
def features_2015_2020():
    """5 ans de features synthétiques (2015 → fin 2019)."""
    return _build_features(make_price_df(n=1300, seed=7), DataConfig())


# ============================================================
# DÉCOUPE
# ============================================================
class TestFolds:

    def test_make_folds_anchored_expanding(self):
        cfg = WalkForwardConfig(test_years=[2018, 2019, 2020], start_date="2010-01-01")
        folds = make_folds(cfg)

        assert [f.test_year for f in folds] == [2018, 2019, 2020]
        # Anchored : le train commence toujours à la même date
        assert all(f.train_start == "2010-01-01" for f in folds)
        # Chaque test couvre exactement une année, consécutives
        for f in folds:
            assert f.test_start == f"{f.test_year}-01-01"
            assert f.test_end == f"{f.test_year + 1}-01-01"

    def test_split_respects_date_boundaries(self, features_2015_2020):
        train, val, test = split_fold_data(features_2015_2020, FOLD_2018, val_ratio=0.15)

        ts = pd.Timestamp(FOLD_2018.test_start)
        te = pd.Timestamp(FOLD_2018.test_end)
        assert train.index.max() < ts
        assert val.index.max() < ts
        assert train.index[-1] < val.index[0], "train avant val (chronologique)"
        assert test.index.min() >= ts
        assert test.index.max() < te
        # ~15 % du pré-test en validation
        ratio = len(val) / (len(train) + len(val))
        assert abs(ratio - 0.15) < 0.01

    def test_scaler_fitted_on_fold_train_only(self, features_2015_2020):
        """Anti look-ahead PAR FOLD : le scaler ne voit ni la val ni l'année de test."""
        data = features_2015_2020
        _, _, test = split_fold_data(data, FOLD_2018, val_ratio=0.15)

        pre = data.loc[data.index < FOLD_2018.test_start]
        train_end = int(len(pre) * 0.85)
        ref = RobustScaler().fit(pre[FEATURES].iloc[:train_end])

        raw_test = data.loc[(data.index >= FOLD_2018.test_start)
                            & (data.index < FOLD_2018.test_end)]
        expected = ref.transform(raw_test[FEATURES])
        np.testing.assert_allclose(test[FEATURES].values, expected, rtol=1e-8)

    def test_build_fold_datasets_segments_and_no_nan(self):
        raw = {"AAA": make_price_df(1300, seed=1), "BBB": make_price_df(1300, seed=2)}
        train, val, tests = build_fold_datasets(raw, FOLD_2018, 0.15, DataConfig())

        for split in (train, val):
            assert set(split["segment_id"].unique()) == {0, 1}
            assert not split[FEATURES].isnull().any().any()
        assert set(tests.keys()) == {"AAA", "BBB"}
        for t in tests.values():
            assert not t[FEATURES].isnull().any().any()
            assert len(t) > 200   # ~1 an de bourse


# ============================================================
# INTÉGRATION TINY (entraînement réel, données synthétiques)
# ============================================================
@pytest.mark.slow
def test_run_walk_forward_tiny(tmp_path):
    raw = {"AAA": make_price_df(1300, seed=1), "BBB": make_price_df(1300, seed=2)}
    cfg = WalkForwardConfig(
        tickers    = ["AAA", "BBB"],
        start_date = "2015-01-01",
        test_years = [2018, 2019],
        timesteps  = 4096,
        out_dir    = str(tmp_path / "wf"),
        json_path  = str(tmp_path / "walk_forward.json"),
    )
    res = run_walk_forward(cfg, raw_data=raw)

    assert os.path.exists(cfg.json_path)
    assert len(res["folds"]) == 2
    for f in res["folds"]:
        assert set(f["per_ticker"].keys()) == {"AAA", "BBB"}
        for m in f["per_ticker"].values():
            assert np.isfinite(m["alpha"])
            assert m["alpha"] == pytest.approx(m["return"] - m["bh"], abs=1e-9)

    agg = res["aggregate"]
    assert agg["n_cells"] == 4
    assert 0.0 <= agg["pct_positive"] <= 1.0
    assert agg["worst_cell_alpha"] <= agg["median_alpha"] <= agg["best_cell_alpha"]
    # Modèle du fold sauvegardé (tiny → pas de best_model, fallback ppo_wf.zip)
    assert os.path.exists(os.path.join(cfg.out_dir, "fold_2018", "ppo_wf.zip"))
