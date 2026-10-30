import numpy as np
import pandas as pd
import pytest

from proxytrust.run import run, summarise
from proxytrust.sim import draw, population_spearman, season, spearman


def test_spearman_matches_pandas():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((3, 12)), rng.standard_normal((3, 12))
    want = [pd.Series(a[i]).corr(pd.Series(b[i]), method="spearman") for i in range(3)]
    assert np.allclose(spearman(a, b), want)


def test_population_spearman():
    assert population_spearman(0.0) == 0.0
    assert population_spearman(1.0) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    q, p, lb = draw(rng, 4, 50_000, np.array([0.6]), 0.8)
    assert spearman(p[0], lb).mean() == pytest.approx(population_spearman(0.6 * 0.8), abs=0.01)


def test_shared_error_correlation():
    rhos, c = np.array([0.5, 0.8]), 0.8
    q, p, lb = draw(np.random.default_rng(2), 1, 200_000, rhos, 0.7, c)
    want = rhos[0] * rhos[1] + np.sqrt((1 - rhos[0] ** 2) * (1 - rhos[1] ** 2)) * c
    assert np.corrcoef(p[0, 0], p[1, 0])[0, 1] == pytest.approx(want, abs=0.01)
    assert np.corrcoef(p[1, 0], lb[0])[0, 1] == pytest.approx(0.8 * 0.7, abs=0.01)


def test_hit_is_undefined_for_equal_proxies():
    out = season(np.random.default_rng(3), 50, [0.7] * 4, 0.9, 10)
    assert np.isnan(out["hit"]).all()


def test_selection_is_right_with_many_clean_points():
    out = season(np.random.default_rng(4), 200, [0.2, 0.9], 0.99, 100)
    assert out["hit"].mean() > 0.98
    assert np.allclose(out["q2_select"], out["q2_oracle"])


def test_top_design_submits_better_configurations():
    rng = np.random.default_rng(5)
    rand = season(rng, 300, [0.5, 0.7], 0.8, 10, design="random")
    top = season(rng, 300, [0.5, 0.7], 0.8, 10, design="top")
    assert top["q1_mean"].mean() > rand["q1_mean"].mean() + 1.0
    with pytest.raises(ValueError):
        season(rng, 2, [0.5, 0.7], 0.8, 5, design="best")


def test_selected_correlation_is_optimistic_and_collapses():
    out = season(np.random.default_rng(6), 2000, [0.6, 0.65, 0.7, 0.75], 0.7, 10)
    assert out["rho1_sel"].mean() > population_spearman(0.75 * 0.7) + 0.05
    assert out["rho2_sel"].mean() < out["rho1_sel"].mean() - 0.3


def test_grid_and_summary():
    df = run(reps=20, n1s=[5])
    assert len(df) == 3 * 2 * 3 * 1 * 2 * 20
    s = summarise(df)
    assert len(s) == 36
    assert s.loc[s.scenario == "equal", "hit"].isna().all()
    assert np.allclose(s.regret_select, s.q2_oracle - s.q2_select)
