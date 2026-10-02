import numpy as np
import pandas as pd
import pytest

from ratings.ladder import (GRID, MAX_M, KeyNumberModel, NormalModel, estimate_key_weights,
                            fit_game, fit_pooled_sigma, game_observations)

FLAT = KeyNumberModel(np.ones(MAX_M + 2))
KEYED = KeyNumberModel(np.where(np.isin(np.arange(MAX_M + 2), [3, 7]), 2.0, 1.0))
LADDER = np.array([-20.5, -13.5, -7.5, -3.5, -1.5, 0.0, 1.5, 2.5, 3.5, 6.5, 7.5, 10.5, 14.5, 20.5])


def synthetic_obs(model, mu, sigma, x=LADDER):
    return pd.DataFrame({"market": [f"m{i}" for i in range(len(x))], "x": x,
                         "q": model.G(x, mu, sigma), "source": "spread", "weight": 1.0})


def test_normal_G_symmetry():
    m = NormalModel()
    assert m.G(np.array([0.0]), 0.0, 13.0)[0] == pytest.approx(0.5)
    assert m.G(np.array([3.5]), 3.5, 13.0)[0] == pytest.approx(0.5)


def test_key_number_pmf_sums_to_one_and_counts_push_half():
    p = KEYED.pmf(0.0, 13.0)
    assert p.sum() == pytest.approx(1.0)
    # pick'em, symmetric: G(0) = P(M>0) + P(M=0)/2 = 0.5 exactly
    assert KEYED.G(np.array([0.0]), 0.0, 13.0)[0] == pytest.approx(0.5)
    # half-point thresholds have no push
    g = KEYED.G(np.array([2.5, 3.0, 3.5]), 0.0, 13.0)
    p3 = p[GRID == 3][0]
    assert g[0] - g[2] == pytest.approx(p3)
    assert g[1] == pytest.approx(g[2] + p3 / 2)


def test_keyed_weights_put_extra_mass_on_3():
    p_flat = FLAT.pmf(3.0, 13.0)
    p_key = KEYED.pmf(3.0, 13.0)
    assert p_key[GRID == 3][0] > 1.8 * p_flat[GRID == 3][0]


def test_pmf_vectorised_over_mu():
    rows = KEYED.pmf(np.array([-3.0, 0.0, 7.0]), 13.0)
    assert rows.shape == (3, len(GRID))
    np.testing.assert_allclose(rows[2], KEYED.pmf(7.0, 13.0))


@pytest.mark.parametrize("model", [NormalModel(), KEYED])
@pytest.mark.parametrize("mu,sigma", [(3.0, 13.0), (-6.5, 12.0), (10.0, 14.5)])
def test_fit_game_recovers_mu_sigma(model, mu, sigma):
    fit = fit_game(model, synthetic_obs(model, mu, sigma))
    assert fit.mu == pytest.approx(mu, abs=1e-3)
    assert fit.sigma == pytest.approx(sigma, abs=1e-3)
    assert fit.rmse_cents < 1e-3


def test_fit_game_moneyline_only_needs_sigma():
    obs = synthetic_obs(NormalModel(), 4.0, 13.0, x=np.array([0.0]))
    with pytest.raises(ValueError):
        fit_game(NormalModel(), obs)
    assert fit_game(NormalModel(), obs, sigma=13.0).mu == pytest.approx(4.0, abs=1e-3)


def test_pooled_sigma():
    games = {k: synthetic_obs(KEYED, mu, 13.0) for k, mu in [("a", 2.0), ("b", -5.0), ("c", 9.5)]}
    sigma, fits = fit_pooled_sigma(KEYED, games)
    assert sigma == pytest.approx(13.0, abs=1e-3)
    assert fits["b"].mu == pytest.approx(-5.0, abs=1e-3)


def test_game_observations_orientation():
    rows = pd.DataFrame([
        dict(market="ML-WAS", kind="game_winner", team="WAS", threshold=np.nan, mid=0.345, width=0.01),
        dict(market="ML-IND", kind="game_winner", team="IND", threshold=np.nan, mid=0.655, width=0.01),
        dict(market="SP-IND4", kind="spread", team="IND", threshold=3.5, mid=0.515, width=0.01),
        dict(market="SP-WAS21", kind="spread", team="WAS", threshold=20.5, mid=0.015, width=0.01),
        dict(market="SP-WAS7", kind="spread", team="WAS", threshold=6.5, mid=0.185, width=0.03),
    ])
    obs = game_observations(rows, home="WAS", away="IND").set_index("market")
    assert obs.loc["ML-WAS", "x"] == 0 and obs.loc["ML-WAS", "q"] == pytest.approx(0.345)
    assert obs.loc["ML-IND", "x"] == 0 and obs.loc["ML-IND", "q"] == pytest.approx(0.345)
    assert obs.loc["SP-IND4", "x"] == -3.5 and obs.loc["SP-IND4", "q"] == pytest.approx(0.485)
    assert obs.loc["SP-WAS7", "x"] == 6.5
    assert obs.loc["SP-WAS7", "weight"] == pytest.approx(1 / 9)
    assert "SP-WAS21" not in obs.index  # 1.5c quote dropped


def test_estimate_key_weights_recovers_planted_weights():
    rng = np.random.default_rng(0)
    spreads = rng.choice(np.arange(-10, 10.5, 0.5), size=40_000)
    p = KEYED.pmf(spreads, 13.0)
    cdf = p.cumsum(axis=1)
    idx = (cdf < rng.random(len(spreads))[:, None]).sum(axis=1)
    results = GRID[idx]
    w, sigma = estimate_key_weights(results, spreads)
    assert sigma == pytest.approx(13.0, abs=0.3)
    ratio = w[3] / np.median(w[[1, 2, 4, 5, 6, 8]])
    assert ratio == pytest.approx(2.0, rel=0.1)


def test_mean_and_median_line():
    n = NormalModel()
    assert n.mean(3.2, 13.0) == n.median_line(3.2, 13.0) == 3.2
    # flat weights: discrete with continuity buckets, mean and median ~ mu
    assert FLAT.mean(4.0, 13.0) == pytest.approx(4.0, abs=1e-6)
    assert FLAT.median_line(4.0, 13.0) == pytest.approx(4.0, abs=0.05)
    # symmetric around 0 -> pick'em
    assert KEYED.median_line(0.0, 13.0) == pytest.approx(0.0, abs=1e-9)
    # heavy mass on 3 pulls the fair line toward 3 from either side
    assert 3.0 <= KEYED.median_line(3.6, 13.0) < 3.6
    assert 2.4 < KEYED.median_line(2.4, 13.0) <= 3.0
