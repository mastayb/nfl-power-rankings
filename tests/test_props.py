import numpy as np
from scipy.stats import gamma, poisson

from ratings.props import fit_gamma, fit_poisson, player_key


def test_player_key_team_codes():
    assert player_key("KXNFLRECYDS-26OCT18DALGB-DALJWILLIAMS33-60")[1] == "DAL"
    assert player_key("KXNFLRSHYDS-26OCT18BUFLV-BUFJCOOK4-80")[1] == "BUF"
    assert player_key("KXNFLREC-26OCT15SEADEN-SEATHORTON15-4")[1] == "SEA"
    assert player_key("KXNFLTD-26OCT18LACKC-KCKWALKER9-1")[1] == "KC"
    assert player_key("KXNFLRECYDS-26OCT18ARILAR-LARPNACUA12-90")[1] == "LAR"


def test_gamma_ladder_recovers_mean():
    a, s = 2.5, 25.0                                   # mean 62.5 yards
    th = np.array([40, 50, 60, 70, 80, 90, 100.0])
    q = gamma.sf(th - 0.5, a, scale=s)
    fa, fs, rmse = fit_gamma(th, q, np.ones_like(th))
    assert abs(fa * fs - a * s) < 0.5 and rmse < 1e-4


def test_gamma_single_rung_with_pooled_shape():
    q = gamma.sf(59.5, 2.0, scale=30.0)
    fa, fs, _ = fit_gamma(np.array([60.0]), np.array([q]), np.array([1.0]), shape=2.0)
    assert abs(fa * fs - 60.0) < 0.5


def test_poisson_ladder():
    lam = 5.3
    th = np.array([3, 4, 5, 6, 7.0])
    lf, rmse = fit_poisson(th, poisson.sf(th - 1, lam), np.ones_like(th))
    assert abs(lf - lam) < 1e-3 and rmse < 1e-6
