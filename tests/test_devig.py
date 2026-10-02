import numpy as np
import pytest

from ratings.devig import power_devig


def test_power_devig_sums_to_one_and_shrinks_longshots_more():
    p = np.array([0.6, 0.3, 0.1, 0.05])          # sums to 1.05
    q = power_devig(p)
    assert q.sum() == pytest.approx(1.0)
    assert (q < p).all()
    assert q[3] / p[3] < q[0] / p[0]             # longshot cut proportionally more


def test_power_devig_handles_underround_and_fair_sets():
    q = power_devig([0.5, 0.3, 0.1])             # sums to 0.9
    assert q.sum() == pytest.approx(1.0) and (q > [0.5, 0.3, 0.1]).all()
    np.testing.assert_allclose(power_devig([0.25] * 4), [0.25] * 4)
