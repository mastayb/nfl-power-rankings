import pandas as pd

from ratings.display import display_ranks


def test_ties_share_rank_and_next_rank_skips():
    r = pd.Series({"LA": 5.042, "BUF": 4.994, "JAX": 4.7, "NO": -2.21, "LAC": -2.24, "MIA": -7.5})
    d = display_ranks(r)
    assert d["LA"] == d["BUF"] == "T-1"
    assert d["JAX"] == "3"
    assert d["NO"] == d["LAC"] == "T-4"
    assert d["MIA"] == "6"
