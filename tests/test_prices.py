import pandas as pd
import pytest

from ratings.kalshi import market_team, mid_width, tidy_markets, to_price


def test_to_price_parses_dollar_strings():
    assert to_price("0.3400") == pytest.approx(0.34)
    assert to_price(0.5) == 0.5
    assert to_price(None) is None
    assert to_price("") is None


@pytest.mark.parametrize("bid,ask,mid,width", [
    ("0.3400", "0.3500", 0.345, 0.01),
    ("0.0000", "0.0100", 0.005, 0.01),   # 1c longshot
    ("0.0000", "1.0000", 0.5, 1.0),      # empty book both sides
    ("0.5000", "0.5000", 0.5, 0.0),      # locked
    ("0.1300", "0.2200", 0.175, 0.09),
])
def test_mid_width(bid, ask, mid, width):
    m, w = mid_width(bid, ask)
    assert m == pytest.approx(mid)
    assert w == pytest.approx(width)


def test_mid_width_missing_side():
    assert mid_width(None, "0.40") == (None, None)


def test_mid_width_rejects_crossed():
    with pytest.raises(ValueError):
        mid_width("0.60", "0.55")


@pytest.mark.parametrize("kind,event,market,team", [
    ("game_winner", "KXNFLGAME-26OCT04INDWAS", "KXNFLGAME-26OCT04INDWAS-WAS", "WAS"),
    ("spread", "KXNFLSPREAD-26OCT04INDWAS", "KXNFLSPREAD-26OCT04INDWAS-IND21", "IND"),
    ("spread", "KXNFLSPREAD-26OCT04JACCIN", "KXNFLSPREAD-26OCT04JACCIN-JAC3", "JAX"),
    ("win_total", "KXNFLWINS-27LAR", "KXNFLWINS-27LAR-9", "LA"),
    ("win_total", "KXNFLWINS-27GB", "KXNFLWINS-27GB-12", "GB"),
    ("division", "KXNFLNFCWEST-27", "KXNFLNFCWEST-27-LAR", "LA"),
    ("seed", "KXNFLSEED-27AFC2", "KXNFLSEED-27AFC2-BAL", "BAL"),
    ("super_bowl", "KXSB-27", "KXSB-27-XYZ", None),
])
def test_market_team(kind, event, market, team):
    assert market_team(kind, event, market) == team


def test_tidy_markets_columns_and_math():
    snap = {"series": {"KXNFLSPREAD": {"kind": "spread", "events": [], "markets": [
        {"event_ticker": "KXNFLSPREAD-26OCT04INDWAS",
         "ticker": "KXNFLSPREAD-26OCT04INDWAS-IND4", "floor_strike": 3.5,
         "yes_bid_dollars": "0.5100", "yes_ask_dollars": "0.5200",
         "close_time": "2026-10-06T13:30:00Z"},
    ]}}}
    df = tidy_markets(snap)
    for col in ["event", "market", "team", "threshold", "yes_bid", "yes_ask",
                "mid", "width", "close_time"]:
        assert col in df.columns
    r = df.iloc[0]
    assert r.team == "IND"
    assert r.threshold == 3.5
    assert r.mid == pytest.approx(0.515)
    assert r.width == pytest.approx(0.01)
    assert r.close_time == pd.Timestamp("2026-10-06T13:30:00Z")
