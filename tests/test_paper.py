import pandas as pd
import pytest

from ratings.paper import fee, model_column, price_markets, update


def test_fee_formula_rounds_up_to_cent():
    assert fee(0.50) == pytest.approx(0.02)     # 0.0175 -> 0.02
    assert fee(0.71) == pytest.approx(0.02)     # 0.0144 -> 0.02
    assert fee(0.05) == pytest.approx(0.01)     # 0.0033 -> 0.01
    assert fee(1.0) == 0


def test_model_column():
    assert model_column("seed", "KXNFLSEED-27NFC5") == "seed_5"
    assert model_column("one_seed", "KXNFL1SEED-NFC26") == "seed_1"
    assert model_column("spread", "x") is None


def priced(yes_bid=0.29, yes_ask=0.30, p=0.215, size_no=10):
    tidy = pd.DataFrame([{"market": "SF", "event": "KXNFL1SEED-NFC26", "kind": "one_seed",
                          "team": "SF", "yes_bid": yes_bid, "yes_ask": yes_ask}])
    raw = {"SF": {"yes_bid_size_fp": str(size_no), "yes_ask_size_fp": "500"}}
    sim = pd.DataFrame({"seed_1": [p]}, index=["SF"])
    return price_markets(tidy, raw, sim)


def test_price_markets_ev_both_sides():
    r = priced().iloc[0]
    assert r.ev_no == pytest.approx((1 - 0.215) - 0.71 - 0.02)
    assert r.ev_yes == pytest.approx(0.215 - 0.30 - 0.02)
    assert r.size_no == 10


def test_update_opens_marks_and_settles(tmp_path):
    tp, mp = tmp_path / "t.csv", tmp_path / "m.csv"
    s = update(priced(), now="d1", trades_path=tp, marks_path=mp)
    assert s["opened"] == [1] and s["open"] == 1
    t = pd.read_csv(tp).iloc[0]
    assert t.side == "no" and t.price == pytest.approx(0.71) and t.model_p == pytest.approx(0.785)
    # same market next day: no duplicate, but a mark
    s = update(priced(0.25, 0.26), now="d2", trades_path=tp, marks_path=mp)
    assert s["opened"] == [] and s["marked"] == 1
    m = pd.read_csv(mp)
    assert m.iloc[-1].side_mid == pytest.approx(1 - 0.255)
    assert m.iloc[-1].mark_pnl == pytest.approx(0.745 - 0.71 - 0.02)
    # settles NO -> the NO trade wins $1
    s = update(priced().iloc[0:0], settled={"SF": "no"}, now="d3", trades_path=tp, marks_path=mp)
    t = pd.read_csv(tp).iloc[0]
    assert s["closed"] == [1] and t.status == "settled"
    assert t.pnl == pytest.approx(1 - 0.71 - 0.02)


def test_no_trade_without_depth_or_edge(tmp_path):
    tp, mp = tmp_path / "t.csv", tmp_path / "m.csv"
    assert update(priced(size_no=0), trades_path=tp, marks_path=mp)["opened"] == []
    assert update(priced(p=0.28), trades_path=tp, marks_path=mp)["opened"] == []
