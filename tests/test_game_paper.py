from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ratings.game_paper import best_per_game, kickoff_utc, market_prob, record_clv
from ratings.ladder import MAX_M, KeyNumberModel, NormalModel
from ratings.paper import TRADE_COLS

KEY = KeyNumberModel(np.where(np.isin(np.arange(MAX_M + 2), [3, 7]), 2.0, 1.0))


@pytest.mark.parametrize("model", [NormalModel(), KEY])
def test_moneyline_probs_complement_and_pickem(model):
    h = market_prob(model, 3.0, 13.0, "game_winner", "H", "H", None)
    a = market_prob(model, 3.0, 13.0, "game_winner", "A", "H", None)
    assert h + a == pytest.approx(1.0) and h > 0.5
    assert market_prob(model, 0.0, 13.0, "game_winner", "H", "H", None) == pytest.approx(0.5, abs=1e-6)


def test_spread_rungs_consistent():
    m = NormalModel()
    # home by more than 3.5 + away by more than 3.5 + |margin| <= 3.5 == 1
    h = market_prob(m, 2.0, 13.0, "spread", "H", "H", 3.5)
    a = market_prob(m, 2.0, 13.0, "spread", "A", "H", 3.5)
    assert 0 < h + a < 1
    assert h > a
    # rung probabilities fall as the threshold rises
    assert market_prob(m, 2.0, 13.0, "spread", "H", "H", 6.5) < h


def test_kickoff_utc_converts_eastern():
    s = pd.DataFrame({"game_id": ["g"], "gameday": ["2026-10-04"], "gametime": ["13:00"]})
    assert kickoff_utc(s)["g"] == "2026-10-04T17:00+00:00"


def priced_rows():
    base = dict(event="KXNFLGAME-26OCT04INDWAS", kind="game_winner", team="WAS",
                game_id="g1", game_key="26OCT04INDWAS", kickoff="2026-10-04T13:30+00:00",
                model_p=0.5, yes_bid=0.40, yes_ask=0.41, size_yes=100.0, size_no=100.0)
    return pd.DataFrame([
        base | {"ticker": "ML-WAS", "ev_yes": 0.07, "ev_no": -0.10},
        base | {"ticker": "SP-WAS3", "kind": "spread", "ev_yes": 0.05, "ev_no": -0.05},
        base | {"ticker": "ML-X", "event": "KXNFLGAME-26OCT04KCLV", "game_key": "26OCT04KCLV",
                "ev_yes": 0.01, "ev_no": 0.02},                 # below threshold
    ])


def test_best_per_game_one_trade_and_skips_traded(tmp_path):
    tp = tmp_path / "t.csv"
    best = best_per_game(priced_rows(), trades_path=tp)
    assert list(best.ticker) == ["ML-WAS"]
    assert best.iloc[0].ev_no == -np.inf             # only the chosen side survives
    pd.DataFrame([{**{c: None for c in TRADE_COLS}, "trade_id": 1,
                   "event": "KXNFLSPREAD-26OCT04INDWAS", "status": "open"}]).to_csv(tp, index=False)
    assert best_per_game(priced_rows(), trades_path=tp).empty   # game already traded


class FakeClient:
    def __init__(self, bid, ask):
        self.bid, self.ask = bid, ask

    def get(self, path, params=None):
        return {"candlesticks": [
            {"yes_bid": {"close_dollars": "0.30"}, "yes_ask": {"close_dollars": "0.32"}},
            {"yes_bid": {"close_dollars": str(self.bid)}, "yes_ask": {"close_dollars": str(self.ask)}},
        ]}


def test_record_clv_after_kickoff_only(tmp_path):
    tp = tmp_path / "t.csv"
    rows = [{**{c: None for c in TRADE_COLS}, "trade_id": 1, "ticker": "KXNFLGAME-X-WAS",
             "side": "yes", "price": 0.40, "status": "open", "kickoff": "2026-10-04T17:00+00:00"},
            {**{c: None for c in TRADE_COLS}, "trade_id": 2, "ticker": "KXNFLGAME-Y-KC",
             "side": "no", "price": 0.55, "status": "open", "kickoff": "2026-10-09T00:15+00:00"}]
    pd.DataFrame(rows).to_csv(tp, index=False)
    done = record_clv(FakeClient(0.44, 0.46), trades_path=tp,
                      now=datetime(2026, 10, 5, tzinfo=timezone.utc))
    t = pd.read_csv(tp)
    assert done == [1]
    assert t.loc[0, "close_price"] == pytest.approx(0.45)
    assert t.loc[0, "clv"] == pytest.approx(0.05)        # bought YES at 40c, closed 45c
    assert pd.isna(t.loc[1, "clv"])                     # kickoff still in the future
