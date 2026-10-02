"""Paper-trading log for Kalshi futures, judged by a futures-blind model.

Each day:
  - every futures market (division, #1 seed, conference, Super Bowl, seeds 2-7,
    playoffs) is priced by the model; buying YES at the ask or NO at the bid
    gets an expected value per contract after fees,
  - a paper trade opens when EV >= MIN_EDGE and the order book shows at least
    one contract at that price (one open trade per market and side),
  - every open trade is marked to today's prices and model probability,
  - trades whose market has settled are closed with their realised result.

Fees: Kalshi's published taker formula, ceil(0.07 * P * (1 - P)) to the cent
per contract. Assumed for these series; check against real fills before
trusting P&L to the cent.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FEE_RATE = 0.07
MIN_EDGE = 0.02
PAPER_DIR = Path(__file__).resolve().parent.parent / "data" / "paper"
TRADES = PAPER_DIR / "trades.csv"
MARKS = PAPER_DIR / "marks.csv"

MODEL_COL = {"division": "division", "one_seed": "seed_1", "conference": "conf",
             "super_bowl": "sb", "playoff": "playoffs"}   # "seed": seed_<k> from the event

TRADE_COLS = ["trade_id", "opened_at", "ticker", "event", "kind", "team", "side", "price",
              "fee", "model_p", "ev", "size_at_price", "status", "closed_at", "result", "pnl"]
MARK_COLS = ["marked_at", "trade_id", "ticker", "side", "yes_bid", "yes_ask", "side_mid",
             "model_p", "mark_pnl"]


def fee(price: float) -> float:
    return math.ceil(round(FEE_RATE * price * (1 - price) * 100, 6)) / 100


def model_column(kind: str, event: str) -> str | None:
    if kind == "seed":
        return f"seed_{event[-1]}"     # KXNFLSEED-27AFC2 -> seed_2
    return MODEL_COL.get(kind)


def price_markets(tidy: pd.DataFrame, raw_markets: dict, sim: pd.DataFrame) -> pd.DataFrame:
    """Model probability, prices, depth and EV for both sides of every futures market."""
    rows = []
    for r in tidy[tidy.team.notna()].itertuples():
        col = model_column(r.kind, r.event)
        if col is None or col not in sim.columns or r.team not in sim.index:
            continue
        m = raw_markets.get(r.market, {})
        p = float(sim.loc[r.team, col])
        no_price = round(1 - r.yes_bid, 4)
        rows.append({
            "ticker": r.market, "event": r.event, "kind": r.kind, "team": r.team,
            "model_p": p, "yes_bid": r.yes_bid, "yes_ask": r.yes_ask,
            "ev_yes": p - r.yes_ask - fee(r.yes_ask),
            "ev_no": (1 - p) - no_price - fee(no_price),
            # buying YES lifts the YES ask; buying NO hits the YES bid
            "size_yes": float(m.get("yes_ask_size_fp") or 0),
            "size_no": float(m.get("yes_bid_size_fp") or 0),
        })
    return pd.DataFrame(rows)


TEXT_COLS = {"opened_at": object, "closed_at": object, "result": object, "status": object,
             "marked_at": object}


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, dtype={c: t for c, t in TEXT_COLS.items() if c in cols})


def update(priced: pd.DataFrame, settled: dict[str, str] | None = None,
           now: str | None = None, trades_path: Path = TRADES,
           marks_path: Path = MARKS) -> dict:
    """Open new trades, mark open ones, close settled ones. Returns a summary.

    settled: ticker -> "yes" / "no" for markets that have resolved.
    """
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    trades = _read(trades_path, TRADE_COLS)
    marks = _read(marks_path, MARK_COLS)
    settled = settled or {}
    by_ticker = priced.set_index("ticker") if len(priced) else priced

    # close settled
    closed = []
    for i, t in trades[trades.status == "open"].iterrows():
        if t.ticker in settled:
            won = (settled[t.ticker] == "yes") == (t.side == "yes")
            trades.loc[i, ["status", "closed_at", "result"]] = ["settled", now, settled[t.ticker]]
            trades.loc[i, "pnl"] = (1.0 if won else 0.0) - t.price - t.fee
            closed.append(t.trade_id)

    # open new
    opened = []
    open_keys = set(zip(trades[trades.status == "open"].ticker, trades[trades.status == "open"].side))
    next_id = int(trades.trade_id.max()) + 1 if len(trades) else 1
    for r in priced.itertuples():
        for side in ("yes", "no"):
            ev = r.ev_yes if side == "yes" else r.ev_no
            size = r.size_yes if side == "yes" else r.size_no
            if ev < MIN_EDGE or size < 1 or (r.ticker, side) in open_keys:
                continue
            price = r.yes_ask if side == "yes" else round(1 - r.yes_bid, 4)
            trades.loc[len(trades)] = {
                "trade_id": next_id, "opened_at": now, "ticker": r.ticker, "event": r.event,
                "kind": r.kind, "team": r.team, "side": side, "price": price, "fee": fee(price),
                "model_p": r.model_p if side == "yes" else 1 - r.model_p, "ev": ev,
                "size_at_price": size, "status": "open", "closed_at": None, "result": None,
                "pnl": None}
            opened.append(next_id)
            next_id += 1

    # mark open trades
    new_marks = []
    for t in trades[trades.status == "open"].itertuples():
        if t.ticker not in by_ticker.index:
            continue
        q = by_ticker.loc[t.ticker]
        yes_mid = (q.yes_bid + q.yes_ask) / 2
        side_mid = yes_mid if t.side == "yes" else 1 - yes_mid
        p = q.model_p if t.side == "yes" else 1 - q.model_p
        new_marks.append({"marked_at": now, "trade_id": t.trade_id, "ticker": t.ticker,
                          "side": t.side, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
                          "side_mid": side_mid, "model_p": p,
                          "mark_pnl": side_mid - t.price - t.fee})
    if new_marks:
        marks = pd.concat([marks, pd.DataFrame(new_marks)], ignore_index=True)

    trades.to_csv(trades_path, index=False)
    marks.to_csv(marks_path, index=False)
    return {"opened": opened, "closed": closed, "open": int((trades.status == "open").sum()),
            "marked": len(new_marks)}
