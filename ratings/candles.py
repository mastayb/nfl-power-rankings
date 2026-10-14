"""Kalshi hourly price history (candlesticks), cached per market, for backtests.

Each market's full history (open -> close, 60-minute candles) is fetched once
and cached as JSON under data/cache/candles/. quotes_at() then answers "what
were the bid and ask at time T" from the last candle that ended at or before T.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ratings.kalshi import KalshiClient

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "candles"


def ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def season_markets(client: KalshiClient, series: str, season_prefix: str) -> list[dict]:
    """All markets (open + settled) of a series whose event ticker starts with season_prefix."""
    events = []
    for status in ("open", "closed", "settled"):
        events += client.paginate("/events", "events", {"series_ticker": series, "status": status})
    tickers = sorted({e["event_ticker"] for e in events
                      if e["event_ticker"].split("-", 1)[1].startswith(season_prefix)})
    markets = []
    for ev in tickers:
        markets += client.get("/markets", {"event_ticker": ev, "limit": 1000}).get("markets", [])
    return markets


def candles(client: KalshiClient, market: dict, refresh: bool = False) -> list[dict]:
    """Hourly candles for one market (cached once the market has closed)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{market['ticker']}.json"
    closed = market.get("status") in ("closed", "settled", "finalized", "determined")
    # reuse the cache only if it was written after the market closed; a history
    # cached while the market was still trading stops early (e.g. before kickoff)
    if path.exists() and not refresh and closed and path.stat().st_mtime >= ts(market["close_time"]):
        return json.loads(path.read_text())
    series = market["ticker"].split("-")[0]
    start = ts(market["open_time"])
    end = min(ts(market["close_time"]), int(time.time()))
    out, chunk = [], 5000 * 3600            # stay under the per-request candle cap
    for s in range(start, end, chunk):
        out += client.get(f"/series/{series}/markets/{market['ticker']}/candlesticks",
                          {"start_ts": s, "end_ts": min(s + chunk, end),
                           "period_interval": 60}).get("candlesticks", [])
    path.write_text(json.dumps(out))
    return out


def quote_table(cs: list[dict]) -> pd.DataFrame:
    """end_ts, bid, ask (dollars) for candles with both sides quoted."""
    rows = []
    for c in cs:
        b = (c.get("yes_bid") or {}).get("close_dollars")
        a = (c.get("yes_ask") or {}).get("close_dollars")
        if b is not None and a is not None:
            rows.append((c["end_period_ts"], float(b), float(a)))
    return pd.DataFrame(rows, columns=["end_ts", "bid", "ask"])


def quote_at(q: pd.DataFrame, t: int) -> tuple[float, float] | None:
    """Last (bid, ask) from a candle that ended at or before unix time t."""
    if q.empty:
        return None
    i = q.end_ts.searchsorted(t, side="right") - 1
    if i < 0:
        return None
    r = q.iloc[i]
    return float(r.bid), float(r.ask)


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())
