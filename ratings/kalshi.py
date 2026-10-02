"""Kalshi public market-data client: paging, mids, spread width, raw snapshots.

Reads need no API key. Prices come back as dollar strings ("0.3400").
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import yaml

from ratings.schedule import normalize_team

# Docs URL first; the elections host is the one public examples use.
BASE_URLS = [
    "https://external-api.kalshi.com/trade-api/v2",
    "https://api.elections.kalshi.com/trade-api/v2",
]

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "series.yaml"


def pick_base_url(timeout: float = 15.0) -> str:
    """Return the first base URL that answers GET /series with HTTP 200."""
    errors = {}
    for base in BASE_URLS:
        try:
            r = httpx.get(f"{base}/series", params={"limit": 1}, timeout=timeout)
            if r.status_code == 200 and "series" in r.json():
                return base
            errors[base] = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            errors[base] = repr(e)
    raise RuntimeError(f"No Kalshi base URL responded: {errors}")


class KalshiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = base_url or pick_base_url()
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        for attempt in range(retries + 1):
            r = self._http.get(path, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < retries:
                    time.sleep(0.5 * 2**attempt)
                    continue
            r.raise_for_status()
            return r.json()
        raise AssertionError("unreachable")

    def paginate(self, path: str, key: str, params: dict | None = None,
                 limit: int = 200) -> list[dict]:
        """Follow Kalshi's cursor until it comes back empty."""
        params = dict(params or {}, limit=limit)
        out: list[dict] = []
        while True:
            page = self.get(path, params)
            out.extend(page.get(key) or [])
            cursor = page.get("cursor")
            if not cursor:
                return out
            params["cursor"] = cursor

    def list_series(self, category: str | None = None) -> list[dict]:
        params = {"category": category} if category else {}
        return self.paginate("/series", "series", params, limit=1000)

    def get_series(self, ticker: str) -> dict:
        return self.get(f"/series/{ticker}")["series"]

    def open_events(self, series_ticker: str) -> list[dict]:
        return self.paginate("/events", "events",
                             {"series_ticker": series_ticker, "status": "open"})

    def open_markets(self, series_ticker: str) -> list[dict]:
        # /events?with_nested_markets also returns finalized markets inside an
        # open event (e.g. win-total rungs already reached), so use /markets.
        return self.paginate("/markets", "markets",
                             {"series_ticker": series_ticker, "status": "open"},
                             limit=1000)


# --- prices -----------------------------------------------------------------

def to_price(x) -> float | None:
    """Kalshi dollar string (or number) -> float in [0, 1]; None if missing."""
    if x is None or x == "":
        return None
    return float(x)


def mid_width(yes_bid, yes_ask) -> tuple[float | None, float | None]:
    """Bid/ask midpoint and spread width, both in dollars (0-1).

    An empty YES bid shows as 0 and an empty YES ask as 1, so a market with
    no quotes at all gets mid 0.5, width 1.0; downstream weights by 1/width.
    """
    bid, ask = to_price(yes_bid), to_price(yes_ask)
    if bid is None or ask is None:
        return None, None
    if ask < bid:
        raise ValueError(f"crossed market: bid {bid} > ask {ask}")
    return (bid + ask) / 2, ask - bid


# --- config + tidy table ------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def market_team(kind: str, event_ticker: str, market_ticker: str) -> str | None:
    """nflverse team code a market is about, or None if it isn't team-specific.

    win_total puts the team on the event (KXNFLWINS-27ARI-9); spread puts it on
    the market with the threshold glued on (KXNFLSPREAD-...-IND21); every other
    team market ends in the bare team code.
    """
    if kind == "win_total":
        code = re.sub(r"^\d{2}", "", event_ticker.rsplit("-", 1)[-1])
    elif kind == "spread":
        code = re.sub(r"\d+$", "", market_ticker.rsplit("-", 1)[-1])
    else:
        code = market_ticker.rsplit("-", 1)[-1]
    try:
        return normalize_team(code)
    except KeyError:
        return None


def tidy_markets(snapshot: dict) -> pd.DataFrame:
    rows = []
    for series_ticker, blob in snapshot["series"].items():
        kind = blob["kind"]
        for m in blob["markets"]:
            mid, width = mid_width(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
            rows.append({
                "series": series_ticker,
                "kind": kind,
                "event": m["event_ticker"],
                "market": m["ticker"],
                "team": market_team(kind, m["event_ticker"], m["ticker"]),
                "threshold": m.get("floor_strike"),
                "yes_bid": to_price(m.get("yes_bid_dollars")),
                "yes_ask": to_price(m.get("yes_ask_dollars")),
                "mid": mid,
                "width": width,
                "close_time": pd.Timestamp(m["close_time"]),
            })
    df = pd.DataFrame(rows)
    df["threshold"] = df["threshold"].astype("float64")
    return df


def take_snapshot(client: KalshiClient, config: dict) -> dict:
    """Pull every open event + market in the pinned series."""
    snap = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "base_url": client.base_url,
        "series": {},
    }
    for s in config["series"]:
        if not s.get("active", True):
            continue
        t = s["ticker"]
        snap["series"][t] = {
            "kind": s["kind"],
            "events": client.open_events(t),
            "markets": client.open_markets(t),
        }
    return snap


def write_snapshot(snap: dict, out_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(snap["fetched_at"]).strftime("%Y%m%dT%H%M%SZ")
    raw_path = out_dir / f"{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(snap, f)
    df = tidy_markets(snap)
    pq_path = out_dir / f"{ts}.parquet"
    df.to_parquet(pq_path, index=False)
    return raw_path, pq_path, df
