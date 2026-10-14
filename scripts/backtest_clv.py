"""Replay the live game-line paper strategy on 2026 history; score closing line value.

Run: python -u scripts/backtest_clv.py [--fetch-only]

For every day at 13:30 UTC (the daily job's time) from the first regular-season
listing to today, rebuild what was knowable then and run the live code:
  - Kalshi game + spread quotes at that hour (last hourly candle before T)
  - win-total ladder quotes at that hour; records from games finished before T
  - nflverse closing lines and results only for games finished before T
  -> Stage A, leave-one-out ratings, pricing (ratings/game_paper.py)
Each game is traded once: the first day its best side clears 3c after fees.
CLV = last pre-kickoff mid on the traded side - entry price.

Caveat: candles carry no order-book depth, so every quote is assumed fillable.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.candles import candles, now_ts, quote_at, quote_table, season_markets, ts  # noqa: E402
from ratings.game_paper import (MIN_EDGE_GAME, kickoff_utc, loo_margins, price_games,  # noqa: E402
                                stage_a_frame)
from ratings.kalshi import KalshiClient, market_team, mid_width  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.schedule import game_key, join_games, load_schedule, schedule_as_of  # noqa: E402

SEASON = 2026
OUT = ROOT / "data" / "backtest_clv.parquet"
DECISION_HOUR_UTC = 13.5
BIG = {"yes_ask_size_fp": 1e9, "yes_bid_size_fp": 1e9}   # depth unknown in candles


def discover(client, sched):
    """Regular-season game/spread markets with kickoff in the past, plus win-total rungs."""
    kick = kickoff_utc(sched)
    games = []
    for series, kind in [("KXNFLGAME", "game_winner"), ("KXNFLSPREAD", "spread")]:
        for m in season_markets(client, series, str(SEASON)[2:]):
            games.append({**m, "kind": kind})
    gm = pd.DataFrame(games)
    gm["key"] = gm.event_ticker.map(game_key)
    ml_events = sorted(gm.loc[gm.kind == "game_winner", "event_ticker"].unique())
    joined, _ = join_games(ml_events, sched)
    joined["key"] = joined.event.map(game_key)
    joined["kickoff"] = joined.game_id.map(kick)
    joined = joined[joined.kickoff.notna()
                    & (joined.kickoff.map(lambda k: ts(k)) < now_ts())]
    gm = gm[gm.key.isin(joined.key)]
    wins = pd.DataFrame(season_markets(client, "KXNFLWINS", str(SEASON + 1)[2:])).assign(kind="win_total")
    return gm, joined, wins


def fetch_all(client, frames):
    out, n, t0 = {}, sum(len(f) for f in frames), time.time()
    for f in frames:
        for m in f.to_dict("records"):
            out[m["ticker"]] = quote_table(candles(client, m))
            if len(out) % 250 == 0:
                print(f"  candles {len(out)}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def tidy_at(markets: pd.DataFrame, quotes: dict, t: int) -> pd.DataFrame:
    rows = []
    for m in markets.itertuples():
        if ts(m.open_time) > t:
            continue
        q = quote_at(quotes[m.ticker], t)
        if q is None:
            continue
        bid, ask = q
        mid, width = mid_width(bid, ask)
        rows.append({"series": m.ticker.split("-")[0], "kind": m.kind, "event": m.event_ticker,
                     "market": m.ticker, "team": market_team(m.kind, m.event_ticker, m.ticker),
                     "threshold": m.floor_strike if pd.notna(getattr(m, "floor_strike", np.nan)) else np.nan,
                     "yes_bid": bid, "yes_ask": ask, "mid": mid, "width": width})
    return pd.DataFrame(rows)


def main(fetch_only: bool = False) -> None:
    sched = load_schedule(SEASON)
    kick = kickoff_utc(sched)
    model = load_key_weights()
    lam_w = yaml.safe_load(open(ROOT / "config" / "win_totals.yaml"))["lambda"]

    with KalshiClient() as client:
        print("discovering markets...", flush=True)
        gm, joined, wins = discover(client, sched)
        print(f"{joined.game_id.nunique()} completed regular-season games, {len(gm)} game/spread "
              f"markets, {len(wins)} win-total rungs", flush=True)
        quotes = fetch_all(client, [gm, wins])
    if fetch_only:
        return
    result = {m.ticker: m.result for m in gm.itertuples()}

    first = min(ts(o) for o in gm.open_time)
    day0 = datetime.fromtimestamp(first, timezone.utc).replace(hour=13, minute=30, second=0)
    if day0.timestamp() < first:
        day0 += timedelta(days=1)
    decisions = []
    d = day0
    while d.timestamp() < now_ts():
        decisions.append(int(d.timestamp()))
        d += timedelta(days=1)

    rows = []
    for i, t in enumerate(decisions):
        open_games = joined[joined.kickoff.map(ts) > t]
        live = gm[gm.key.isin(open_games.key)]
        tg = tidy_at(live, quotes, t)
        if tg.empty or "game_winner" not in set(tg.kind):
            continue
        ml_events = sorted(tg.loc[tg.kind == "game_winner", "event"].unique())
        games = open_games[open_games.event.isin(ml_events)]
        if games.empty:
            continue
        s_t = schedule_as_of(sched, kick, t)
        played_wins = s_t[s_t.result.notna()]
        wins_now = pd.concat([played_wins.loc[played_wins.result > 0, "home_team"],
                              played_wins.loc[played_wins.result < 0, "away_team"]]).value_counts()
        wt = tidy_at(wins, quotes, t)
        if not wt.empty:
            wt = wt[wt.threshold > wt.team.map(wins_now).fillna(0)]
        stage_a, sigma = stage_a_frame(tg, games, model)
        if stage_a.empty:
            continue
        stamp = datetime.fromtimestamp(t, timezone.utc)
        margins = loo_margins(stage_a, games, s_t, model, sigma, stamp, wt, lam_w)
        raw = {m: BIG for m in tg.market}
        priced = price_games(tg, raw, games, margins, stage_a.set_index("game_id")["mean"],
                             model, sigma, kick)
        if priced.empty:
            continue
        priced["best_side"] = np.where(priced.ev_yes >= priced.ev_no, "yes", "no")
        priced["best_ev"] = np.maximum(priced.ev_yes, priced.ev_no)
        best = priced.loc[priced.groupby("game_key").best_ev.idxmax()]
        for r in best.itertuples():
            q = quote_at(quotes[r.ticker], ts(r.kickoff))
            close_mid = (q[0] + q[1]) / 2 if q else np.nan
            price = r.yes_ask if r.best_side == "yes" else 1 - r.yes_bid
            close_side = close_mid if r.best_side == "yes" else 1 - close_mid
            res = result.get(r.ticker)
            won = None if res not in ("yes", "no") else (res == r.best_side)
            rows.append({"t": stamp, "game_key": r.game_key, "game_id": r.game_id,
                         "ticker": r.ticker, "kind": r.kind, "side": r.best_side,
                         "ev": r.best_ev, "price": price, "model_p": r.model_p,
                         "mu_model": r.mu_model, "mu_market": r.mu_market, "kickoff": r.kickoff,
                         "hours_to_kick": (ts(r.kickoff) - t) / 3600,
                         "close_side": close_side, "clv": close_side - price,
                         "won": won, "sigma": sigma})
        print(f"[{stamp:%m-%d}] {i + 1}/{len(decisions)}: {len(games)} games open, "
              f"{(best.best_ev >= MIN_EDGE_GAME).sum()} over threshold", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)} ({len(df)} game-day rows)")
    report(df)


def report(df: pd.DataFrame) -> None:
    from ratings.paper import fee

    sig = df[df.ev >= MIN_EDGE_GAME].sort_values("t")
    trades = sig.groupby("game_key").head(1).copy()          # live rule: first signal per game
    trades["fee"] = trades.price.map(fee)
    trades["pnl"] = np.where(trades.won.astype(bool), 1.0, 0.0) - trades.price - trades.fee
    n = len(trades)
    se = trades.clv.std() / np.sqrt(n)
    print(f"\n=== Live rule replay: first day each game clears {MIN_EDGE_GAME:.0%} ===")
    print(f"trades {n} of {df.game_key.nunique()} games")
    print(f"CLV: mean {trades.clv.mean() * 100:+.2f}c (se {se * 100:.2f}c), median "
          f"{trades.clv.median() * 100:+.2f}c, positive {(trades.clv > 0).sum()}/{n}; "
          f"after entry fee {(trades.clv - trades.fee).mean() * 100:+.2f}c")
    settled = trades[trades.won.notna()]
    print(f"record {settled.won.astype(bool).sum()}-{(~settled.won.astype(bool)).sum()}, P&L "
          f"${settled.pnl.sum():+.2f} on ${(settled.price + settled.fee).sum():.2f}; model expected "
          f"${settled.ev.sum():+.2f}")
    trades["ev_bucket"] = pd.cut(trades.ev, [0, 0.04, 0.06, 1], labels=["3-4c", "4-6c", "6c+"])
    trades["lead"] = pd.cut(trades.hours_to_kick, [0, 48, 120, 1e4], labels=["<2d", "2-5d", ">5d"])
    for col in ["ev_bucket", "lead", "kind"]:
        g = trades.groupby(col, observed=True).clv
        print(f"\nCLV by {col}:\n" + (g.agg(["size", "mean"]).assign(mean=lambda x: x["mean"] * 100)
                                       .round(2).to_string()))
    allsig = sig.copy()
    per_game = allsig.groupby("game_key").clv.mean()
    print(f"\nAll signal-days (not just first): {len(allsig)} rows over {len(per_game)} games; "
          f"mean CLV {allsig.clv.mean() * 100:+.2f}c, game-clustered se "
          f"{per_game.std() / np.sqrt(len(per_game)) * 100:.2f}c")
    no_sig = df[df.ev < MIN_EDGE_GAME]
    print(f"Control (best side on days under threshold): mean CLV {no_sig.clv.mean() * 100:+.2f}c "
          f"over {len(no_sig)} rows")


if __name__ == "__main__":
    main(fetch_only="--fetch-only" in sys.argv)
