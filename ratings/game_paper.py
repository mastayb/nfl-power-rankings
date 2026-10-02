"""Paper trading on Kalshi game markets (moneylines + spread ladders).

For each open game, the model's margin comes from market ratings refitted
WITHOUT that game's own Kalshi price (leave-one-out): past closing lines, win
totals and every other game's line. Its moneyline and spread rungs are then
priced with the Stage A key-number margin model at the pooled sigma.

At most one paper trade per game: the single best expected-value instrument,
opened the first time it clears the edge threshold. Closing line value (CLV) is
recorded after kickoff from Kalshi's hourly candles: the last pre-kickoff mid
on the traded side, minus the entry price. Positive CLV = beat the close.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from ratings.kalshi import KalshiClient, tidy_markets
from ratings.ladder import fit_game, fit_pooled_sigma, game_observations, loc_for
from ratings.linear import fit_margins, line_observations
from ratings.paper import PAPER_DIR, _read, TRADE_COLS, fee
from ratings.schedule import NFLVERSE_TEAMS, game_key, join_games
from ratings.wins import Season, WinProbs, fit_joint, market_ladders

TEAMS = sorted(NFLVERSE_TEAMS)
GAME_SERIES = {"KXNFLGAME": "game_winner", "KXNFLSPREAD": "spread"}
GAME_TRADES = PAPER_DIR / "game_trades.csv"
GAME_MARKS = PAPER_DIR / "game_marks.csv"
MIN_EDGE_GAME = 0.03     # the model misses current lines by ~1.8 pts; demand more than futures
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent


def fetch_game_markets(client: KalshiClient) -> tuple[pd.DataFrame, dict]:
    """Open game-winner and spread markets as a tidy frame, plus raw markets by ticker."""
    snap = {"series": {}}
    for series, kind in GAME_SERIES.items():
        snap["series"][series] = {"kind": kind, "events": [], "markets": client.open_markets(series)}
    raw = {m["ticker"]: m for blob in snap["series"].values() for m in blob["markets"]}
    return tidy_markets(snap), raw


def kickoff_utc(sched: pd.DataFrame) -> pd.Series:
    """game_id -> kickoff time (UTC ISO) from nflverse gameday + gametime (Eastern)."""
    out = {}
    for r in sched.itertuples():
        if isinstance(r.gametime, str) and r.gametime:
            local = datetime.fromisoformat(f"{r.gameday}T{r.gametime}").replace(tzinfo=ET)
            out[r.game_id] = local.astimezone(timezone.utc).isoformat(timespec="minutes")
    return pd.Series(out)


def stage_a_frame(tidy: pd.DataFrame, games: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    """In-memory Stage A: expected home margin per game from its moneyline + ladder."""
    gm = tidy.assign(key=tidy.event.map(game_key))
    obs = {g.game_id: game_observations(gm[gm.key == game_key(g.event)], g.home_team, g.away_team)
           for g in games.itertuples()}
    obs = {k: o for k, o in obs.items() if len(o)}
    ladders = {k: o for k, o in obs.items() if (o.source == "spread").any()}
    sigma, _ = fit_pooled_sigma(model, ladders)
    rows = []
    for g in games.itertuples():
        if g.game_id not in obs:
            continue
        f = fit_game(model, obs[g.game_id]) if g.game_id in ladders else \
            fit_game(model, obs[g.game_id], sigma=sigma)
        rows.append({"game_id": g.game_id, "neutral": g.neutral, "mean": f.mean})
    return pd.DataFrame(rows), sigma


def loo_margins(stage_a: pd.DataFrame, games: pd.DataFrame, sched: pd.DataFrame, model,
                sigma: float, asof, win_tidy: pd.DataFrame, lam_w: float) -> pd.Series:
    """game_id -> model expected home margin, fitted without that game's Kalshi line."""
    obs = line_observations(stage_a, games, sched, model, sigma, asof)
    season = Season.build(sched, TEAMS)
    probs = WinProbs(model, sigma)
    ladders = market_ladders(win_tidy)
    full_lines = fit_margins(obs, "y", TEAMS, weights=obs.weight.to_numpy())
    full = fit_joint(obs, season, ladders, probs, lam_w,
                     start=(full_lines.ratings.reindex(TEAMS).to_numpy(), full_lines.hfa),
                     fixed_hfa=full_lines.hfa)
    out = {}
    for g in games.itertuples():
        held = (obs.source == "kalshi") & (obs.game_id == g.game_id)
        if not held.any():
            continue
        o = obs[~held]
        lines = fit_margins(o, "y", TEAMS, weights=o.weight.to_numpy())
        f = fit_joint(o, season, ladders, probs, lam_w,
                      start=(full.ratings.reindex(TEAMS).to_numpy(), lines.hfa),
                      fixed_hfa=lines.hfa)
        r = f.ratings
        out[g.game_id] = float(r[g.home_team] - r[g.away_team] + (0 if g.neutral else f.hfa))
    return pd.Series(out)


def market_prob(model, mean: float, sigma: float, kind: str, team: str, home: str,
                threshold: float | None) -> float:
    """Model probability that a game market resolves YES (moneyline ties count half)."""
    loc = loc_for(model, "mean", mean, sigma)
    if kind == "game_winner":
        g0 = float(model.G(np.array([0.0]), loc, sigma)[0])
        return g0 if team == home else 1.0 - g0
    if team == home:                                        # home wins by more than t
        return float(model.G(np.array([threshold]), loc, sigma)[0])
    return 1.0 - float(model.G(np.array([-threshold]), loc, sigma)[0])   # away by more than t


def price_games(tidy: pd.DataFrame, raw: dict, games: pd.DataFrame, margins: pd.Series,
                market_means: pd.Series, model, sigma: float, kickoffs: pd.Series) -> pd.DataFrame:
    gm = tidy.assign(key=tidy.event.map(game_key))
    rows = []
    for g in games.itertuples():
        if g.game_id not in margins.index:
            continue
        for r in gm[gm.key == game_key(g.event)].itertuples():
            if r.team not in (g.home_team, g.away_team) or pd.isna(r.yes_bid):
                continue
            p = market_prob(model, margins[g.game_id], sigma, r.kind, r.team, g.home_team,
                            None if pd.isna(r.threshold) else r.threshold)
            m = raw.get(r.market, {})
            no_price = round(1 - r.yes_bid, 4)
            rows.append({
                "ticker": r.market, "event": r.event, "kind": r.kind, "team": r.team,
                "game_id": g.game_id, "game_key": game_key(g.event),
                "kickoff": kickoffs.get(g.game_id), "mu_model": margins[g.game_id],
                "mu_market": market_means.get(g.game_id, np.nan),
                "model_p": p, "yes_bid": r.yes_bid, "yes_ask": r.yes_ask,
                "ev_yes": p - r.yes_ask - fee(r.yes_ask),
                "ev_no": (1 - p) - no_price - fee(no_price),
                "size_yes": float(m.get("yes_ask_size_fp") or 0),
                "size_no": float(m.get("yes_bid_size_fp") or 0),
            })
    return pd.DataFrame(rows)


def best_per_game(priced: pd.DataFrame, trades_path: Path = GAME_TRADES,
                  min_edge: float = MIN_EDGE_GAME) -> pd.DataFrame:
    """One candidate per game: the best tradable side, skipping games already traded."""
    if priced.empty:
        return priced
    traded = set(_read(trades_path, TRADE_COLS).event.map(game_key).dropna())
    p = priced.copy()
    p["best_ev"] = np.where(p.size_yes >= 1, p.ev_yes, -np.inf)
    p["best_ev"] = np.maximum(p.best_ev, np.where(p.size_no >= 1, p.ev_no, -np.inf))
    p = p[(p.best_ev >= min_edge) & ~p.game_key.isin(traded)]
    idx = p.groupby("game_key").best_ev.idxmax()
    best = p.loc[idx].copy()
    # keep only the chosen side so update() opens exactly one trade per game
    yes_better = (best.ev_yes >= best.ev_no) & (best.size_yes >= 1) | (best.size_no < 1)
    best.loc[yes_better, "ev_no"] = -np.inf
    best.loc[~yes_better, "ev_yes"] = -np.inf
    return best


def record_clv(client: KalshiClient, trades_path: Path = GAME_TRADES,
               now: datetime | None = None) -> list[int]:
    """Fill close_price / clv for game trades whose kickoff has passed."""
    now = now or datetime.now(timezone.utc)
    trades = _read(trades_path, TRADE_COLS)
    done = []
    for i, t in trades.iterrows():
        if pd.notna(t.clv) or not isinstance(t.kickoff, str):
            continue
        ko = datetime.fromisoformat(t.kickoff)
        if ko > now:
            continue
        series = t.ticker.split("-")[0]
        k = client.get(f"/series/{series}/markets/{t.ticker}/candlesticks",
                       {"start_ts": int((ko - timedelta(hours=12)).timestamp()),
                        "end_ts": int(ko.timestamp()), "period_interval": 60}).get("candlesticks", [])
        quotes = [(c["yes_bid"].get("close_dollars"), c["yes_ask"].get("close_dollars"))
                  for c in k if c.get("yes_bid") and c.get("yes_ask")]
        quotes = [(float(b), float(a)) for b, a in quotes if b is not None and a is not None]
        if not quotes:
            continue
        bid, ask = quotes[-1]
        mid = (bid + ask) / 2
        close_side = mid if t.side == "yes" else 1 - mid
        trades.loc[i, ["close_price", "clv"]] = [close_side, close_side - t.price]
        done.append(int(t.trade_id))
    if done:
        trades.to_csv(trades_path, index=False)
    return done


def game_pass(base_snapshot: Path, client: KalshiClient, sched: pd.DataFrame, model,
              now: datetime | None = None) -> dict:
    """Price every open game with the leave-one-out model; open / mark / settle trades."""
    from ratings.paper import settled_results, update

    now = now or datetime.now(timezone.utc)
    tidy, raw = fetch_game_markets(client)
    events = sorted(tidy.loc[tidy.kind == "game_winner", "event"].unique())
    games, unmatched = join_games(events, sched)
    if games.empty:
        return {"games": 0}
    stage_a, sigma = stage_a_frame(tidy, games, model)
    lam_w = yaml.safe_load(open(ROOT / "config" / "win_totals.yaml"))["lambda"]
    win_tidy = pd.read_parquet(base_snapshot)
    margins = loo_margins(stage_a, games, sched, model, sigma, now, win_tidy, lam_w)
    priced = price_games(tidy, raw, games, margins, stage_a.set_index("game_id")["mean"],
                         model, sigma, kickoff_utc(sched))
    cands = best_per_game(priced)

    trades = _read(GAME_TRADES, TRADE_COLS)
    missing = sorted(set(trades[trades.status == "open"].ticker) - set(priced.ticker))
    summary = update(pd.concat([cands, priced[priced.ticker.isin(trades.ticker)
                                              & ~priced.ticker.isin(cands.ticker)]
                                .assign(ev_yes=-np.inf, ev_no=-np.inf)]),
                     settled=settled_results(client, missing),
                     now=now.isoformat(timespec="seconds"), trades_path=GAME_TRADES,
                     marks_path=GAME_MARKS, min_edge=MIN_EDGE_GAME)
    summary["clv_recorded"] = record_clv(client, now=now)
    summary.update(games=len(games), unmatched=len(unmatched), sigma=sigma,
                   loo_rmse=float(np.sqrt(np.mean(
                       (margins - stage_a.set_index("game_id")["mean"].reindex(margins.index)) ** 2))))
    summary["top"] = priced.assign(ev=np.maximum(priced.ev_yes, priced.ev_no)) \
        .sort_values("ev", ascending=False).head(8)
    return summary
