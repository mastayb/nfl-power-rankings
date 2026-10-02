"""Phase 1-2: snapshot pinned Kalshi series, join games to nflverse, print checks.

Run: python scripts/snapshot.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.kalshi import KalshiClient, load_config, take_snapshot, write_snapshot  # noqa: E402
from ratings.schedule import (game_key, join_games, load_schedule,  # noqa: E402
                              season_from_events)

SET_KINDS = ["division", "conference", "super_bowl", "one_seed", "seed"]
SNAP_DIR = ROOT / "data" / "snapshots"

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main() -> None:
    cfg = load_config()
    with KalshiClient(cfg.get("base_url")) as client:
        snap = take_snapshot(client, cfg)
    raw_path, pq_path, df = write_snapshot(snap, SNAP_DIR)

    section("Snapshot")
    print(f"base_url  {snap['base_url']}\nfetched   {snap['fetched_at']}")
    print(f"raw       {raw_path.relative_to(ROOT)}\ntidy      {pq_path.relative_to(ROOT)}")
    print(df.groupby("kind").agg(events=("event", "nunique"), markets=("market", "size"))
          .to_string())

    # --- games ---------------------------------------------------------------
    game_events = sorted(df.loc[df.kind == "game_winner", "event"].unique())
    spread_events = sorted(df.loc[df.kind == "spread", "event"].unique())
    season = season_from_events(game_events)
    sched = load_schedule(season)
    matched, unmatched = join_games(game_events, sched)
    games_path = pq_path.with_name(pq_path.stem + "_games.parquet")
    matched.to_parquet(games_path, index=False)

    section(f"Kalshi game events -> nflverse game_id (season {season})")
    print(f"matched {len(matched)} / {len(game_events)}   -> {games_path.relative_to(ROOT)}")
    print(f"UNMATCHED: {len(unmatched)}")
    if len(unmatched):
        print(unmatched.to_string(index=False))
    flipped = matched[~matched.home_away_agree]
    print(f"home/away order disagrees with nflverse: {len(flipped)}")
    if len(flipped):
        print(flipped[["event", "game_id"]].to_string(index=False))
    orphan_spreads = sorted({game_key(e) for e in spread_events} - {game_key(e) for e in game_events})
    print(f"spread events with no game event: {len(orphan_spreads)} {orphan_spreads or ''}")

    print("\nNeutral-site games (nflverse location == 'Neutral'):")
    neutral = sched[sched.neutral][["game_id", "gameday", "stadium"]].copy()
    on_kalshi = set(matched.game_id)
    neutral["on_kalshi"] = neutral.game_id.isin(on_kalshi)
    print(neutral.to_string(index=False))

    remaining = sched[sched.result.isna() & (sched.gameday >= date.today())]
    missing = remaining[~remaining.game_id.isin(on_kalshi)]
    print(f"\nRemaining nflverse games: {len(remaining)}; with a KXNFLGAME event: "
          f"{remaining.game_id.isin(on_kalshi).sum()}; without: {len(missing)}")
    if len(missing):
        print("  without, by week:", missing.groupby("week").size().to_dict())

    # --- futures sets ----------------------------------------------------------
    section("Mutually exclusive futures sets: sum of mids")
    fut = df[df.kind.isin(SET_KINDS)]
    sets = (fut.groupby(["kind", "event"])
            .agg(n=("market", "size"), sum_mid=("mid", "sum"), sum_bid=("yes_bid", "sum"),
                 sum_ask=("yes_ask", "sum"), le_2c=("mid", lambda m: (m <= 0.02).sum()),
                 med_width=("width", "median"))
            .reset_index())
    sets["kind"] = pd.Categorical(sets["kind"], SET_KINDS, ordered=True)
    sets = sets.sort_values(["kind", "event"])
    sets["flag"] = ""
    sets.loc[sets.sum_mid > 1.15, "flag"] = "sum>1.15"
    sets.loc[(sets.kind == "division") & (sets.n != 4), "flag"] += " n!=4"
    print(sets.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    po = df[df.kind == "playoff"]
    if len(po):
        print(f"\n(extra, not exclusive) playoff qualifiers: {len(po)} teams, "
              f"sum of mids {po['mid'].sum():.3f} vs 14 berths")

    # --- spread rungs ------------------------------------------------------------
    section("Spread rungs per game")
    sp = df[df.kind == "spread"].assign(key=lambda d: d.event.map(game_key))
    g = matched.assign(key=matched.event.map(game_key))
    rows = []
    for _, r in g.sort_values(["gameday", "game_id"]).iterrows():
        ladder = sp[sp.key == r.key]
        rows.append({
            "game_id": r.game_id, "neutral": r.neutral,
            "rungs": len(ladder),
            "home_rungs": (ladder.team == r.home_team).sum(),
            "away_rungs": (ladder.team == r.away_team).sum(),
            "k_range": (f"{ladder.threshold.min():g}-{ladder.threshold.max():g}"
                        if len(ladder) else ""),
            "med_width": ladder.width.median() if len(ladder) else None,
        })
    rungs = pd.DataFrame(rows)
    print(rungs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\ngames with a ladder: {(rungs.rungs > 0).sum()} / {len(rungs)}")


if __name__ == "__main__":
    main()
