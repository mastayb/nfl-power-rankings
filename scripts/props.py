"""Snapshot Kalshi player props and turn them into market-implied player projections.

Run: python -u scripts/props.py
Writes data/props/<UTC ts>_markets.parquet (raw rows) and <UTC ts>_projections.parquet.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.kalshi import KalshiClient  # noqa: E402
from ratings.props import fetch, match_gsis, project  # noqa: E402

OUT = ROOT / "data" / "props"
pd.set_option("display.width", 200)


def main() -> pd.DataFrame:
    stamp = datetime.now(timezone.utc)
    with KalshiClient() as client:
        mk = fetch(client)
    proj = project(mk)
    season = stamp.year if stamp.month >= 3 else stamp.year - 1
    proj["gsis_id"] = match_gsis(proj, season)
    OUT.mkdir(parents=True, exist_ok=True)
    ts = stamp.strftime("%Y%m%dT%H%M%SZ")
    mk.to_parquet(OUT / f"{ts}_markets.parquet", index=False)
    proj.to_parquet(OUT / f"{ts}_projections.parquet", index=False)

    print(f"{len(mk)} prop markets -> {len(proj)} player-stat projections "
          f"({proj.slug.nunique()} players, {proj.game.nunique()} games); "
          f"matched to nflverse ids: {proj.gsis_id.notna().mean():.0%}")
    print(proj.groupby("stat").agg(players=("slug", "nunique"), median_rungs=("rungs", "median"),
                                   median_fit_cents=("fit_rmse_cents", "median"),
                                   median_width=("median_width", "median")).round(2).to_string())
    for stat in ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "touchdowns"):
        t = proj[proj.stat == stat].sort_values("mean", ascending=False).head(6)
        if len(t):
            print(f"\nTop {stat}:")
            print(t[["player", "team", "game", "mean", "median", "rungs", "fit_rmse_cents"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nwrote {(OUT / f'{ts}_projections.parquet').relative_to(ROOT)}")
    return proj


if __name__ == "__main__":
    main()
