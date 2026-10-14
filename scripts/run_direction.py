"""Do run-direction splits persist? Inside vs outside, left vs right (2016-2025).

Run: python -u scripts/run_direction.py

inside = middle or guard gap; outside = tackle or end gap (nflverse run_location /
run_gap). Designed runs only (scrambles and kneels excluded). Each measure is the
team's/rusher's situation-adjusted EPA per carry in the context minus its own
EPA per carry on all runs; defense measures are the same for EPA allowed.
Persistence: odd vs even games, first vs second half, season to season.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from player_persistence import evaluate  # noqa: E402

COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
        "epa", "down", "ydstogo", "yardline_100", "wp", "rush", "pass", "qb_scramble",
        "rusher_player_id", "run_location", "run_gap"]
CACHE = ROOT / "data" / "cache"


def load() -> pd.DataFrame:
    out = []
    for y in range(2016, 2026):
        path = CACHE / f"pbp_runs_{y}.parquet"
        if not path.exists():
            import nflreadpy as nfl

            nfl.load_pbp(seasons=[y]).select(COLS).to_pandas().to_parquet(path, index=False)
        out.append(pd.read_parquet(path))
    d = pd.concat(out, ignore_index=True)
    d = d[(d.season_type == "REG") & (d.play_type == "run") & (d.qb_scramble != 1) & d.epa.notna()
          & d.run_location.notna() & d.down.notna() & d.wp.notna()].copy()
    d["inside"] = (d.run_location == "middle") | (d.run_gap == "guard")
    d["outside"] = d.run_gap.isin(["tackle", "end"])
    dd = d.down.astype(int).astype(str) + pd.cut(d.ydstogo, [0, 2, 5, 10, 99], labels=list("smlx")).astype(str)
    fp = pd.cut(d.yardline_100, [0, 10, 20, 40, 60, 80, 100]).astype(str)
    wpb = pd.cut(d.wp, [-0.01, 0.1, 0.25, 0.75, 0.9, 1.01]).astype(str)
    d["adj"] = d.epa - d.groupby([dd, fp, wpb]).epa.transform("mean")
    order = d[["season", "posteam", "week"]].drop_duplicates().sort_values("week")
    order["gidx"] = order.groupby(["season", "posteam"]).cumcount()
    d = d.merge(order, on=["season", "posteam", "week"])
    d["oddeven"] = np.where(d.gidx % 2 == 0, 1, 2)
    d["halves"] = np.where(d.week <= 9, 1, 2)
    return d


def main() -> None:
    d = load()
    allrun = pd.Series(True, index=d.index)
    print(f"{len(d):,} designed runs; inside {d.inside.mean():.0%}, outside {d.outside.mean():.0%}; "
          f"EPA/carry inside {d[d.inside].epa.mean():+.3f}, outside {d[d.outside].epa.mean():+.3f}")
    rb = d.rusher_player_id.notna()
    dd = d.assign(defunit=d.defteam)
    rows = [
        evaluate("Team offense: outside runs (vs own runs)", d, "posteam", d.outside, allrun, 30, 120),
        evaluate("Team offense: inside runs (vs own runs)", d, "posteam", d.inside, allrun, 30, 120),
        evaluate("Team offense: runs left (vs own runs)", d, "posteam", d.run_location == "left", allrun, 30, 120),
        evaluate("Rusher: outside runs (vs own carries)", d, "rusher_player_id", d.outside, rb, 15, 50),
        evaluate("Rusher: inside runs (vs own carries)", d, "rusher_player_id", d.inside, rb, 15, 50),
        evaluate("Defense: outside runs allowed (vs own)", dd, "defteam", d.outside, allrun, 30, 120),
        evaluate("Defense: inside runs allowed (vs own)", dd, "defteam", d.inside, allrun, 30, 120),
    ]
    tab = pd.DataFrame(rows)
    print(tab[["measure", "r_oddeven", "n_oddeven", "r_halves", "r_season_to_season", "n_s2s"]]
          .round(3).to_string(index=False))
    # tendencies (usage), for contrast: how stable is a team's outside-run share?
    share = d.groupby(["season", "posteam", "oddeven"]).outside.mean().unstack()
    print(f"\nTendency, not skill: team outside-run share odd/even r = {share[1].corr(share[2]):.2f}")
    tab.to_parquet(ROOT / "data" / "run_direction.parquet", index=False)


if __name__ == "__main__":
    main()
