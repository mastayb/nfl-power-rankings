"""Markdown rankings table for the Reddit post (ties marked).

Run: python scripts/reddit_table.py [data/snapshots/<ts>.parquet] > table.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.display import display_ranks  # noqa: E402
from ratings.ladder import load_key_weights, win_prob  # noqa: E402

SNAP_DIR = ROOT / "data" / "snapshots"


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def pct(p: float) -> str:
    return "<1%" if 0 < p < 0.01 else f"{p:.0%}"


def main(snap: Path) -> None:
    import nflreadpy as nfl

    b = pd.read_parquet(snap.with_name(snap.stem + "_stage_b.parquet")).set_index("team")
    sim = pd.read_parquet(snap.with_name(snap.stem + "_sim.parquet")).set_index("team")
    nick = nfl.load_teams().to_pandas().set_index("team_abbr").team_nick
    model, sigma = load_key_weights(), float(b.sigma.iloc[0])
    b["shown"] = display_ranks(b.blend)
    mrank = b.market.rank(ascending=False).astype(int)
    rrank = b.resume.rank(ascending=False).astype(int)
    print("| # | Team | Record | Rating | Win % vs. avg | Market rank | Résumé rank | Playoffs | Super Bowl |")
    print("|--:|:--|:-:|--:|--:|--:|--:|--:|--:|")
    for t, r in b.sort_values("blend", ascending=False).iterrows():
        print(f"| {r.shown} | {nick[t]} | {r.record} | {r.blend:+.1f} | "
              f"{win_prob(model, r.blend, sigma):.0%} | {mrank[t]} | {rrank[t]} | "
              f"{sim.loc[t, 'playoffs']:.0%} | {pct(sim.loc[t, 'sb'])} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
