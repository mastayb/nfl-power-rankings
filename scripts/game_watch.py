"""Hourly: catch newly listed / newly tightened game markets; record CLV after kickoffs.

Run: python -u scripts/game_watch.py      (cron, hourly at :15)

Cheap by default: lists open game markets and compares them with the last
run's state (data/paper/watch_state.json). The full leave-one-out game pass
runs only when a game is newly listed, or its median bid/ask width first drops
to TIGHT or below (quotes at listing are placeholders, e.g. 24c / 74c). CLV for
game trades past kickoff is recorded every run. Commits data/paper when the
logs change.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.game_paper import fetch_game_markets, game_pass, record_clv  # noqa: E402
from ratings.kalshi import KalshiClient  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.paper import PAPER_DIR, JobLock, commit_paths  # noqa: E402
from ratings.schedule import game_key, load_schedule  # noqa: E402

STATE = PAPER_DIR / "watch_state.json"
TIGHT = 0.03
SNAP_DIR = ROOT / "data" / "snapshots"


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def triggers(tidy, prev: dict) -> tuple[dict, list[str]]:
    """Per game: median width now; games that are new or newly tight."""
    width = tidy.assign(key=tidy.event.map(game_key)).groupby("key").width.median()
    now = {k: float(v) for k, v in width.items()}
    fired = [k for k, w in now.items()
             if k not in prev or (w <= TIGHT and prev[k] > TIGHT)]
    return now, fired


def main() -> None:
    stamp = datetime.now(timezone.utc)
    with JobLock(), KalshiClient() as client:
        tidy, _ = fetch_game_markets(client)
        prev = json.loads(STATE.read_text()) if STATE.exists() else {}
        now, fired = triggers(tidy, prev)
        print(f"[{time.strftime('%H:%M:%S')}] {len(now)} open games; triggers: {fired or 'none'}")
        changed = False
        if fired:
            g = game_pass(latest_snapshot(), client, load_schedule(stamp.year if stamp.month >= 3
                                                                   else stamp.year - 1),
                          load_key_weights(), now=stamp)
            print(f"game pass: {g.get('games', 0)} games; opened {g.get('opened', [])}, closed "
                  f"{g.get('closed', [])}, CLV {g.get('clv_recorded', [])}")
            changed = bool(g.get("opened") or g.get("closed") or g.get("clv_recorded"))
        else:
            clv = record_clv(client, now=stamp)
            if clv:
                print(f"CLV recorded for trades {clv}")
            changed = bool(clv)
        STATE.write_text(json.dumps(now, indent=0, sort_keys=True))
    if changed:
        print(commit_paths(ROOT, ["data/paper"], f"game watch: {stamp:%Y-%m-%d %H:%M}Z"))


if __name__ == "__main__":
    main()
