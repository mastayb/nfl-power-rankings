"""Check the simulator's standings logic against real playoff fields, 2020-2025.

Run: python scripts/validate_tiebreakers.py

Feeds each completed season's actual results through the same standings code
the simulation uses, then compares with who really made the playoffs (from
nflverse playoff games): the 7 qualifiers, the 4 division winners, the #1
seed, and the three wild-card pairings (home team = higher seed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nflreadpy as nfl  # noqa: E402

from ratings.schedule import NFLVERSE_TEAMS  # noqa: E402
from ratings.season_sim import CONFS, League, Schedule, _Sim  # noqa: E402

SEASONS = range(2020, 2026)  # 7-team playoff format


def actual_field(post, conf_of):
    """Per conference: qualifiers, division winners, #1 seed, WC pairs (home, away)."""
    wc = post[post.game_type == "WC"]
    div = post[post.game_type == "DIV"]
    out = {}
    for c in CONFS:
        w = wc[wc.home_team.map(conf_of) == c]
        wc_teams = set(w.home_team) | set(w.away_team)
        d = div[div.home_team.map(conf_of) == c]
        one = (set(d.home_team) | set(d.away_team)) - wc_teams
        assert len(one) == 1, (c, one)
        one = one.pop()
        out[c] = {"field": wc_teams | {one}, "division": set(w.home_team) | {one},
                  "seed1": one, "pairs": set(zip(w.home_team, w.away_team))}
    return out


def main() -> None:
    teams = sorted(NFLVERSE_TEAMS)
    league = League.from_nflverse(teams)
    conf_of = {t: CONFS[c] for t, c in zip(teams, league.conf)}
    sched = nfl.load_schedules(seasons=list(SEASONS)).to_pandas()
    sched["neutral"] = sched.location.eq("Neutral")
    checks = {"field": 0, "division": 0, "seed1": 0, "pairs": 0}
    total = 0
    for season in SEASONS:
        s = sched[sched.season == season]
        reg = s[s.game_type == "REG"]
        cancelled = reg.result.isna().sum()
        sch = Schedule.from_frame(reg, league, complete=True)
        out = np.where(reg.result > 0, 1.0, np.where(reg.result < 0, 0.0, 0.5)).astype(float)
        actual = actual_field(s[s.game_type != "REG"], conf_of)
        # coin flips are vanishingly rare at playoff cut lines; fixed seed for reproducibility
        sim = _Sim(league, sch, out, np.random.default_rng(0))
        for ci, c in enumerate(CONFS):
            seeds, divs = sim.seed_conference(ci)
            names = [teams[t] for t in seeds]
            mine = {"field": set(names), "division": {teams[o[0]] for o in divs},
                    "seed1": names[0], "pairs": {(names[i], names[j]) for i, j in [(1, 6), (2, 5), (3, 4)]}}
            ok = {k: mine[k] == actual[c][k] for k in checks}
            for k, v in ok.items():
                checks[k] += v
            total += 1
            flag = "" if all(ok.values()) else "  <-- MISMATCH " + ", ".join(k for k, v in ok.items() if not v)
            print(f"{season} {c}: seeds {' '.join(names)}"
                  + (f"  ({cancelled} cancelled game)" if cancelled else "") + flag)
            if not all(ok.values()):
                print(f"         actual: #1 {actual[c]['seed1']}, div {sorted(actual[c]['division'])}, "
                      f"WC {sorted(actual[c]['pairs'])}")
    print("\nmatches out of", total, "conference-seasons:",
          ", ".join(f"{k} {v}/{total}" for k, v in checks.items()))


if __name__ == "__main__":
    main()
