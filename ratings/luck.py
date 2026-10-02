"""Luck indicators for the résumé: close-game record and fumble recovery luck.

Neither feeds the ratings; they explain gaps between résumé and market.

close_luck  wins above .500 in one-score games (|margin| <= 8), ties half.
fum_luck    fumbles recovered minus 50% of all recoverable fumbles in the
            team's games (own and opponent's). Recovery is a coin flip:
            offenses kept 50.2% of their own fumbles in 2022-2025.
            Fumbles out of bounds are excluded (no recovery to win).
fum_pts     fum_luck x the EPA swing between keeping and losing a fumble.
to_margin   takeaways - giveaways (INTs + fumbles lost), for context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ONE_SCORE = 8
# Mean EPA of offensive run/pass fumbles kept (-0.84) minus lost (-5.00),
# 2022-2025 regular + post season, n=2021, se 0.07.
FUMBLE_SWING_PTS = 4.16


def close_games(schedule: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    p = schedule[schedule.result.notna()]
    rows = []
    for t in teams:
        m = pd.concat([p[p.home_team == t].result, -p[p.away_team == t].result])
        c = m[m.abs() <= ONE_SCORE]
        w, l, ties = int((c > 0).sum()), int((c < 0).sum()), int((c == 0).sum())
        rows.append({"team": t, "close_w": w, "close_l": l, "close_t": ties,
                     "close_luck": w + 0.5 * ties - 0.5 * len(c)})
    return pd.DataFrame(rows).set_index("team")


def turnovers(pbp: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    f = pbp[(pbp.fumble == 1) & pbp.fumbled_1_team.notna()]
    rec = f[f.fumble_recovery_1_team.notna()]          # drops out-of-bounds fumbles
    ints = pbp[pbp.interception == 1]
    rows = []
    for t in teams:
        in_game = rec[(rec.posteam == t) | (rec.defteam == t)]
        recovered = int((in_game.fumble_recovery_1_team == t).sum())
        fum_lost = int(((rec.fumbled_1_team == t) & (rec.fumble_recovery_1_team != t)).sum())
        fum_taken = int(((rec.fumbled_1_team != t) & (rec.fumble_recovery_1_team == t)
                         & ((rec.posteam == t) | (rec.defteam == t))).sum())
        int_thrown = int((ints.posteam == t).sum())
        int_taken = int((ints.defteam == t).sum())
        luck = recovered - 0.5 * len(in_game)
        rows.append({"team": t, "fumbles": len(in_game), "recovered": recovered,
                     "fum_luck": luck, "fum_pts": luck * FUMBLE_SWING_PTS,
                     "to_margin": (int_taken + fum_taken) - (int_thrown + fum_lost)})
    return pd.DataFrame(rows).set_index("team")


def luck_table(schedule: pd.DataFrame, pbp: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    return close_games(schedule, teams).join(turnovers(pbp, teams))


def load_pbp(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    cols = ["game_id", "season_type", "posteam", "defteam", "play_type", "fumble",
            "fumbled_1_team", "fumble_recovery_1_team", "interception"]
    p = nfl.load_pbp(seasons=[season]).select(cols).to_pandas()
    return p[p.season_type == "REG"]
