"""Per-game résumé features, home-team perspective (home minus away).

margin         final margin (nflverse result)
net_epa        offensive EPA/play, home minus away (run + pass plays)
net_sr         offensive success rate, home minus away
net_epa_live   net_epa on plays with posteam win prob in [5%, 95%] (no garbage time)
luck_pts       points of fumble-recovery + interception luck in the home team's favour
margin_neutral margin - luck_pts
avg_wp         time-weighted mean home win probability (nflverse `wp`, no betting
               line in it) minus 0.5: a "game control" measure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ratings.luck import FUMBLE_SWING_PTS

# EPA of an incompletion with a pass breakup (-0.84) minus an interception
# (-4.24), 1999-2025 regular season.
INT_SWING_PTS = 3.40
WP_LIVE = (0.05, 0.95)


def int_share_by_season(pbp: pd.DataFrame, trailing: int = 3) -> pd.Series:
    """Interceptions / passes defended, trailing `trailing` seasons (no lookahead).

    The share drifts (about 22% around 2000, 17% in 2025), so a fixed constant
    would mis-score whole eras. A season with no history uses its own share.
    """
    pd_ = pbp[pbp.pass_defense_1_player_id.notna()]
    by = pd_.groupby("season").agg(ints=("interception", "sum"), pds=("interception", "size"))
    out = {}
    for s in by.index:
        prior = by[(by.index < s) & (by.index >= s - trailing)]
        src = prior if len(prior) else by.loc[[s]]
        out[s] = src.ints.sum() / src.pds.sum()
    return pd.Series(out)


def _time_weighted_wp(g: pd.DataFrame) -> float:
    g = g[g.home_wp.notna() & g.game_seconds_remaining.notna()]
    g = g.sort_values("game_seconds_remaining", ascending=False)
    t = g.game_seconds_remaining.to_numpy(float)
    dt = np.clip(t - np.append(t[1:], 0.0), 0, None)
    if dt.sum() == 0:
        return np.nan
    return float(np.average(g.home_wp.to_numpy(float), weights=dt))


def game_features(pbp: pd.DataFrame, int_share: pd.Series) -> pd.DataFrame:
    pbp = pbp[pbp.season_type == "REG"]
    games = pbp.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                       home_team=("home_team", "first"),
                                       away_team=("away_team", "first"),
                                       margin=("result", "first"))

    plays = pbp[pbp.play_type.isin(["run", "pass"]) & pbp.epa.notna() & pbp.posteam.notna()]
    off = plays.groupby(["game_id", "posteam"]).agg(epa=("epa", "mean"), sr=("success", "mean"))
    live = plays[plays.wp.between(*WP_LIVE)]
    off_live = live.groupby(["game_id", "posteam"]).epa.mean().rename("epa_live")
    off = off.join(off_live)

    def side(team_col):
        idx = pd.MultiIndex.from_arrays([games.index, games[team_col]])
        return off.reindex(idx).set_axis(games.index)

    h, a = side("home_team"), side("away_team")
    games["net_epa"] = h.epa - a.epa
    games["net_sr"] = h.sr - a.sr
    games["net_epa_live"] = h.epa_live - a.epa_live

    # fumble luck: recoveries above 50% of recoverable fumbles, worth the EPA swing
    fum = pbp[(pbp.fumble == 1) & pbp.fumbled_1_team.notna() & pbp.fumble_recovery_1_team.notna()]
    fum_home = (fum.fumble_recovery_1_team == fum.home_team).groupby(fum.game_id).agg(["sum", "size"])
    games["fum_pts"] = ((fum_home["sum"] - 0.5 * fum_home["size"]) * FUMBLE_SWING_PTS
                        ).reindex(games.index).fillna(0.0)

    # interception luck: each defense's INTs vs share x passes defended
    pdf = pbp[pbp.pass_defense_1_player_id.notna() & pbp.defteam.notna()]
    d = pdf.groupby(["game_id", "home_team", "defteam"]).agg(
        ints=("interception", "sum"), pds=("interception", "size")).reset_index()
    d["season"] = d.game_id.map(games.season)
    d["luck"] = (d.ints - d.season.map(int_share) * d.pds) * INT_SWING_PTS
    d["signed"] = np.where(d.defteam == d.home_team, d.luck, -d.luck)
    games["int_pts"] = d.groupby("game_id").signed.sum().reindex(games.index).fillna(0.0)

    games["luck_pts"] = games.fum_pts + games.int_pts
    games["margin_neutral"] = games.margin - games.luck_pts
    games["avg_wp"] = pbp.groupby("game_id").apply(_time_weighted_wp,
                                                   include_groups=False).reindex(games.index) - 0.5
    return games.reset_index()


def season_features(seasons, load) -> pd.DataFrame:
    """game_features for many seasons; `load(season)` returns that season's pbp."""
    pbp_all = {s: load(s) for s in seasons}
    share = int_share_by_season(pd.concat(pbp_all.values()))
    return pd.concat([game_features(p, share) for p in pbp_all.values()], ignore_index=True)
