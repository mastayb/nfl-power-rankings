"""Xs and Os play table: nflverse play-by-play joined to participation, normalised.

Participation (2016-2025; nflverse, from NGS through 2022 and FTN Data from 2023,
CC-BY-SA) gives personnel, formation, box count, pass rushers and, from 2018,
coverage on pass plays. The two sources format personnel differently:

  2016-22  offense "1 RB, 1 TE, 3 WR"            defense "4 DL, 2 LB, 5 DB"
  2023-25  offense "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR"
           defense "3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS"

Normalised fields:
  off_pers    "11", "12", "13", "21", "22", "10", ... (RBs then TEs), "other"
  jumbo       6+ offensive linemen (only knowable when linemen are listed)
  def_pers    "base" (<= 4 DBs), "nickel" (5), "dime" (6+)
  shell       "cover0", "single-high" (Cover 1/3), "two-high" (Cover 2/4/6, 2-Man),
              "other" (Cover 9, combo, blown, prevent); pass plays 2018+ only
  man         True / False / NaN (man vs zone, pass plays 2018+)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
SEASONS = range(2016, 2026)

PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "home_team", "away_team",
            "posteam", "defteam", "play_type", "epa", "success", "yards_gained", "down", "ydstogo",
            "yardline_100", "qtr", "half_seconds_remaining", "score_differential", "wp",
            "pass", "rush", "qb_dropback", "xpass", "pass_oe", "shotgun", "no_huddle"]

DB_POS = {"DB", "CB", "FS", "SS", "S", "SAF"}
OL_POS = {"OL", "C", "G", "T", "OT", "OG"}
SINGLE_HIGH = {"COVER_1", "COVER_3"}
TWO_HIGH = {"COVER_2", "COVER_4", "COVER_6", "2_MAN"}


def parse_personnel(s) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(s, str):
        return out
    for n, pos in re.findall(r"(\d+)\s+([A-Z]+)", s):
        out[pos] = out.get(pos, 0) + int(n)
    return out


def off_code(c: dict[str, int]) -> str:
    if not c:
        return "unknown"
    rb = c.get("RB", 0) + c.get("FB", 0)
    te = c.get("TE", 0)
    if rb > 3 or te > 4:
        return "other"
    return f"{rb}{te}"


def def_group(c: dict[str, int]) -> str:
    if not c:
        return "unknown"
    db = sum(v for k, v in c.items() if k in DB_POS)
    return "base" if db <= 4 else "nickel" if db == 5 else "dime"


def shell(cov) -> str | float:
    if not isinstance(cov, str) or not cov:
        return np.nan
    if cov == "COVER_0":
        return "cover0"
    if cov in SINGLE_HIGH:
        return "single-high"
    if cov in TWO_HIGH:
        return "two-high"
    return "other"


def _pbp(season: int) -> pd.DataFrame:
    path = CACHE / f"pbp_xo_{season}.parquet"
    if not path.exists():
        import nflreadpy as nfl

        CACHE.mkdir(parents=True, exist_ok=True)
        nfl.load_pbp(seasons=[season]).select(PBP_COLS).to_pandas().to_parquet(path, index=False)
    return pd.read_parquet(path)


def _participation(season: int) -> pd.DataFrame:
    path = CACHE / f"participation_{season}.parquet"
    if not path.exists():
        import nflreadpy as nfl

        nfl.load_participation(seasons=[season]).to_pandas().to_parquet(path, index=False)
    return pd.read_parquet(path)


def load_plays(seasons=SEASONS) -> pd.DataFrame:
    """Regular-season run/pass plays with EPA, joined to normalised participation."""
    out = []
    for y in seasons:
        p = _pbp(y)
        p = p[(p.season_type == "REG") & p.play_type.isin(["run", "pass"]) & p.epa.notna()
              & p.posteam.notna()]
        part = _participation(y)[["nflverse_game_id", "play_id", "offense_personnel",
                                  "defense_personnel", "offense_formation", "defenders_in_box",
                                  "number_of_pass_rushers", "defense_man_zone_type",
                                  "defense_coverage_type", "time_to_throw", "was_pressure"]]
        part = part.assign(play_id=part.play_id.astype(float))
        p = p.assign(play_id=p.play_id.astype(float)).merge(
            part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
        oc = p.offense_personnel.map(parse_personnel)
        dc = p.defense_personnel.map(parse_personnel)
        p["off_pers"] = oc.map(off_code)
        p["jumbo"] = oc.map(lambda c: sum(v for k, v in c.items() if k in OL_POS) >= 6)
        p["def_pers"] = dc.map(def_group)
        p["shell"] = p.defense_coverage_type.map(shell)
        mz = p.defense_man_zone_type.replace("", np.nan)
        p["man"] = mz.map({"MAN_COVERAGE": True, "ZONE_COVERAGE": False})
        p["off_ts"] = p.posteam + "_" + p.season.astype(str)
        p["def_ts"] = p.defteam + "_" + p.season.astype(str)
        out.append(p.drop(columns=["nflverse_game_id", "offense_personnel", "defense_personnel",
                                   "defense_man_zone_type", "defense_coverage_type"]))
    return pd.concat(out, ignore_index=True)
