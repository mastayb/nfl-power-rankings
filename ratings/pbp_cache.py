"""Season-by-season play-by-play cache (selected columns) under data/cache/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

PBP_COLS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam",
    "play_type", "epa", "success", "wp", "home_wp", "qb_dropback", "game_seconds_remaining",
    "fumble", "fumble_lost", "fumbled_1_team", "fumble_recovery_1_team",
    "interception", "incomplete_pass", "pass_defense_1_player_id", "result",
]


def load_pbp_cached(season: int, refresh: bool = False) -> pd.DataFrame:
    path = CACHE / f"pbp_{season}.parquet"
    if refresh or not path.exists():
        import nflreadpy as nfl

        CACHE.mkdir(parents=True, exist_ok=True)
        nfl.load_pbp(seasons=[season]).select(PBP_COLS).to_pandas().to_parquet(path, index=False)
    return pd.read_parquet(path)
