"""nflverse schedule + results, team code mapping, Kalshi game -> game_id join."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

NFLVERSE_TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})

# Kalshi codes that differ from nflverse.
KALSHI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA"}
KALSHI_TEAMS = (NFLVERSE_TEAMS - set(KALSHI_TO_NFLVERSE.values())) | set(KALSHI_TO_NFLVERSE)

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

# KXNFLGAME-26OCT04INDWAS: yy MON dd, then away code + home code, unseparated.
GAME_EVENT_RE = re.compile(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$")


def normalize_team(code: str) -> str:
    """Kalshi team code -> nflverse team code. KeyError if unknown."""
    t = KALSHI_TO_NFLVERSE.get(code, code)
    if t not in NFLVERSE_TEAMS:
        raise KeyError(code)
    return t


def parse_game_event(event_ticker: str) -> tuple[date, str, str]:
    """Kalshi game event ticker -> (date, away, home) in nflverse codes.

    Codes are 2-3 letters with no separator, so try every split and require
    exactly one where both halves are real teams.
    """
    m = GAME_EVENT_RE.match(event_ticker)
    if not m:
        raise ValueError(f"not a game event ticker: {event_ticker}")
    yy, mon, dd, teams = m.groups()
    d = date(2000 + int(yy), MONTHS[mon], int(dd))
    splits = [(teams[:i], teams[i:]) for i in range(2, len(teams) - 1)
              if teams[:i] in KALSHI_TEAMS and teams[i:] in KALSHI_TEAMS]
    if len(splits) != 1:
        raise ValueError(f"cannot split teams in {event_ticker}: {splits}")
    away, home = splits[0]
    return d, normalize_team(away), normalize_team(home)


def nfl_season(d: date) -> int:
    """Season year for a game date (Jan/Feb games belong to the prior season)."""
    return d.year if d.month >= 3 else d.year - 1


def load_schedule(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    s = nfl.load_schedules(seasons=[season]).to_pandas()
    s["gameday"] = pd.to_datetime(s["gameday"]).dt.date
    s["neutral"] = s["location"].eq("Neutral")
    return s


def join_games(event_tickers: list[str], schedule: pd.DataFrame,
               max_day_diff: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match Kalshi game events to nflverse game_ids.

    Match on the unordered team pair within `max_day_diff` days (Kalshi dates
    are US-local, so a late or overseas kickoff can straddle a day). Returns
    (matched, unmatched); matched carries `home_away_agree`, False when Kalshi
    lists the teams in the opposite order to nflverse.
    """
    sched = schedule[["game_id", "season", "week", "gameday", "away_team",
                      "home_team", "neutral", "stadium"]]
    matched, unmatched = [], []
    for ev in event_tickers:
        try:
            d, away, home = parse_game_event(ev)
        except ValueError as e:
            unmatched.append({"event": ev, "reason": str(e)})
            continue
        pair = {away, home}
        cand = sched[
            sched.apply(lambda r: {r.away_team, r.home_team} == pair, axis=1)
            & sched.gameday.map(lambda g: abs((g - d).days) <= max_day_diff)
        ]
        if len(cand) != 1:
            unmatched.append({"event": ev, "kalshi_date": d, "away": away, "home": home,
                              "reason": f"{len(cand)} nflverse candidates"})
            continue
        g = cand.iloc[0]
        matched.append({
            "event": ev, "kalshi_date": d, "kalshi_away": away, "kalshi_home": home,
            "game_id": g.game_id, "week": int(g.week), "gameday": g.gameday,
            "away_team": g.away_team, "home_team": g.home_team,
            "neutral": bool(g.neutral), "stadium": g.stadium,
            "home_away_agree": g.away_team == away and g.home_team == home,
        })
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def game_key(event_ticker: str) -> str:
    """Strip the series prefix so KXNFLGAME-X and KXNFLSPREAD-X share a key."""
    return event_ticker.split("-", 1)[1]


def season_from_events(event_tickers: list[str]) -> int:
    seasons = {nfl_season(parse_game_event(e)[0]) for e in event_tickers}
    if len(seasons) != 1:
        raise ValueError(f"events span seasons {seasons}")
    return seasons.pop()
