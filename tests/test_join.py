from datetime import date

import pandas as pd
import pytest

from ratings.schedule import (game_key, join_games, nfl_season, normalize_team,
                              parse_game_event, season_from_events)


@pytest.mark.parametrize("ticker,expected", [
    ("KXNFLGAME-26OCT04INDWAS", (date(2026, 10, 4), "IND", "WAS")),
    ("KXNFLGAME-26OCT04NYJCHI", (date(2026, 10, 4), "NYJ", "CHI")),
    ("KXNFLGAME-26OCT04LACSEA", (date(2026, 10, 4), "LAC", "SEA")),
    ("KXNFLGAME-26OCT04LARPHI", (date(2026, 10, 4), "LA", "PHI")),
    ("KXNFLGAME-26OCT04JACCIN", (date(2026, 10, 4), "JAX", "CIN")),
    ("KXNFLGAME-26OCT04GBTB", (date(2026, 10, 4), "GB", "TB")),
    ("KXNFLGAME-26OCT11LVNE", (date(2026, 10, 11), "LV", "NE")),
    ("KXNFLGAME-26OCT04KCLV", (date(2026, 10, 4), "KC", "LV")),
    ("KXNFLSPREAD-26OCT12BUFLAR", (date(2026, 10, 12), "BUF", "LA")),
    ("KXNFLGAME-27JAN03NOTB", (date(2027, 1, 3), "NO", "TB")),
])
def test_parse_game_event(ticker, expected):
    assert parse_game_event(ticker) == expected


@pytest.mark.parametrize("bad", ["KXNFLGAME-26OCT04XXXYYY", "KXSB-27", "KXNFLGAME-26OCT04IND"])
def test_parse_game_event_rejects(bad):
    with pytest.raises(ValueError):
        parse_game_event(bad)


def test_normalize_team():
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("LAR") == "LA"
    assert normalize_team("KC") == "KC"
    with pytest.raises(KeyError):
        normalize_team("STL")


def test_nfl_season():
    assert nfl_season(date(2026, 10, 4)) == 2026
    assert nfl_season(date(2027, 1, 10)) == 2026
    assert season_from_events(["KXNFLGAME-26OCT04INDWAS", "KXNFLGAME-27JAN03NOTB"]) == 2026


def test_game_key_shared_across_series():
    assert game_key("KXNFLGAME-26OCT04INDWAS") == game_key("KXNFLSPREAD-26OCT04INDWAS")


@pytest.fixture
def sched():
    return pd.DataFrame([
        # London game: nflverse keeps the designated home team, location Neutral.
        dict(game_id="2026_04_IND_WAS", season=2026, week=4, gameday=date(2026, 10, 4),
             away_team="IND", home_team="WAS", neutral=True, stadium="Tottenham"),
        dict(game_id="2026_04_NE_BUF", season=2026, week=4, gameday=date(2026, 10, 4),
             away_team="NE", home_team="BUF", neutral=False, stadium="Highmark"),
        dict(game_id="2026_05_BUF_LA", season=2026, week=5, gameday=date(2026, 10, 12),
             away_team="BUF", home_team="LA", neutral=False, stadium="SoFi"),
        # Rematch later in the season: same pair, different date.
        dict(game_id="2026_15_BUF_NE", season=2026, week=15, gameday=date(2026, 12, 20),
             away_team="BUF", home_team="NE", neutral=False, stadium="Gillette"),
    ])


def test_join_matches_and_flags_neutral(sched):
    matched, unmatched = join_games(
        ["KXNFLGAME-26OCT04INDWAS", "KXNFLGAME-26OCT04NEBUF", "KXNFLGAME-26OCT12BUFLAR"], sched)
    assert unmatched.empty
    m = matched.set_index("event")
    assert m.loc["KXNFLGAME-26OCT04INDWAS", "game_id"] == "2026_04_IND_WAS"
    assert m.loc["KXNFLGAME-26OCT04INDWAS", "neutral"]
    assert not m.loc["KXNFLGAME-26OCT04NEBUF", "neutral"]
    assert m.loc["KXNFLGAME-26OCT12BUFLAR", "game_id"] == "2026_05_BUF_LA"  # LAR -> LA
    assert m.home_away_agree.all()


def test_join_uses_date_to_pick_between_rematches(sched):
    matched, _ = join_games(["KXNFLGAME-26DEC20BUFNE"], sched)
    assert matched.iloc[0].game_id == "2026_15_BUF_NE"


def test_join_tolerates_one_day_shift(sched):
    # e.g. an overseas or late kickoff listed under the neighbouring date
    matched, unmatched = join_games(["KXNFLGAME-26OCT05NEBUF"], sched)
    assert unmatched.empty and matched.iloc[0].game_id == "2026_04_NE_BUF"
    matched, unmatched = join_games(["KXNFLGAME-26OCT06NEBUF"], sched)
    assert matched.empty and len(unmatched) == 1


def test_join_flags_reversed_order(sched):
    matched, unmatched = join_games(["KXNFLGAME-26OCT04BUFNE"], sched)
    assert unmatched.empty
    assert matched.iloc[0].game_id == "2026_04_NE_BUF"
    assert not matched.iloc[0].home_away_agree


def test_join_reports_unmatched(sched):
    matched, unmatched = join_games(["KXNFLGAME-26OCT04KCLV", "KXNFLGAME-26OCT04XXXYYY"], sched)
    assert matched.empty
    assert list(unmatched.event) == ["KXNFLGAME-26OCT04KCLV", "KXNFLGAME-26OCT04XXXYYY"]
