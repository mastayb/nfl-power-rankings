"""Discover NFL series from Kalshi's /series listing and pin them in config.

Run: python -m ratings.discover

Candidates come only from the /series listing (title patterns over Football
series), never from guessed tickers. Each candidate is then checked for open
markets; ones with none are kept in the config with active: false.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import yaml

from ratings.kalshi import CONFIG_PATH, KalshiClient

# kind -> regex over series title. Order matters: first match wins.
KIND_PATTERNS = [
    ("game_winner", r"^NFL Game$"),
    ("spread", r"^Pro Football Spread$"),
    ("win_total", r"^Pro Football Wins$|^Pro football wins (?!any)"),
    ("exact_wins", r"^Pro football exact wins"),
    ("division", r"^(American|National) Football Conference (East|North|South|West) Winner$"),
    ("conference", r"^(American|National) Football Conference Champion$|^(AFC|NFC) Championship$"),
    ("super_bowl", r"^Super Bowl$"),
    ("one_seed", r"win conference in regular season"),
    # Not requested, but same shape and cheap; flagged extra.
    ("seed", r"^NFL Playoff Seed$"),
    ("playoff", r"^Pro Football Playoff Qualifiers$"),
]
EXTRA_KINDS = {"seed", "playoff"}

# Plain-English settlement summaries, from reading each series' market rules.
SETTLEMENT = {
    "game_winner": (
        "One YES market per team per game: YES if that team wins the game "
        "'originally scheduled for' the date. Tie -> both markets settle $0.50. "
        "Postponed: stays open if the game starts within 48h of the original "
        "kickoff, else resolves to a 'fair price'. Source: NFL."),
    "spread": (
        "Ladder per team: YES if that team wins by MORE than k.5 points "
        "(strike_type greater, floor_strike = k.5). Each team has its own "
        "rungs; a home ladder rung at k.5 is P(home margin > k.5). No tie/OT "
        "special case beyond the final margin. Source: NFL."),
    "win_total": (
        "Per team event; market n is YES if the team wins AT LEAST n "
        "regular-season games (strike_type greater_or_equal). Rungs already "
        "reached are finalized and drop out of the open set. Ties count as 0 "
        "wins toward 'at least n'. Source: NFL / ESPN."),
    "exact_wins": "YES if the team finishes with exactly n regular-season wins.",
    "division": (
        "One event per division, mutually exclusive over its 4 teams: YES if "
        "the team wins the division (official NFL tiebreakers). Settles at end "
        "of regular season. Source: NFL."),
    "conference": (
        "Mutually exclusive over the conference's 16 teams: YES if the team "
        "wins the AFC/NFC Championship Game, i.e. the conference's Super Bowl "
        "entrant (not the regular-season #1 seed). Source: NFL."),
    "super_bowl": (
        "Mutually exclusive over 32 teams: YES if the team wins the Super Bowl "
        "(the '2027 Pro Football Championship' for the 2026 season). "
        "Source: AP / ESPN / Fox Sports."),
    "one_seed": (
        "One event per conference, mutually exclusive over its 16 teams: YES "
        "if the team is the #1 seed at the end of the regular season. Note the "
        "event suffix is AFC26/NFC26 (not 27 like the other futures) but the "
        "rules say the 2026-2027 regular season. Source: NFL / ESPN."),
    "seed": (
        "One event per conference per seed #2-#7, mutually exclusive over 16 "
        "teams: YES if the team holds that playoff seed. Source: NFL / ESPN."),
    "playoff": (
        "One market per team, NOT mutually exclusive: YES if the team makes "
        "the playoffs. Mids should sum to ~14. Source: ESPN / NFL."),
}

BOILERPLATE = re.compile(r"\s*Kalshi is not affiliated.*$", re.S)


def classify(title: str) -> str | None:
    for kind, pat in KIND_PATTERNS:
        if re.search(pat, title, re.I):
            return kind
    return None


def discover(client: KalshiClient) -> dict:
    all_series = client.list_series(category="Sports")
    entries = []
    for s in sorted(all_series, key=lambda s: s["ticker"]):
        if "Football" not in (s.get("tags") or []):
            continue
        kind = classify(s["title"])
        if kind is None or "NCAA" in s["ticker"]:
            continue
        events = client.open_events(s["ticker"])
        markets = client.open_markets(s["ticker"]) if events else []
        sample = markets[0] if markets else {}
        entries.append({
            "ticker": s["ticker"],
            "title": s["title"],
            "kind": kind,
            "extra": kind in EXTRA_KINDS,
            "active": bool(markets),
            "mutually_exclusive": bool(events and all(e.get("mutually_exclusive") for e in events)),
            "open_events": sorted(e["event_ticker"] for e in events),
            "n_open_markets": len(markets),
            "settlement": SETTLEMENT[kind],
            "rules_example": {
                "market": sample.get("ticker"),
                "primary": sample.get("rules_primary"),
                "secondary": BOILERPLATE.sub("", sample.get("rules_secondary") or "") or None,
            } if sample else None,
            "settlement_sources": [x["name"] for x in s.get("settlement_sources") or []],
            "contract_terms_url": s.get("contract_terms_url"),
        })
    return {
        "discovered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": client.base_url,
        "n_sports_series_listed": len(all_series),
        "series": entries,
    }


def main() -> None:
    with KalshiClient() as client:
        cfg = discover(client)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Pinned Kalshi NFL series. Generated by `python -m ratings.discover`\n"
              "# from the /series listing; edit `active` to pin/unpin.\n")
    with open(CONFIG_PATH, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, width=100, allow_unicode=True)
    for e in cfg["series"]:
        flag = "active" if e["active"] else "no open markets"
        print(f"{e['ticker']:22} {e['kind']:12} events={len(e['open_events']):3} "
              f"markets={e['n_open_markets']:4}  {flag}{'  (extra)' if e['extra'] else ''}")


if __name__ == "__main__":
    main()
