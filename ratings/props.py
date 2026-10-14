"""Market-implied player projections from Kalshi player-prop markets.

Two market shapes:
  thresholds  "Player: 60+ receiving yards" (floor_strike 59.5, strike_type greater),
              priced P(stat >= 60). A player usually has several rungs = a ladder.
  scalar      "Player receiving yards ladder": pays `Payout Per Unit` per unit up to
              `scalar_cap`, so price / payout_per_unit = market expected value.

Per player and stat, from mid prices (weights 1/width^2, quotes <= 2c / >= 98c dropped):
  yards stats   Gamma(shape, scale) fitted to the survival points; with a single rung
                the shape is fixed at the pooled value for that stat
  count stats   Poisson(lambda) fitted to the survival points
  scalar        mean = mid / payout per unit
Outputs expected value, median, rungs used and fit error (cents).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import gamma, poisson

from ratings.kalshi import KalshiClient, mid_width, to_price

THRESHOLD_SERIES = {
    "KXNFLPASSYDS": ("passing_yards", "yards"),
    "KXNFLRSHYDS": ("rushing_yards", "yards"),
    "KXNFLRECYDS": ("receiving_yards", "yards"),
    "KXNFLRRYDS": ("rush_rec_yards", "yards"),
    "KXNFLREC": ("receptions", "count"),
    "KXNFLPASSTDS": ("passing_tds", "count"),
    "KXNFLPASSATT": ("pass_attempts", "count"),
    "KXNFLPASSCOMP": ("completions", "count"),
    "KXNFLTD": ("touchdowns", "count"),
}
SCALAR_SERIES = {
    "KXNFLLADDERRECYDS": "receiving_yards",
    "KXNFLLADDERRSHYDS": "rushing_yards",
    "KXNFLLADDERREC": "receptions",
}
EDGE, MIN_WIDTH = 0.02, 0.01


def player_key(ticker: str) -> tuple[str, str, str]:
    """KXNFLRECYDS-26OCT18DALGB-DALJWILLIAMS33-60 -> (game key, team, player slug)."""
    from ratings.schedule import KALSHI_TEAMS

    parts = ticker.split("-")
    game, slug = parts[1], parts[2]
    # known team codes, longest first (a lazy regex would cut DAL to DA)
    team = next((slug[:n] for n in (3, 2) if slug[:n] in KALSHI_TEAMS), slug[:3])
    return game, team, slug


def fetch(client: KalshiClient) -> pd.DataFrame:
    rows = []
    for series, (stat, kind) in THRESHOLD_SERIES.items():
        for m in client.paginate("/markets", "markets", {"series_ticker": series, "status": "open"}, limit=1000):
            if m.get("floor_strike") is None:
                continue
            rows.append(_row(m, series, stat, kind, threshold=float(m["floor_strike"]) + 0.5))
    for series, stat in SCALAR_SERIES.items():
        for m in client.paginate("/markets", "markets", {"series_ticker": series, "status": "open"}, limit=1000):
            cs = m.get("custom_strike") or {}
            unit = to_price(str(cs.get("Payout Per Unit", "")).replace("$", "")) if cs.get("Payout Per Unit") else None
            if not unit:
                continue
            rows.append(_row(m, series, stat, "scalar", threshold=np.nan, unit=unit))
    return pd.DataFrame(rows)


def _row(m, series, stat, kind, threshold, unit=None):
    mid, width = mid_width(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
    game, team, slug = player_key(m["ticker"])
    name = (m.get("title") or "").split(":")[0].replace(" receiving yards ladder", "") \
        .replace(" rushing yards ladder", "").replace(" Receptions Ladder", "").strip()
    return {"series": series, "stat": stat, "kind": kind, "ticker": m["ticker"], "game": game,
            "team": team, "slug": slug, "player": name,
            "kalshi_player_id": (m.get("custom_strike") or {}).get("football_player"),
            "threshold": threshold, "unit": unit, "mid": mid, "width": width,
            "close_time": m.get("close_time")}


# --- fitting ------------------------------------------------------------------------------

def _usable(g: pd.DataFrame) -> pd.DataFrame:
    return g[(g.mid > EDGE) & (g.mid < 1 - EDGE) & g.width.notna()]


def fit_gamma(th, q, w, shape=None):
    """Survival points P(X >= t) = q -> gamma (shape, scale). Returns (shape, scale, rmse)."""
    th, q, sw = np.asarray(th, float), np.asarray(q, float), np.sqrt(np.asarray(w, float))
    m0 = float(np.interp(0.5, q[::-1], th[::-1])) if len(th) > 1 else th[0] * (1 + (q[0] - 0.5))
    m0 = max(m0, 1.0)
    if shape is None:
        f = lambda p: (gamma.sf(th - 0.5, np.exp(p[0]), scale=np.exp(p[1])) - q) * sw
        r = least_squares(f, [np.log(2.0), np.log(m0 / 2.0)])
        a, s = np.exp(r.x)
    else:
        f = lambda p: (gamma.sf(th - 0.5, shape, scale=np.exp(p[0])) - q) * sw
        r = least_squares(f, [np.log(m0 / shape)])
        a, s = shape, float(np.exp(r.x[0]))
    resid = gamma.sf(th - 0.5, a, scale=s) - q
    return a, s, float(np.sqrt(np.mean(resid**2)))


def fit_poisson(th, q, w):
    th, q, sw = np.asarray(th, float), np.asarray(q, float), np.sqrt(np.asarray(w, float))
    f = lambda p: (poisson.sf(th - 1, np.exp(p[0])) - q) * sw      # P(X >= t)
    r = least_squares(f, [np.log(max(float(th.mean()), 0.3))])
    lam = float(np.exp(r.x[0]))
    return lam, float(np.sqrt(np.mean((poisson.sf(th - 1, lam) - q) ** 2)))


def project(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, stat): expected value, median, rungs, fit error."""
    out = []
    thr = df[df.kind != "scalar"]
    # pooled gamma shape per yards stat, from players with 3+ usable rungs
    pooled = {}
    for stat, g in thr[thr.kind == "yards"].groupby("stat"):
        shapes = []
        for _, p in g.groupby("slug"):
            u = _usable(p)
            if len(u) >= 3:
                a, _, _ = fit_gamma(u.threshold, u.mid, (MIN_WIDTH / u.width.clip(lower=MIN_WIDTH)) ** 2)
                shapes.append(a)
        pooled[stat] = float(np.median(shapes)) if shapes else 2.0
    for (stat, slug), p in thr.groupby(["stat", "slug"]):
        u = _usable(p).sort_values("threshold")
        if u.empty:
            continue
        w = (MIN_WIDTH / u.width.clip(lower=MIN_WIDTH)) ** 2
        base = {"stat": stat, "slug": slug, "player": p.player.iloc[0], "team": p.team.iloc[0],
                "game": p.game.iloc[0], "kalshi_player_id": p.kalshi_player_id.iloc[0],
                "rungs": len(u), "median_width": float(u.width.median()), "source": "thresholds"}
        if p.kind.iloc[0] == "yards":
            a, s, rmse = fit_gamma(u.threshold, u.mid, w, shape=None if len(u) >= 3 else pooled[stat])
            out.append(base | {"mean": a * s, "median": float(gamma.median(a, scale=s)),
                               "fit_rmse_cents": 100 * rmse, "shape": a})
        else:
            lam, rmse = fit_poisson(u.threshold, u.mid, w)
            out.append(base | {"mean": lam, "median": float(poisson.median(lam)),
                               "fit_rmse_cents": 100 * rmse, "shape": np.nan})
    for (stat, slug), p in df[df.kind == "scalar"].groupby(["stat", "slug"]):
        r = p.iloc[0]
        if r.mid is None or np.isnan(r.mid):
            continue
        out.append({"stat": stat, "slug": slug, "player": r.player, "team": r.team, "game": r.game,
                    "kalshi_player_id": r.kalshi_player_id, "rungs": 1, "median_width": r.width,
                    "source": "scalar", "mean": r.mid / r.unit, "median": np.nan,
                    "fit_rmse_cents": np.nan, "shape": np.nan})
    return pd.DataFrame(out)


def match_gsis(proj: pd.DataFrame, season: int) -> pd.Series:
    """nflverse gsis_id by team + jersey number (from the ticker slug) + surname."""
    import nflreadpy as nfl

    r = nfl.load_rosters(seasons=[season]).to_pandas()
    r = r[["team", "jersey_number", "last_name", "full_name", "gsis_id"]].dropna(subset=["gsis_id"])
    alias = {"LA": "LAR", "JAX": "JAC"}           # nflverse -> Kalshi team codes
    r["kteam"] = r.team.map(lambda t: alias.get(t, t))
    r["jersey"] = pd.to_numeric(r.jersey_number, errors="coerce")
    out = {}
    for row in proj[["slug", "team", "player"]].drop_duplicates().itertuples():
        m = re.search(r"(\d+)$", row.slug)
        cand = r[r.kteam == row.team]
        if m:
            cand = cand[cand.jersey == int(m.group(1))]
        last = row.player.split()[-1].lower() if row.player else ""
        hit = cand[cand.last_name.str.lower() == last]
        if len(hit) == 0 and len(cand) == 1:
            hit = cand
        out[row.slug] = hit.gsis_id.iloc[0] if len(hit) == 1 else None
    return proj.slug.map(out)
