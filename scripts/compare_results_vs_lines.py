"""Results-only vs closing-lines-only ratings on the games played so far.

Run: python scripts/compare_results_vs_lines.py [season]

Both fits use the same games and model (home margin = r_home - r_away + H).
Holdout: fit on weeks before the latest played week, then predict the market
spread for every later game that nflverse has a line for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.linear import fit_margins, predict_margin  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402

CAP = 21  # blowout cap for the capped-results variant
TEAMS = sorted(NFLVERSE_TEAMS)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_all(games: pd.DataFrame) -> dict:
    return {
        "results": fit_margins(games, "result", TEAMS),
        "results_cap21": fit_margins(games, "result_cap", TEAMS),
        "lines": fit_margins(games, "spread_line", TEAMS),
    }


def main(season: int) -> None:
    s = load_schedule(season)
    # nflverse: result = home - away score; spread_line > 0 means home favoured.
    s["result_cap"] = s["result"].clip(-CAP, CAP)
    played = s[s.result.notna() & s.spread_line.notna()]
    last_week = int(played.week.max())
    print(f"Season {season}: {len(played)} played games with a closing line "
          f"(weeks {int(played.week.min())}-{last_week})")

    # --- in-sample fit on everything played ---------------------------------
    fits = fit_all(played)
    tab = pd.DataFrame({
        "r_lines": fits["lines"].ratings, "se_lines": fits["lines"].se,
        "r_results": fits["results"].ratings, "se_results": fits["results"].se,
        "r_cap21": fits["results_cap21"].ratings,
    })
    tab["diff"] = tab.r_results - tab.r_lines
    tab = tab.sort_values("r_lines", ascending=False)
    pd.set_option("display.width", 160)
    print("\nRatings (points vs average team, neutral field), sorted by lines fit:")
    print(tab.to_string(float_format=lambda x: f"{x:6.1f}"))

    print("\n            HFA (se)      resid SD   median team se   SD of ratings")
    for name, f in fits.items():
        print(f"{name:13} {f.hfa:5.2f} ({f.hfa_se:4.2f})   {f.resid_sd:6.2f}    "
              f"{f.se.median():6.2f}           {f.ratings.std():5.2f}")

    c = tab[["r_lines", "r_results", "r_cap21"]].corr()
    print(f"\ncorr(lines, results) = {c.loc['r_lines', 'r_results']:.2f}   "
          f"corr(lines, cap21) = {c.loc['r_lines', 'r_cap21']:.2f}")

    # --- holdout: fit through week L-1, predict later lines ------------------
    train = played[played.week < last_week]
    test = s[(s.week >= last_week) & s.spread_line.notna()]
    hfits = fit_all(train)
    print(f"\nHoldout: fit weeks {int(train.week.min())}-{last_week - 1} "
          f"({len(train)} games), predict market spread for {len(test)} games in "
          f"weeks {last_week}-{int(test.week.max())}")
    hfa_only = fits["lines"].hfa * ~test.neutral.to_numpy(bool)
    print(f"  {'HFA only (no team info)':26} RMSE {rmse(hfa_only, test.spread_line):5.2f}")
    for name, f in hfits.items():
        pred = predict_margin(f, test)
        print(f"  {name:26} RMSE {rmse(pred, test.spread_line):5.2f}   "
              f"(HFA {f.hfa:4.1f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
