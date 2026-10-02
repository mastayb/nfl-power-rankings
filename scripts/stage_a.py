"""Stage A: per-game implied home margin from the latest (or a given) snapshot.

Run: python scripts/stage_a.py [data/snapshots/<ts>.parquet]

Fits a normal and a key-number margin model to each game's spread ladder plus
moneyline, reports pooled sigma, and compares each game's fair line (median
margin) to the sportsbook spread nflverse carries (spread_line, home
perspective). Saves the expected margin `mean` for Stage B.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.ladder import (NormalModel, fit_game, fit_pooled_sigma,  # noqa: E402
                            game_observations, load_key_weights)
from ratings.schedule import game_key, load_schedule  # noqa: E402

SNAP_DIR = ROOT / "data" / "snapshots"

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def latest_snapshot() -> Path:
    snaps = sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)
    if not snaps:
        raise SystemExit("no snapshots; run scripts/snapshot.py")
    return snaps[-1]


def main(snap_path: Path) -> None:
    df = pd.read_parquet(snap_path)
    games = pd.read_parquet(snap_path.with_name(snap_path.stem + "_games.parquet"))
    games["key"] = games.event.map(game_key)
    gm = df[df.kind.isin(["game_winner", "spread"])].assign(key=lambda d: d.event.map(game_key))

    obs = {}
    for g in games.itertuples():
        obs[g.game_id] = game_observations(gm[gm.key == g.key], home=g.home_team, away=g.away_team)
    ladder_ids = [k for k, o in obs.items() if (o.source == "spread").any()]
    ml_only = [k for k in obs if k not in ladder_ids]

    models = {"normal": NormalModel(), "key": load_key_weights()}
    pooled, per_game, ml_fits = {}, {}, {}
    for name, model in models.items():
        sigma, _ = fit_pooled_sigma(model, {k: obs[k] for k in ladder_ids})
        pooled[name] = sigma
        per_game[name] = {k: fit_game(model, obs[k]) for k in ladder_ids}
        ml_fits[name] = {k: fit_game(model, obs[k], sigma=sigma) for k in ml_only}

    sched = load_schedule(int(games.game_id.str[:4].iloc[0])).set_index("game_id")
    rows = []
    for g in games.sort_values(["gameday", "game_id"]).itertuples():
        k = g.game_id
        has_ladder = k in ladder_ids
        fits = {n: (per_game if has_ladder else ml_fits)[n][k] for n in models}
        rows.append({
            "game_id": k, "neutral": g.neutral,
            "n_obs": fits["key"].n_obs,
            "mu_norm": fits["normal"].mu, "sd_norm": fits["normal"].sigma,
            "rmse_norm": fits["normal"].rmse_cents,
            "loc_key": fits["key"].mu, "sd_key": fits["key"].sigma,
            "mean_key": fits["key"].mean, "line_key": fits["key"].median,
            "rmse_key": fits["key"].rmse_cents, "max_key": fits["key"].max_abs_cents,
            "book": sched.loc[k, "spread_line"],
            "sigma_source": "ladder" if has_ladder else "pooled",
        })
    out = pd.DataFrame(rows)
    out["diff_key"] = out.line_key - out.book

    print(f"snapshot {snap_path.relative_to(ROOT)}")
    print(f"pooled sigma across {len(ladder_ids)} ladders: normal {pooled['normal']:.2f}, "
          f"key-number {pooled['key']:.2f}  (historical result-vs-close fit: see "
          f"data/key_number_weights.csv header)")

    fmt = {c: "{:.2f}".format for c in ["mu_norm", "sd_norm", "loc_key", "sd_key", "mean_key",
                                        "line_key", "diff_key"]}
    fmt |= {c: "{:.1f}".format for c in ["rmse_norm", "rmse_key", "max_key", "book"]}
    for label, part in [("Games with a spread ladder (mu, sigma fitted per game)",
                         out[out.sigma_source == "ladder"]),
                        ("Moneyline-only games (sigma fixed at pooled value)",
                         out[out.sigma_source == "pooled"])]:
        print(f"\n=== {label} ===")
        print(part.drop(columns="sigma_source").to_string(index=False, formatters=fmt))

    lad = out[out.sigma_source == "ladder"]
    print("\nmu vs sportsbook spread (key model compared on its fair line = median):")
    for label, part in [("ladder", out[out.sigma_source == "ladder"]),
                        ("ML-only", out[out.sigma_source == "pooled"])]:
        for col in ["mu_norm", "line_key"]:
            d = part[col] - part.book
            print(f"  {label:8} {col:9} bias {d.mean():+.2f}  median |diff| {d.abs().median():.2f}"
                  f"  within 0.5: {(d.abs() <= 0.5).sum()}/{len(d)}"
                  f"  within 1.0: {(d.abs() <= 1.0).sum()}/{len(d)}")
    lad = out[out.sigma_source == "ladder"]
    print(f"  ladder fit RMSE median: normal {lad.rmse_norm.median():.1f}c, "
          f"key-number {lad.rmse_key.median():.1f}c")

    dest = snap_path.with_name(snap_path.stem + "_stage_a.parquet")
    keep = out.rename(columns={"loc_key": "loc", "sd_key": "sigma", "mean_key": "mean",
                               "line_key": "fair_line", "rmse_key": "rmse_cents"})
    keep[["game_id", "neutral", "mean", "fair_line", "loc", "sigma", "sigma_source",
          "rmse_cents", "n_obs", "book"]].to_parquet(dest, index=False)
    print(f"\nwrote {dest.relative_to(ROOT)} (key-number model; Stage B uses `mean`)")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
