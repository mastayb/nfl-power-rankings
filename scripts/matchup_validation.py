"""Do pre-game coverage matchup scores predict actual player outcomes? (2018-2025)

Run: python -u scripts/matchup_validation.py

Everything pre-game uses only EARLIER weeks of the same season. Weeks 5+.

Receivers (WR/TE with a prior target share >= 10%, active per snap counts):
  delta_share = (opp man rate - league man rate) x shrunk(share vs man - share vs zone)
  outcome     = actual share of team targets - prior share
Quarterbacks (prior 100+ dropbacks):
  delta_epa   = (opp single-high rate - league) x shrunk(EPA vs 1-high - EPA vs 2-high)
  outcome     = game EPA/dropback (situation-adjusted) - prior EPA/dropback
A slope of ~1 means the score predicts as well as its construction implies; 0 = nothing.
"Oracle" versions use the opponent's actual coverage mix in that game (upper bound).
Season-cluster bootstrap 90% intervals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from player_persistence import load as load_plays  # noqa: E402
from xo_round2 import boot, ols  # noqa: E402

CACHE = ROOT / "data" / "cache"
K_SHARE = 180.0     # team targets vs man for reliability ~0.5 (persistence test: r~0.38 at ~110)
K_QB = 475.0        # dropbacks vs single-high for reliability ~0.5 (r~0.24 at ~150, pooled seasons;
                    # season-demeaned r is ~0, see persistence_audit.py: the QB test is a null)
M_OPP = 60.0        # prior dropbacks for opponent tendency shrinkage
pd.set_option("display.width", 200)


def prior_cumsum(df, keys, cols):
    """Cumulative sums over strictly earlier weeks, per keys (season, entity)."""
    df = df.sort_values("week")
    c = df.groupby(keys)[cols].cumsum() - df[cols]
    return df[keys + ["week"]].join(c)


def active_receivers() -> pd.DataFrame:
    snaps = pd.read_parquet(CACHE / "snaps_2013_2025.parquet")
    ids = pd.read_parquet(CACHE / "player_ids.parquet").set_index("pfr_id").gsis_id
    s = snaps[(snaps.game_type == "REG") & (snaps.season >= 2018) & (snaps.offense_snaps > 0)
              & snaps.position.isin(["WR", "TE"])].copy()
    s["gsis_id"] = s.pfr_player_id.map(ids)
    alias = {"LA": "LA", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}
    s["team"] = s.team.map(lambda t: alias.get(t, t))
    return s.dropna(subset=["gsis_id"])[["season", "week", "team", "gsis_id"]].drop_duplicates()


def receiver_test(d: pd.DataFrame) -> pd.DataFrame:
    t = d[(d["pass"] == 1) & d.receiver_player_id.notna() & d.man.notna()].copy()
    t["m"] = (t.man == True).astype(float)  # noqa: E712
    t["z"] = 1 - t.m
    team = t.groupby(["season", "week", "posteam"]).agg(tt=("m", "size"), tm=("m", "sum"), tz=("z", "sum")).reset_index()
    pl = t.groupby(["season", "week", "posteam", "receiver_player_id"]).agg(
        pt=("m", "size"), pm=("m", "sum"), pz=("z", "sum")).reset_index()
    act = active_receivers().rename(columns={"team": "posteam", "gsis_id": "receiver_player_id"})
    pg = act.merge(pl, on=["season", "week", "posteam", "receiver_player_id"], how="left").fillna(
        {"pt": 0, "pm": 0, "pz": 0}).merge(team, on=["season", "week", "posteam"])
    pg = pg.sort_values("week")
    cols = ["pt", "pm", "pz", "tt", "tm", "tz"]
    pr = pg.groupby(["season", "posteam", "receiver_player_id"])[cols].cumsum() - pg[cols]
    pg = pg.join(pr.add_prefix("prior_"))
    dfn = d[(d.qb_dropback == 1) & d.man.notna()].assign(m=lambda x: (x.man == True).astype(float))  # noqa: E712
    dg = dfn.groupby(["season", "week", "defteam"]).agg(dm=("m", "sum"), dn=("m", "size")).reset_index()
    dg = dg.sort_values("week")
    dp = dg.groupby(["season", "defteam"])[["dm", "dn"]].cumsum() - dg[["dm", "dn"]]
    dg = dg.join(dp.add_prefix("prior_"))
    league = dfn.groupby("season").m.mean()
    opp = d[["season", "week", "posteam", "defteam"]].drop_duplicates()
    pg = pg.merge(opp, on=["season", "week", "posteam"]).merge(dg, on=["season", "week", "defteam"])
    lg = pg.season.map(league)
    pg["opp_man_prior"] = (pg.prior_dm + lg * M_OPP) / (pg.prior_dn + M_OPP)
    pg["opp_man_actual"] = pg.dm / pg.dn
    pg["share_prior"] = pg.prior_pt / pg.prior_tt
    raw = pg.prior_pm / pg.prior_tm - pg.prior_pz / pg.prior_tz
    pg["diff_shrunk"] = raw * pg.prior_tm / (pg.prior_tm + K_SHARE)
    pg["delta"] = (pg.opp_man_prior - lg) * pg.diff_shrunk
    pg["delta_oracle"] = (pg.opp_man_actual - lg) * pg.diff_shrunk
    pg["outcome"] = pg.pt / pg.tt - pg.share_prior
    pg["targets_delta"] = pg.delta * pg.tt
    return pg[(pg.week >= 5) & (pg.share_prior >= 0.10) & (pg.prior_tm >= 30) & (pg.tt >= 15)].dropna(
        subset=["delta", "outcome"])


def qb_test(d: pd.DataFrame) -> pd.DataFrame:
    q = d[(d.qb_dropback == 1) & d.passer_player_id.notna()].copy()
    q["one"] = (q.shell == "single-high").astype(float)
    q["two"] = (q.shell == "two-high").astype(float)
    q["e1"] = q.adj * q.one
    q["e2"] = q.adj * q.two
    g = q.groupby(["season", "week", "posteam", "defteam", "passer_player_id"]).agg(
        n=("adj", "size"), e=("adj", "sum"), n1=("one", "sum"), n2=("two", "sum"),
        s1=("e1", "sum"), s2=("e2", "sum")).reset_index().sort_values("week")
    cols = ["n", "e", "n1", "n2", "s1", "s2"]
    pr = g.groupby(["season", "passer_player_id"])[cols].cumsum() - g[cols]
    g = g.join(pr.add_prefix("p_"))
    dd = q.groupby(["season", "week", "defteam"]).agg(d1=("one", "sum"), dk=("shell", lambda s: s.isin(["single-high", "two-high"]).sum())).reset_index().sort_values("week")
    dp = dd.groupby(["season", "defteam"])[["d1", "dk"]].cumsum() - dd[["d1", "dk"]]
    dd = dd.join(dp.add_prefix("p_"))
    league = q[q.shell.isin(["single-high", "two-high"])].groupby("season").apply(
        lambda x: (x.shell == "single-high").mean(), include_groups=False)
    g = g.merge(dd, on=["season", "week", "defteam"])
    lg = g.season.map(league)
    g["opp_1h_prior"] = (g.p_d1 + lg * M_OPP) / (g.p_dk + M_OPP)
    g["opp_1h_actual"] = g.d1 / g.dk.replace(0, np.nan)
    base = g.p_e / g.p_n
    skill = (g.p_s1 / g.p_n1 - g.p_s2 / g.p_n2)
    w = np.minimum(g.p_n1, g.p_n2) / (np.minimum(g.p_n1, g.p_n2) + K_QB)
    g["skill_shrunk"] = skill * w
    g["delta"] = (g.opp_1h_prior - lg) * g.skill_shrunk
    g["delta_oracle"] = (g.opp_1h_actual - lg) * g.skill_shrunk
    g["outcome"] = g.e / g.n - base
    return g[(g.week >= 5) & (g.p_n >= 100) & (g.n >= 15) & (g.p_n1 >= 30) & (g.p_n2 >= 30)].dropna(
        subset=["delta", "outcome", "delta_oracle"])


def show(df, label, x, y="outcome"):
    b, r2, _ = ols(df, y, [x])
    ci = boot(df, y, [x], n=500)
    print(f"  {label:44} slope {b[x]:+.2f}  90% [{ci.loc[x, 0.05]:+.2f}, {ci.loc[x, 0.95]:+.2f}]  "
          f"n={len(df)}  SD(score)={df[x].std():.4f}")


def main() -> None:
    d = load_plays()
    print("=== Receivers: share of team targets vs man coverage ===")
    r = receiver_test(d)
    show(r, "pre-game (opp prior man rate)", "delta")
    show(r, "oracle (opp actual man rate that game)", "delta_oracle")
    print(f"  typical |pre-game delta|: {r.delta.abs().mean() * 100:.2f} pts of share = "
          f"{r.targets_delta.abs().mean():.2f} targets; 90th pct {r.targets_delta.abs().quantile(.9):.2f} targets")
    # are opponent man tendencies predictive of the game's actual man rate?
    print(f"  opp prior man rate vs actual that game: r = {r.opp_man_prior.corr(r.opp_man_actual):.2f}")

    print("\n=== QBs: EPA/dropback vs one-high / two-high ===")
    q = qb_test(d)
    show(q, "pre-game (opp prior 1-high rate)", "delta")
    show(q, "oracle (opp actual 1-high rate that game)", "delta_oracle")
    print(f"  typical |pre-game delta|: {q.delta.abs().mean():.3f} EPA/dropback "
          f"(~{q.delta.abs().mean() * 38:.2f} pts/game at 38 dropbacks); "
          f"opp prior vs actual 1-high rate r = {q.opp_1h_prior.corr(q.opp_1h_actual):.2f}")
    pd.concat([r.assign(test="receiver"), q.assign(test="qb")]).to_parquet(
        ROOT / "data" / "matchup_validation.parquet", index=False)


if __name__ == "__main__":
    main()
