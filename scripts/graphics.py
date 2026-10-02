"""Social graphics (X, 1600x900): rankings, market vs results, playoff odds.

Run: python scripts/graphics.py [data/snapshots/<ts>.parquet]
Writes PNGs to reports/graphics/<season>-wk<NN>/.

Team colours are branding, not the identity channel: every mark also carries
the team's logo and name, so nothing depends on telling 32 hues apart.
Logos are cached from the URLs nflverse publishes (data/cache/logos/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SNAP_DIR = ROOT / "data" / "snapshots"
LOGOS = ROOT / "data" / "cache" / "logos"
W, H, DPI = 16, 9, 100                      # 1600 x 900 px

SURFACE = "#FAFAF7"
INK = "#1B1B1F"
INK_2 = "#55575E"
MUTED = "#8A8C93"
GRID = "#E4E4DF"
FONT = "Noto Sans"

plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# --- data -----------------------------------------------------------------------------

def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def load(snap_path: Path) -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl

    stem = snap_path.stem
    b = pd.read_parquet(snap_path.with_name(stem + "_stage_b.parquet")).set_index("team")
    sim = pd.read_parquet(snap_path.with_name(stem + "_sim.parquet")).set_index("team")
    teams = nfl.load_teams().to_pandas().set_index("team_abbr")
    df = pd.DataFrame({
        "nick": teams.team_nick.reindex(b.index),
        "conf": teams.team_conf.reindex(b.index),
        "color": pd.Series([bar_color(teams.loc[t, "team_color"], teams.loc[t, "team_color2"])
                            for t in b.index], index=b.index),   # align by team, not position
        "record": b.record, "rating": b.blend, "rank": b.blend_rank,
        "market": b.market, "resume": b.resume,
        "playoffs": sim.playoffs, "division": sim.division, "sb": sim.sb,
    })
    meta = {"week": int(b.blend_cutoff_week.iloc[0]) + 1,
            "date": pd.Timestamp(stem).strftime("%b %-d, %Y"),
            "season": int(stem[:4]), "n_sims": 10_000}
    return df, meta


def _lum(hex_color: str) -> float:
    rgb = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(rgb @ [0.2126, 0.7152, 0.0722])


def bar_color(primary: str, secondary: str) -> str:
    """Team primary colour, unless it is too faint on the light surface (< 3:1)."""
    surface = _lum(SURFACE)
    for c in (primary, secondary):
        if (surface + 0.05) / (_lum(c) + 0.05) >= 3.0:
            return c
    return INK


_logo_cache: dict[str, np.ndarray] = {}


def logo(team: str, px: int) -> np.ndarray:
    key = f"{team}-{px}"
    if key not in _logo_cache:
        im = Image.open(LOGOS / f"{team}.png").convert("RGBA")
        im.thumbnail((px * 2, px * 2), Image.LANCZOS)        # 2x for crispness
        _logo_cache[key] = np.asarray(im)
    return _logo_cache[key]


def place_logo(ax, team, x, y, px, transform=None):
    img = OffsetImage(logo(team, px), zoom=0.5)
    ab = AnnotationBbox(img, (x, y), xycoords=transform or ax.transData, frameon=False,
                        box_alignment=(0.5, 0.5), zorder=5)
    ax.add_artist(ab)


# --- chrome -----------------------------------------------------------------------------

def frame(title: str, subtitle: str, meta: dict):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.035, 0.935, title, fontsize=30, fontweight="bold", va="baseline")
    fig.text(0.035, 0.885, subtitle, fontsize=15, color=INK_2, va="baseline")
    fig.text(0.035, 0.03, f"Data: Kalshi prediction-market prices ({meta['date']}) and nflverse "
             f"results. Market + résumé blend; playoff odds from {meta['n_sims']:,} simulated seasons.",
             fontsize=11.5, color=MUTED, va="baseline")
    fig.text(0.965, 0.03, f"{meta['season']} · Week {meta['week']}", fontsize=11.5, color=MUTED,
             ha="right", va="baseline")
    return fig


def rounded_bar(ax, y, x0, x1, height, color, radius_px=4):
    """Horizontal bar from x0 (baseline) to x1 with a rounded data end only."""
    if x1 == x0:
        return
    bbox = ax.get_window_extent()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_per_px = (xlim[1] - xlim[0]) / bbox.width
    y_per_px = abs(ylim[1] - ylim[0]) / bbox.height
    r = min(radius_px * x_per_px, abs(x1 - x0) / 2)
    left, width = min(x0, x1), abs(x1 - x0)
    ax.add_patch(FancyBboxPatch((left, y - height / 2), width, height,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=y_per_px / x_per_px, fc=color, ec="none",
                                zorder=3))
    # square off the baseline end
    sq = min(r, width)
    sx = x0 if x1 > x0 else x0 - sq
    ax.add_patch(Rectangle((sx, y - height / 2), sq, height, fc=color, ec="none", zorder=3))


# --- 1. rankings ----------------------------------------------------------------------------

def rankings(df: pd.DataFrame, meta: dict, out: Path) -> Path:
    fig = frame(f"NFL Power Rankings · Week {meta['week']}",
                "Rating = points better or worse than an average team on a neutral field. "
                "Market prices plus on-field résumé.", meta)
    d = df.sort_values("rank")
    lo, hi = np.floor(d.rating.min()) - 1.5, np.ceil(d.rating.max()) + 1.5
    for col, part in enumerate([d.iloc[:16], d.iloc[16:]]):
        left = 0.035 + col * 0.485
        ax = fig.add_axes([left + 0.20, 0.10, 0.25, 0.74])
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(part) - 0.5, -0.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_yticks([])
        ticks = [t for t in range(int(lo), int(hi) + 1) if t % 4 == 0]
        ax.set_xticks(ticks, [f"{t:+d}" if t else "0" for t in ticks], fontsize=11)
        ax.tick_params(axis="x", length=0, pad=6)
        for t in ticks:
            ax.axvline(t, color=GRID, lw=1, zorder=0)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        fig.canvas.draw()
        ytr = ax.get_yaxis_transform()
        for i, (team, r) in enumerate(part.iterrows()):
            rounded_bar(ax, i, 0.0, r.rating, 0.56, r.color)
            ha, dx = ("left", 0.18) if r.rating >= 0 else ("right", -0.18)
            ax.text(r.rating + dx, i, f"{r.rating:+.1f}", va="center", ha=ha, fontsize=12.5,
                    color=INK_2, fontweight="bold", zorder=4)
            ax.text(-0.80, i, f"{int(r['rank'])}", transform=ytr, ha="right", va="center",
                    fontsize=15, fontweight="bold")
            place_logo(ax, team, -0.67, i, 34, transform=ytr)
            ax.text(-0.56, i, r.nick, transform=ytr, ha="left", va="center", fontsize=14.5)
            ax.text(-0.06, i, r.record, transform=ytr, ha="right", va="center", fontsize=12.5,
                    color=MUTED)
    path = out / "1_rankings.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --- 2. market vs results ---------------------------------------------------------------------

def market_vs_results(df: pd.DataFrame, meta: dict, out: Path) -> Path:
    fig = frame("Who does the market believe?",
                "Market rating vs. results so far (scoring margin adjusted for opponents). "
                "Both in points vs. an average team.", meta)
    ax = fig.add_axes([0.075, 0.12, 0.89, 0.71])
    xpad, ypad = 2.5, 1.2
    ax.set_xlim(df.resume.min() - xpad, df.resume.max() + xpad)
    ax.set_ylim(df.market.min() - ypad, df.market.max() + ypad)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.tick_params(length=0, labelsize=11)
    ax.set_xlabel("Results so far (higher = better)", fontsize=13, color=INK_2, labelpad=8)
    ax.set_ylabel("Market rating (higher = better)", fontsize=13, color=INK_2, labelpad=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
    ax.grid(color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x0 + 0.4, y1 - 0.35, "Market rates them higher than their results",
            fontsize=12.5, color=MUTED, va="top")
    ax.text(x1 - 0.4, y0 + 0.35, "Results ahead of the market", fontsize=12.5, color=MUTED,
            ha="right", va="bottom")
    # (text, label offset in data units); called-out teams are drawn last, on top
    callouts = {"HOU": ("Texans 0-3: market still above average", (3.0, 1.1)),
                "LV": ("Raiders 3-0: market unconvinced", (1.8, -1.4)),
                "LA": ("Rams 1-2: market's #1", (2.0, 0.4))}
    order = [t for t in df.index if t not in callouts] + [t for t in callouts if t in df.index]
    for team in order:
        r = df.loc[team]
        place_logo(ax, team, r.resume, r.market, 40)
    for team, (text, (dx, dy)) in callouts.items():
        if team not in df.index:
            continue
        r = df.loc[team]
        ax.annotate(text, (r.resume, r.market), xytext=(r.resume + dx, r.market + dy),
                    fontsize=12, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=2, shrinkB=15),
                    zorder=6)
    path = out / "2_market_vs_results.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --- 3. playoff odds ----------------------------------------------------------------------------

def playoff_odds(df: pd.DataFrame, meta: dict, out: Path) -> Path:
    fig = frame("Playoff odds", f"Share of {meta['n_sims']:,} simulated seasons, with full NFL "
                "tiebreakers. Bars: make the playoffs.", meta)
    for col, conf in enumerate(["AFC", "NFC"]):
        part = df[df.conf == conf].sort_values("playoffs", ascending=False)
        left = 0.035 + col * 0.485
        ax = fig.add_axes([left + 0.155, 0.10, 0.19, 0.71])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(len(part) - 0.5, -0.5)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        for t in (0.25, 0.5, 0.75, 1.0):
            ax.axvline(t, color=GRID, lw=1, zorder=0)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        fig.canvas.draw()
        ytr = ax.get_yaxis_transform()
        fig.text(left, 0.835, conf, fontsize=18, fontweight="bold", va="bottom")
        for x, label in [(1.20, "Division"), (1.50, "Super Bowl")]:
            ax.text(x, -0.95, label, transform=ytr, ha="center", va="bottom", fontsize=11.5,
                    color=MUTED)
        ax.text(0.5, -0.95, "Make playoffs", transform=ytr, ha="center", va="bottom",
                fontsize=11.5, color=MUTED)
        for i, (team, r) in enumerate(part.iterrows()):
            rounded_bar(ax, i, 0.0, max(r.playoffs, 0.004), 0.56, r.color)
            ax.text(r.playoffs + 0.02, i, f"{r.playoffs:.0%}", va="center", fontsize=12.5,
                    color=INK_2, fontweight="bold", zorder=4)
            place_logo(ax, team, -0.70, i, 32, transform=ytr)
            ax.text(-0.60, i, r.nick, transform=ytr, ha="left", va="center", fontsize=14)
            ax.text(-0.04, i, r.record, transform=ytr, ha="right", va="center", fontsize=12,
                    color=MUTED)
            ax.text(1.20, i, pct(r.division), transform=ytr, ha="center", va="center",
                    fontsize=12.5, color=INK_2)
            ax.text(1.50, i, pct(r.sb), transform=ytr, ha="center", va="center", fontsize=12.5,
                    color=INK_2)
    path = out / "3_playoff_odds.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def pct(p: float) -> str:
    if p < 0.01:                      # avoid "0%" from rounding a small positive chance
        return "<1%" if p > 0 else "0%"
    return f"{p:.0%}"


def main(snap_path: Path) -> None:
    df, meta = load(snap_path)
    out = ROOT / "reports" / "graphics" / f"{meta['season']}-wk{meta['week']:02d}"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (rankings, market_vs_results, playoff_odds):
        print("wrote", fn(df, meta, out).relative_to(ROOT))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
