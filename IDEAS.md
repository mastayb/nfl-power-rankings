# Ideas parking lot

## Tune-your-own power rankings site (future side project)

A public site with a small, neat UI where anyone can hand-tune the weight of
every input and watch the ranking update: Does résumé matter? Do markets?
QBR, EPA, success rate, time of possession, luck, close games...

- Backend: this repo's per-team, per-game features (market rating, résumé
  variants, EPA/success rate, fumble/INT luck, game control), refreshed weekly.
- Default weights = the empirically measured blend (scripts/calibrate_blend.py,
  scripts/resume_round1.py), so users start from "what the data says" and
  can see how far their own preferences move teams.
- Could show, live, how predictive the user's weighting would have been
  historically (same LOSO backtest), as a score next to their ranking.

## Team-specific ATS inefficiencies (future research, after the rankings model)

Hunch: league-wide ATS patterns don't hold (confirmed: scripts/historical_test.py
finds no ATS momentum, no league-wide "playing down"), but individual teams may
be mispriced for discrete runs. Ben's experience: ~15-3 betting one NBA team
that played to its opponent's level when the spread was > 5.

Data points as of 2026-10-02:
- TB has failed to cover 12 straight (2025 wk 10 -> 2026 wk 3). Longest in
  1999-2025 nflverse data; no team had a streak > 10, and 10+ happened once.
- Back-of-envelope: a fair coin over ~13k team-games predicts ~6 streaks of
  10+; we saw 1. Long ATS streaks look shorter than chance -> the market seems
  to correct runs. Needs a proper test (cover rate ~49%, pushes, offseason).

How to test without fooling ourselves:
- Pre-register the pattern definition (e.g. "team's ATS residual by spread
  bucket", "plays to opponent level") and detect it on games 1..n only, then
  score it strictly forward on games n+1.. (walk-forward, no peeking).
- Correct for the search: 32 teams x many windows x many pattern types will
  always produce some 15-3 runs. Calibrate with permutation/simulation of the
  same search on shuffled ATS outcomes.
- Regime/run detection: changepoint or hidden-Markov models on each team's
  ATS residual series; ask whether a detected regime persists out of sample.
- Measure against the line available when you'd bet (opener vs close if we
  can get openers) and against the -110 breakeven (52.4%).

## Xs and Os: "styles make the fight" (strategy-level analysis)

Goal: bridge the numbers crowd and the film crowd. Break the efficiency
numbers (EPA, success rate, explosive rate) down by scheme and personnel,
estimate which style-vs-style matchups actually matter, and test whether
they explain games beyond team quality. Team quality comes from the market
ratings; this layer explains *how* teams get there and where specific
matchups deviate.

Motivating example (Ben, 2026 wk 5): the Rams seemed to stay out of base
defense against the Bills' 13 personnel, limiting what Buffalo could do
with its pass-catching TEs. Not checkable yet: 2026 participation
(personnel/coverage) isn't published (see data table).

### Data (checked 2026-10-14)

| | 2016-21 | 2022-25 | 2026 |
|---|---|---|---|
| Participation (NGS): formation, off/def personnel, box, pass rushers, coverage type, man/zone, routes, time to throw, pressure, all 22 player IDs | yes | yes | no: `pbp_participation_2026` not released (404) |
| FTN charting: motion, play action, RPO, screen, QB out of pocket, box, blitzers, rushers, interception-worthy, catchable, contested, created reception, drops, read thrown | no | yes | yes (live) |
| Play-by-play: EPA, success, xpass/pass_oe, WP | yes | yes | yes |
| NGS player-week (separation, CPOE, YACOE, RYOE, time to throw); PFR advanced (pressures, blitzes, missed tackles, coverage allowed) | yes | yes | NGS yes / PFR through 2025 |

Not public: per-play coverage assignments (who covered whom) and raw tracking
(Big Data Bowl samples only); player grades (PFF, paywalled).

### Paid data options for live 2026 (checked 2026-10-14)

Since the NGS feed died in 2023, nflverse participation comes from FTN Data and is released only *after* each postseason, so 2026 personnel/coverage arrives free around February.

| Source | Gives | Price | Verdict |
|---|---|---|---|
| **PFF Pro** (Premium Stats Pro + API/CLI, `developer.pff.com/openapi.json`) | Player + team **grades** by facet (incl. pass/run blocking). **Aggregated** splits per player per game/season: vs **man / zone** (~50/56 cols), vs **blitz** (~86 cols), pressure/clean, depth/direction, passing concept, **alignment** (slot / wide / inline / box snaps), coverage matchup. Filter by week or game_id; CSV export + API. | ~$200/yr (verify at pff.com/subscribe) | Best-value add-on: grades (the film layer, esp. OL) + live man/zone and blitz splits. |
| PFF Pro: what it does **not** have | **No play-by-play rows; no personnel groupings (11/12/13) or formations** (zero mentions in the API spec); coverage only man vs zone, no Cover 1/2/3 shells. | | Can't answer "base vs 13 personnel" directly. Partial proxies: TE inline snaps, opponent man/zone and blitz rates per game. |
| FTN Data | Play-level charting incl. personnel, coverage, routes, motion, pressure (participation appears to be API tier) | CSV from $599; API from ~$3k/yr private, $5k commercial | Only route to live 2026 play-level personnel/coverage. Too pricey for a side project. |
| SIS DataHub | Web tool with personnel/formation filters and on/off splits | $99.99/mo, $749.99/yr | Good for one-off by-hand checks (e.g. the Rams-Bills question); no bulk data. |

Plan: build and validate on free nflverse 2016-2025; optionally add PFF Pro for grades + live man/zone/blitz splits.

**License rule (public repo):** nflverse is CC-BY-SA (credit FTN Data via nflverse, 2023+; NFL NGS via nflverse, 2022 and earlier). Paid pulls (PFF/FTN/SIS) go in a git-ignored folder and are never committed. Publish only derived results, and check each provider's terms first (PFF's redistribution terms weren't visible).

### Plan

1. **Style profiles.**
   - Offense: EPA / success / explosive rate by personnel (11/12/13/21), formation, motion, play action, RPO, screens.
   - Defense: the same by personnel (base/nickel/dime), coverage (C0-C4, man/zone), blitz rate, rusher count.
   - Coaching: early-down pass rate over expected (pbp `pass_oe`), fourth-down aggressiveness vs a decision model, pace.
2. **Shrinkage is the core, not an add-on.** Team x situation splits are small samples ("Rams vs 13 personnel" might be 15 snaps). Use empirical Bayes / a hierarchical model so every split is pulled toward league and team baselines by its sample size. No raw split tables.
3. **Matchup model.** Play EPA ~ offense quality + defense quality + style x style interactions (e.g. heavy personnel vs nickel, play action vs single-high, quick game vs blitz), all regularised. Outputs:
   - league-wide interaction effects in EPA/play and in points;
   - a per-game "style matchup adjustment" (points) from both teams' tendencies.
4. **Test it honestly.** Does the style adjustment explain game results relative to the closing spread, out of sample (leave-one-season-out)? Prior: real but small (~1 point in specific matchups), partly priced. A validated map of which matchups matter is still new public work.
5. **Player matchups (stretch).** On/off splits ("with this CB on the field") from participation player IDs, with the same shrinkage. Weaker than true matchups, which need tracking data.

### First milestone

2024-2025 style profiles plus the shrinkage model, built on participation + FTN + pbp, with a check that shrunk splits predict the second half of each season better than raw splits. Then the league-wide interaction model on 2022-2025. For 2026, a reduced live version from FTN + pbp tags only, until 2026 participation is released.

## Opponent-quality sensitivity: "flat-track bullies", "giant killers" and "steady" teams

Lower-dimensional cousin of the Xs and Os matchup model (Ben, 2026-10-14).
Instead of style-specific skill, ask whether a team's performance depends on
*opponent quality* in a team-specific way. Three archetypes:

- **Flat-track bully:** offense crushes bad defenses but underperforms against elite ones (steep slope vs defensive quality). Defensive version: dominates median offenses, gets exposed by elite ones.
- **Giant killer / matchup-proof:** the opposite, a flat slope; productive against elite opponents but never runs up the score on bad ones.
- **Steady vs volatile:** low vs high game-to-game variance around expectation (floor vs ceiling), separate from the slope.

### Model

Per game (or per play, aggregated), team offensive efficiency vs expectation:

    EPA_off(t, g) = a_t + b_t * DefQuality(opp(g)) + e,   Var(e) = s_t^2

- a_t: overall skill; b_t: team-specific sensitivity to opponent quality (league slope = 1 after opponent adjustment); s_t: consistency.
- Defense mirrored with offensive opponent quality.
- Opponent quality from a leave-team-out rating (market rating, or opponent-adjusted EPA excluding games against team t) to avoid circularity.
- Shrink b_t and s_t hierarchically: about 17 opponents per season means slopes are noisy.

### Tests (same discipline as H2)

1. **Persistence:** split-half correlation of b_t (opponents in weeks 1-9 vs 10+) and of s_t; also season to season for the same team (and same QB/play-caller, since coaching and QB continuity may carry the trait).
2. **Size:** persistent SD of b_t in points per game against a top-5 vs bottom-5 opponent.
3. **Market:** does the market already price it? Regress (closing line - aggregate ratings) on b_t x opponent quality; then results vs closing line (H5-style).
4. **Consistency vs totals:** do "steady" teams' games have tighter margin distributions? Links to the sigma-vs-total finding (no league-wide effect; this asks it per team).

### Expectations / power

- Prior: weak or no persistence for slopes. 17 games per season, with a narrow spread of opponent quality, give each slope a large standard error; the H2 personnel contexts showed the same problem.
- Most promising: multi-season pooling with QB and play-caller continuity, and play-level data to raise power (but the slope is still identified only by about 17 distinct opponents per season).
- Cheap to run on 1999-2025 game-level data (closing lines + results + nflverse EPA), so a quick test before investing.
