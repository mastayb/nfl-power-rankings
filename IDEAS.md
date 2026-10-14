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
