# Market + Résumé Power Rankings, entering Week 6

These rankings combine two things: what betting markets think of each team, and what each team has done on the field. No expert opinions or hand adjustments go in.

| # | Team | Record | Rating | Win % vs. avg | Market rank | Résumé rank | Playoffs | Super Bowl |
|--:|:--|:-:|--:|--:|--:|--:|--:|--:|
| T-1 | Rams | 2-3 | +5.0 | 66% | 1 | 19 | 66% | 9% |
| T-1 | Bills | 4-1 | +5.0 | 66% | 2 | 10 | 88% | 10% |
| 3 | Jaguars | 4-1 | +4.7 | 65% | 6 | 1 | 83% | 7% |
| 4 | Chiefs | 4-0 | +4.3 | 64% | 5 | 3 | 85% | 9% |
| 5 | Ravens | 4-1 | +3.9 | 63% | 4 | 14 | 80% | 9% |
| 6 | 49ers | 5-0 | +3.8 | 62% | 7 | 5 | 87% | 9% |
| 7 | Seahawks | 3-2 | +3.4 | 61% | 3 | 17 | 62% | 6% |
| 8 | Patriots | 3-2 | +3.1 | 60% | 11 | 2 | 68% | 4% |
| 9 | Bears | 3-2 | +2.5 | 58% | 9 | 12 | 54% | 4% |
| 10 | Bengals | 3-2 | +2.2 | 57% | 12 | 7 | 60% | 3% |
| 11 | Texans | 1-4 | +1.7 | 56% | 8 | 22 | 31% | 2% |
| 12 | Lions | 3-2 | +1.5 | 55% | 14 | 13 | 61% | 4% |
| 13 | Vikings | 5-0 | +1.4 | 55% | 16 | 4 | 78% | 4% |
| 14 | Broncos | 3-2 | +1.3 | 54% | 13 | 18 | 62% | 3% |
| 15 | Cowboys | 2-3 | +1.1 | 54% | 10 | 24 | 46% | 3% |
| 16 | Packers | 3-2 | +0.3 | 51% | 17 | 8 | 37% | 2% |
| 17 | Panthers | 2-2 | +0.1 | 50% | 19 | 6 | 43% | 2% |
| 18 | Falcons | 2-3 | -0.1 | 50% | 21 | 9 | 34% | 1% |
| 19 | Steelers | 3-2 | -0.4 | 49% | 22 | 11 | 34% | <1% |
| 20 | Eagles | 2-3 | -0.6 | 48% | 15 | 25 | 38% | 2% |
| 21 | Colts | 2-3 | -1.2 | 46% | 20 | 23 | 28% | 1% |
| 22 | Buccaneers | 1-4 | -1.3 | 46% | 26 | 15 | 19% | <1% |
| 23 | Raiders | 3-2 | -1.5 | 45% | 24 | 20 | 32% | <1% |
| T-24 | Saints | 1-4 | -2.2 | 43% | 23 | 27 | 23% | <1% |
| T-24 | Chargers | 0-5 | -2.2 | 43% | 18 | 29 | 7% | <1% |
| 26 | Browns | 3-2 | -3.1 | 40% | 28 | 16 | 24% | <1% |
| 27 | Commanders | 1-3-1 | -3.6 | 39% | 25 | 30 | 17% | <1% |
| 28 | Jets | 2-3 | -4.1 | 37% | 30 | 21 | 14% | <1% |
| 29 | Giants | 3-1-1 | -4.3 | 36% | 27 | 28 | 32% | <1% |
| 30 | Cardinals | 1-4 | -6.4 | 30% | 29 | 31 | 3% | <1% |
| 31 | Titans | 0-5 | -7.1 | 28% | 31 | 32 | 2% | <1% |
| 32 | Dolphins | 0-5 | -7.5 | 27% | 32 | 26 | 1% | <1% |

**How to read it**

- **Rating:** points better or worse than an average team on a neutral field.
- **Win % vs. avg:** chance to beat an average team on a neutral field.
- **Market rank / Résumé rank:** where each half alone would place the team.
- **Playoffs / Super Bowl:** share of 10,000 simulated seasons.

**How it works**

1. **Market.** I use Kalshi prediction-market prices for the next two weeks of games and for every team's season win total, plus closing spreads from earlier weeks. One rating per team is fitted to all of these prices together.
2. **Résumé.** I rate each team on its scoring margins, adjusted for opponent strength. Recent games count more, and blowouts are capped at 21 points.
3. **Blend.** I tested the mix on 27 seasons of results. The résumé's share grows as games pile up: about 8% two weeks ago, about 20% now, and about 38% by Week 12. Luck (fumble recoveries and interceptions above normal rates) is now being phased out.
4. **Playoff odds.** I simulate the season with full NFL tiebreakers. On real 2020–2025 results, the code matched every playoff field and #1 seed.

**What stands out**

- **Rams tied for #1 at 2-3.** Markets rate them the best team in the league. Their opponent-adjusted results rank only 19th.
- **Jaguars #3 with the league's best résumé.** Markets agree they are good (6th), and their results push them higher.
- **Vikings #13 at 5-0.** All three close games went their way. Markets rank them 16th, and they face a much harder schedule from here.
- **Texans #11 at 1-4.** They are 0-3 in one-score games, and markets still rate them 8th.
- **Commanders: the luckiest team in the league, and still 1-3-1.** Fumble luck has been worth about 12 points to them so far.

**Limits**

- This does not beat the spread. Résumé added nothing once the closing spread was known.
- Injuries count only as far as markets price them.
