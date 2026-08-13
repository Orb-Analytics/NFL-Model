# Orb NFL Model

Two things live in this repo: the research/backtesting pipeline that
validated the model (everything below "Structure"), and the live weekly
scoring path that publishes real picks to
[orb-analytics-web](https://github.com/Orb-Analytics/orb-analytics-web)'s
predictions page. Start with **Live scoring & site integration** below if
you're trying to get this week's picks live -- the rest of this README is
the backtesting methodology that justified the model this repo now runs.

## Live scoring & site integration

`orb-analytics-web`'s predictions page is fully static: it fetches
`predictions.json` from this repo's `main` branch and renders whatever's
there (see that repo's README, "Predictions Integration"). There's no API
or deploy step on the site side -- committing an updated `predictions.json`
here IS the integration.

**Weekly flow** (mirrors `Orb-Analytics/MLB-Model`'s daily ETL pattern,
adapted to a weekly NFL cadence):

```
python scripts/01_fetch_historical.py       # refresh raw data, including the current season
python scripts/25_live_weekly_scoring.py    # score the next unplayed week -> data/processed/live_picks.csv
python scripts/26_write_predictions_json.py # live_picks.csv -> predictions.json
git add predictions.json && git commit -m "Weekly NFL predictions" && git push
```

`.github/workflows/run-weekly.yml` runs exactly this sequence automatically
(default: Tuesday mornings) and pushes the result -- see that file for the
schedule and how to trigger it manually from the Actions tab.

**Confirmed against a real live Novig NFL preseason slate (Aug 2026)**.
Novig's spread-market format turned out to differ from the MLB-by-analogy
guess this client shipped with originally:

- Moneyline market description is the bare team abbreviation (e.g. `"PIT"`).
- Totals market description is `"{AWAY} @ {HOME} t{NUMBER}"`, matching MLB.
- **Spread market description is `"{HOME_ABBR} {home_team's_own_signed_spread}"`**
  (e.g. `"JAX -7.5"`, `"NE +2.5"`) -- NOT `"AWAY @ HOME s-3.5"` as originally
  guessed. Confirmed against every game in a real debug dump: the consensus
  spread market (`is_consensus=True`) is always labeled with the home
  team's own line. `spread_line` (nflverse convention, positive = home
  favored) is one sign flip from that number.
- Event `description` uses full team display names (e.g. "Cleveland Browns
  @ Jacksonville Jaguars"), not abbreviations -- `novig_client.py` matches
  events by full name via a hardcoded `NFL_TEAM_FULL_NAMES` table, not by
  scanning for abbreviation substrings.
- One team abbreviation mismatch found: Novig uses `"WSH"` for Washington;
  nflverse uses `"WAS"`. Recorded in `config.NOVIG_TEAM_ABBR_MAP`, though
  the full-name-based event matching sidesteps needing it for that lookup
  specifically.
- Some consensus markets can have one side's price still unposted
  (`None`) -- `novig_client.py` skips those games rather than writing a
  half-populated row; they'll show up as unmatched in the "Matched N of M"
  summary it prints.

`_extract_novig_spread()` and the full event-matching path were both
tested against synthetic reconstructions of the real debug output (see
git history / test output) before being trusted, since this sandbox has no
network access to Novig itself. Still worth a final sanity check: run
`python scripts/novig_client.py --debug` once games are close to kickoff
and eyeball a few events to confirm nothing about the format has changed.

`scripts/25_live_weekly_scoring.py` fits FINAL production models (logit +
XGBoost + Naive Bayes) on every completed game in `training_set.csv`
(no walk-forward split -- that's for validating the method, not for live
use), scores the next unplayed week, and applies the exact rule validated
in `24_final_combined_rule.py`: 3-way agreement required, every underdog
kept, favorites kept only if edge <= `config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD`.
Same caveat as `24`: that threshold came from a backtest decile boundary,
not a nested walk-forward re-derivation -- update it if a more rigorous
version gets built later.

`scripts/26_write_predictions_json.py` formats the week's picks into the
schema `orb-analytics-web` expects: `pick` is the picked team's own posted
spread (e.g. `"Chiefs -3.5"`), `confidence` is the 3-model average
probability for the picked side, `line` is the picked side's American odds.

## Structure

```
NFL-Model/
  config.py                    # every tunable knob lives here
  .github/workflows/
    run-weekly.yml              # scheduled live scoring + predictions.json publish
  scripts/
    01_fetch_historical.py     # pulls raw data from nflreadpy -> data/raw/
    build_features.py          # raw predictor intake: rolling/EWMA, produced/allowed
    feature_engineering.py     # derived features: differentials, matchups, context
    02_build_training_set.py   # raw parquet -> training_set.csv + feature_manifest.csv
    03_prune_features.py       # training_set.csv -> training_set_pruned.csv + prune_log.csv
    04_feature_analysis.py     # univariate stats + multicollinearity flags vs. home_cover
    05_train_baseline_model.py # relevance filter + dedup + chronological holdout + XGBoost
    06_train_curated_model.py  # 12 hand-picked features -- logistic regression + XGBoost
    07_walk_forward_validation.py  # same 12 features, expanding-window multi-season test
    08_edge_based_evaluation.py    # market-price-aware pick selection, same walk-forward folds
    09_consensus_edge_walk_forward.py  # agreement gate AND edge gate combined
    10_season_backtest_report.py   # any picks file -> week-by-week record/units + running total
    11_edge_calibration_check.py   # does bigger edge actually mean more accurate? (deciles + correlation)
    12_confidence_calibration_check.py  # does the model's OWN confidence (no market) predict accuracy?
    13_pick_type_breakdown.py      # accuracy/units/ROI by home/away x favorite/underdog
    14_pick_type_edge_breakdown.py # does edge matter WITHIN favorite/underdog groups specifically?
    15_combined_rule.py            # underdogs unfiltered + low-edge favorites only, from 13+14's findings
    16_market_base_rate_check.py   # model-free: do underdogs really cover more often historically?
    17_home_favorite_edge_breakdown.py  # is 14's favorite-edge finding actually home-favorite-specific?
    18_combined_rule_v2.py         # 15's rule refined with separate home/away favorite thresholds
    19_positive_edge_rule.py       # single edge floor applied to EVERY pick (favorite + underdog)
    20_independent_edge_rule.py    # separate favorite/underdog edge floors, swept as a 2D grid
    21_three_way_consensus.py      # adds a Naive Bayes model, requires 3-way agreement instead of 2
    22_edge_only_breakdown.py      # no consensus gate -- bet the higher-edge side on every game, bucket by edge
    23_three_way_edge_aligned.py   # 3-way agreement AND the 3-model-average edge must point the same way
    24_final_combined_rule.py      # 3-way consensus + low-edge favorite filter, everything found so far combined
    novig_client.py            # LIVE Novig GraphQL client (spread markets, NFL) -- confirmed format, see above
    25_live_weekly_scoring.py  # fits final models on all history, scores the next unplayed week
    26_write_predictions_json.py  # live_picks.csv -> predictions.json for the site
    feature_utils.py           # shared get_predictor_columns()/evaluate()/odds+edge helpers, used by 04-14
  data/
    raw/                       # cached nflreadpy pulls (parquet)
    processed/                 # training_set.csv is the model's source of truth
  predictions.json             # published to main branch weekly; this is what the site reads
```

## Setup

```
pip install -r requirements.txt
python scripts/01_fetch_historical.py
python scripts/02_build_training_set.py
```

This produces `data/processed/training_set.csv` (source of truth for
training) and a `training_set_snapshot.xlsx` (for eyeballing only —
never hand-edit this file and feed it back in; regenerate it instead).

## Design decisions worth knowing about

**Rolling stats, not season-to-date.** Every `*_produced`/`*_allowed` stat
gets a trailing N-game rolling mean and an EWMA (config: `ROLLING_WINDOW_GAMES`,
`EWMA_HALFLIFE_GAMES`).

**No leakage.** Every stat is `.shift(1)`'d per team before being rolled, so
a team's own game that week is never part of its own pre-game features.

**Cross-season EWMA blending.** The EWMA is computed on each team's
continuous game log, without resetting at season boundaries. That's what
makes Week 1 features mostly reflect last season's play (since there's no
current-season data yet) and taper to mostly-current-season by Week 5-6 —
no separate hand-written blending formula needed.

**Offense vs. defense via opponent join.** `nflreadpy.load_team_stats()`
returns each team's own offensive production per week, not what it
allowed. `build_features.py` derives "allowed" stats by joining each
team's game to its opponent's same-game row. If nflreadpy's schema changes
column names, `_pick_team_column()` and `_stat_columns()` are the two
places to check first.

**`spread_line` is (most likely) the closing line.** nflverse's data
dictionary doesn't explicitly say "closing" or "opening," but it's
understood to be the market's final number. Since the model runs early in
the week, `spread_line` from this historical set is fine for training and
backtesting, but the live path should feed the model whatever line Novig
shows at run time — same feature slot, different source, see next section.

**One shared `build_features.py` for train and live.** This module doesn't
care whether you feed it 10 years of history or one upcoming week of
schedule/odds data — same functions, same logic. That's deliberate: it's
the single best defense against train/serve skew (the model behaving
differently live than it did in backtesting because the live feature
pipeline quietly diverged from the training one).

## Maximizing predictors, then pruning deliberately

Per design intent, `build_features.py` doesn't hand-pick which nflreadpy
columns matter -- it auto-detects every numeric column in `load_team_stats()`
and rolls/EWMAs all of them (produced + allowed, home + away). `config.py`'s
`INCLUDE_PFR_ADVSTATS` also pulls in Pro Football Reference's advanced stats
(pressure rate, time to throw, yards before/after contact, broken tackles,
etc., 2018+) via `merge_extra_sources()`, which aggregates any extra
player-level source to team-week grain and merges it in -- so those columns
get swept into the same rolling/EWMA machinery automatically. To bring in
another nflreadpy source later (snap counts, injuries, officials, etc.), add
it to `load_extra_sources()` in `02_build_training_set.py` the same way.

**Rate stats, not just totals.** `load_team_stats()` only returns counting
totals -- `passing_epa` is EPA summed across every attempt that game, not a
per-play rate. Nearly every raw counting stat has this problem, not just
EPA: yards, TDs, sacks, INTs, etc. all scale with how many plays got run
that game (pace/game script). A team with 45 dropbacks racks up more total
EPA, yards, and TDs than an efficient 25-play day even if the second team
was better per play -- and rolling the raw per-game total doesn't fix this,
since the pace confound is baked into every individual game before rolling
ever sees it.

`build_features.add_rate_stats()` derives ~25 rate stats (EPA/play,
completion %, catch rate, yards/attempt, TD rate, INT rate, sack rate, and
their defensive counterparts) at the per-game level -- BEFORE rolling,
since rolling raw totals and dividing afterward would let high-volume games
dominate the average. New rate columns use the same `{name}_produced` /
`{name}_allowed` suffix convention as everything else, so they flow through
rolling, differentials, and matchup features automatically.

One subtlety worth understanding if you extend `config.RATE_STAT_DEFINITIONS`:
each entry has a `pairing` (default `"same"`). Offense-side stats (passing/
rushing/receiving) use `"same"` -- both the numerator and denominator come
from the same team's own game (e.g. `passing_yards_allowed` divides by
`attempts_allowed`, both describing the opponent). Defense-side stats
(`def_sacks`, `def_interceptions`, etc.) use `"cross"` -- the numerator is
this team's own defensive production, but the meaningful denominator is the
OPPONENT's snap count that game (how many dropbacks did this defense
actually face). Getting that backwards -- dividing a team's own defensive
sacks by its own offense's attempts -- would silently produce a nonsense
rate; this was verified directly (hand-computed a single game's sack rate
and confirmed the pipeline divides by the opponent's attempts, not the
defense's own team's attempts). Definitions are skipped gracefully (with a
printed warning) if the underlying raw columns aren't present.

`feature_engineering.py` then layers derived features on top:
- **Differentials** (`diff_X = home_X - away_X`) for every rolled produced/
  allowed stat -- restricted to that naming pattern specifically so it can't
  accidentally diff a label or score column and leak the outcome.
- **Matchup features** pairing a team's offensive tendency with the
  opponent's tendency to allow that same stat.
- **Schedule context** (rest-day differential, short week flags, dome flag).

Every run of `02_build_training_set.py` also writes `feature_manifest.csv`:
one row per candidate predictor with % non-null and a constant-column flag,
so you have a concrete list to prune from rather than guessing. The script
also prints any constant or >50%-null columns it finds on each run.

### Decision: full 2010-2025 history, PFR features null before 2018

`SEASONS` (config.py) covers 2010-2025, but `PFR_ADVSTATS_MIN_SEASON` is
2018 -- PFR's advanced stats (pressure rate, yards before/after contact,
broken tackles, etc.) don't exist before then. Rather than truncating all
16 seasons down to match PFR's shorter window, the deliberate choice here
is to keep full history and let PFR-derived columns be NaN for 2010-2017.

This means: **the model needs to handle missing values natively** (e.g.
XGBoost, LightGBM, or any tree-based classifier with built-in NaN support).
If a future iteration switches to something that can't handle NaN directly
(plain sklearn LogisticRegression, etc.), those PFR columns will need
either explicit imputation or a `season >= 2018` filter applied at that
point -- don't assume every predictor is populated for every row.

## Pruning (`03_prune_features.py`)

Run after `02_build_training_set.py`:

```
python scripts/03_prune_features.py
```

This drops two categories of column, and only these two -- it's deliberately
conservative, since correlation filtering and model-driven feature
importance need an actual model in the loop and belong in the next step,
not here:

1. **Constant columns** (zero variance -- e.g. `fg_missed_0_19`, since teams
   essentially never miss a field goal from inside the 20).
2. **Columns sparse for an unexplained reason.** A column over
   `SPARSE_NULL_THRESHOLD_PCT` (50%) null gets dropped UNLESS it's a
   PFR-derived column whose nulls are fully explained by the pre-2018
   coverage gap above -- those get re-checked against
   `SPARSE_NULL_THRESHOLD_PCT_WITHIN_PFR_ERA` restricted to `season >= 2018`
   only. A PFR column that's well-populated since 2018 is kept even though
   it looks >50% null across the full history; a PFR column that's STILL
   sparse within its own coverage era gets dropped as genuinely broken.

Output:
- `training_set_pruned.csv` -- the pruned set.
- `prune_log.csv` -- every candidate column, whether it was dropped, and
  why. Worth a skim after every run to make sure nothing surprising got cut
  (or kept).

## Feature analysis (`04_feature_analysis.py`)

Run after pruning:

```
python scripts/04_feature_analysis.py
```

Target is `home_cover` (1 if the home team covered `spread_line`, else 0).
`config.OUTCOME_COLUMNS` excludes every alternate representation of that
outcome (`home_score`, `away_score`, `margin`, `home_win`, `away_win`,
`away_cover`) from ever being treated as a predictor -- these would trivially
"predict" the target because they ARE the target, restated.

This is two passes, not one:

1. **Univariate**: every remaining predictor, one at a time, against
   `home_cover` -- point-biserial correlation (the correct correlation type
   for continuous-vs-binary), a p-value, and an AUC (rank-based, via
   Mann-Whitney U, so it doesn't assume a linear relationship). Output:
   `feature_univariate_stats.csv`, sorted by p-value.

   With ~2,600 columns tested individually, some will look "significant" by
   chance alone -- the script prints how many clear p<0.05 and p<0.01, but
   treat this as a screening/ranking tool, not proof. A stricter cutoff or
   a multiple-comparisons correction (Benjamini-Hochberg) is worth applying
   before trusting any single result here.

2. **Multicollinearity, deliberately scoped down.** A full pairwise
   correlation matrix (or true VIF) across all ~2,600 pruned columns would
   mostly surface the structural redundancy built into this pipeline on
   purpose: `diff_X = home_X - away_X` and `matchup_X = (produced +
   allowed) / 2` are exact deterministic functions of the raw
   produced/allowed columns, not independently-arising correlation. Running
   real VIF against the full set hits exact linear dependencies and either
   fails (singular matrix) or reports infinite VIF for large blocks of
   columns. Instead, this script takes the top `TOP_N_FOR_CORRELATION_CHECK`
   (300) columns by univariate significance and flags pairs among THOSE
   above `HIGH_CORRELATION_THRESHOLD` (0.85). Output:
   `feature_high_correlation_pairs.csv` -- for each flagged pair, keep one,
   drop the other, rather than feeding both to the model.

Neither pass accounts for predictors interacting with each other -- that's
what the actual model (next step: a regularized or tree-based classifier)
is for. This script's job is to cut down "2,600 columns" to "a shortlist
worth actually modeling with," not to be the final word on any one feature.

## Baseline model (`05_train_baseline_model.py`)

Run after pruning (doesn't require 04 to have run first, though reading its
output first is a good sanity check on what to expect):

```
python scripts/05_train_baseline_model.py
```

Four deliberate choices here, each addressing something discussed earlier:

1. **Chronological holdout, not random.** `config.TEST_SEASONS` (default:
   [2024, 2025]) is held out entirely; everything before trains the model.
   A random split would let the model train on games from the same stretch
   of a season it's tested on, leaking information a real weekly deployment
   would never have at prediction time.

2. **Relevance pre-filter BEFORE dedup** (`filter_by_relevance`). This
   exists because of a concrete failure that showed up running this against
   real data: with ~3,000 candidate columns, correlation-dedup alone only
   cut it down to ~1,800 -- still enough for the model to hit 0.97 train AUC
   while scoring 0.48 AUC (worse than a coin flip) on the holdout. That's
   overfitting, not signal -- dedup removes near-DUPLICATE columns, not
   irrelevant ones. `filter_by_relevance` ranks columns by `|correlation|`
   with `home_cover` on TRAIN data only and keeps just the top
   `MAX_FEATURES_BEFORE_DEDUP` (default 150), forcing the feature count into
   a range the sample size can actually support. Verified directly: the
   exact same overfitting failure was reproduced synthetically (3,000
   injected noise columns against ~3,500 training rows) and confirmed fixed
   -- holdout AUC went from 0.48 (worse than random) to 0.75 once this
   filter was added.

3. **Correlation-based dedup**, applied after the relevance filter, across
   whatever survived it (not just the top 300 from `04_feature_analysis.py`).
   Columns are ranked by `|correlation|` with `home_cover` (computed on
   TRAIN data only), then greedily kept unless too correlated
   (`CORRELATION_DEDUP_THRESHOLD`, default 0.90) with something already
   kept. This is what actually thins out the diff_X/matchup_X/
   produced-allowed structural duplication before a model ever sees it.

4. **Deliberately conservative XGBoost hyperparameters**
   (`config.XGB_PARAMS`) -- shallow trees, heavy L1/L2 regularization, row/
   column subsampling. Even after both filters above, there's still more
   features than you'd want relative to a few thousand training rows: an
   unregularized model will happily memorize noise.

Evaluation compares the model against two naive baselines on the holdout,
not just its own accuracy in isolation: always predicting the majority
class, and always predicting the training data's base rate. **Given how
weak the univariate signal looked in step 4 (roughly what pure chance would
produce across ~2,600 tests), don't be surprised if the model doesn't
clearly beat these baselines.** That would be a legitimate, useful finding
about how hard this target is against an efficient market -- not proof
the pipeline is broken. The script prints an explicit note if holdout log
loss is worse than the base-rate baseline, or if AUC is only marginally
above 0.5.

A synthetic test with 50 injected pure-noise columns showed something worth
knowing before trusting this on real data: several noise columns still
picked up importance scores comparable to genuine signal, purely by chance,
given how many more columns there are than training rows. Treat
`baseline_feature_importance.csv`'s top-20 as leads to sanity-check against
football intuition, not as settled fact -- and consider pre-filtering with
step 4's univariate results if noise columns are dominating the ranking.

Output:
- `baseline_model.json` -- the trained XGBoost model.
- `baseline_model_metrics.json` -- train/holdout/baseline metrics (accuracy,
  AUC, log loss, Brier score).
- `baseline_feature_importance.csv` -- every kept feature, ranked.
- `baseline_kept_features.csv` -- exactly which columns survived dedup.

**What actually happened running this against real data**, worth recording
here rather than losing it in chat history: the first run picked up 1,816
columns via dedup alone (dedup only removes near-duplicates, not irrelevant
columns), and the model hit 0.97 train AUC while scoring 0.48 AUC (WORSE
than a coin flip) on the holdout -- pure memorization. That's what
motivated adding `filter_by_relevance()` (see `MAX_FEATURES_BEFORE_DEDUP`
above). After that fix, holdout AUC came back to 0.51 (no longer
anti-predictive, but also not clearly beating the baselines) -- consistent
with 04's finding that individual predictive power across this column set
is close to indistinguishable from chance. See `06_train_curated_model.py`
for the follow-up test this motivated.

## Curated model (`06_train_curated_model.py`)

Run after `02_build_training_set.py` (reads the FULL `training_set.csv`,
not the pruned or dedup'd output -- this hand-picked list doesn't depend on
what 03 or 05 decided to keep):

```
python scripts/06_train_curated_model.py
```

This exists because of what 05 found: correlation-based selection run
against ~3,000 columns with near-zero true effect sizes ended up picking
features that fit the training seasons without holding up on the holdout.
The concern is that when true signal is this weak, a purely statistical
selection process is fishing in noise. This script tests the opposite
philosophy -- `config.CURATED_FEATURES`, a short list (12) of things a
football person would actually expect to matter: `spread_line` and
`abs_spread_line` (spread magnitude, independent of which side is favored),
matchup-blended EPA/play (offense, passing, rushing, receiving), completion
%, sack rates (offense and defense), yards/carry, rest-day differential,
and dome. Deliberately built from base `team_stats`-derived rate stats only (no
PFR columns), so every feature has full coverage across all of 2010-2025 --
no pre-2018 NaN gaps to work around, and a genuinely independent test
rather than just a smaller random slice of the same PFR-heavy columns.

**Matchup-based, not produced/allowed-based (as of the current version).**
Earlier versions paired each team's own offense trend with its own defense
trend as two separate diffs (`diff_X_produced_ewma` = home offense vs. away
offense, `diff_X_allowed_ewma` = home defense vs. away defense) -- neither
one is actually "the matchup," they're each side's stats in isolation. This
version uses `diff_X_matchup_ewma` instead: `add_matchup_features()`
(`feature_engineering.py`) already builds `home_X_matchup_ewma =
(home_X_produced_ewma + away_X_allowed_ewma) / 2` (home offense blended
with away defense) and the away-side mirror, and `engineer_all()` now runs
matchup features BEFORE differential features specifically so
`diff_X_matchup_ewma = home_X_matchup_ewma - away_X_matchup_ewma` gets
built automatically the same way every other `diff_` column is.

Worth being precise about what this changes, since it's easy to overstate:
`diff_X_matchup_ewma` is an EXACT linear combination of the two features
the old list had -- `0.5 * (diff_X_produced_ewma - diff_X_allowed_ewma)`
(verified directly on synthetic data). So this isn't new information the
model couldn't already see; it's a change in what the model is ALLOWED to
fit. The old list let the regression find its own weight on offense vs.
defense from the training data (including whatever noise was in it); this
version hard-codes a 50/50 blend a priori, using football judgment instead
of a fitted weight. That's also fewer effective parameters (9 matchup
diffs vs. what was 13 separate produced/allowed diffs), which is an
independent reason this could generalize better given how much of this
build's evidence points to overfitting-on-noise as the dominant risk. For
XGBoost this is a more genuine change either way, since a tree sees the
blended matchup value as a single feature to split on rather than two raw
ingredients it has to learn to combine itself.

**Turnover-rate features are deliberately excluded.** Interception rate and
fumble rate were tried, then removed on the reasoning that turnovers are
largely random and shouldn't generalize as a team "skill." Removing them
made a borderline holdout result (consensus-pick z=2.0) drop to
non-significant (z=1.1), which raised the temptation to add them back to
recover the significant result -- doing that would be tuning the feature
set to a result rather than testing a fixed hypothesis, so they stayed out.
Follow-up research backs the original call: NFL fumble recovery rate has
essentially zero year-over-year correlation (recovering ~75% of your
fumbles one year predicts almost nothing about the next), and turnover
margin broadly regresses hard to the mean rather than persisting as a team
trait. See `config.py`'s comment above `CURATED_FEATURES` for the full
reasoning and the decision to resolve this with walk-forward validation on
a feature set fixed in advance, rather than by further tweaking the list.

Two models are fit on the identical curated set and evaluated against the
same baselines on the same chronological holdout as `05`, so this is a real
apples-to-apples comparison, not a vibes call:
- **Logistic regression** (statsmodels) -- appropriate now that the
  features-to-rows ratio is sane (~15 features, ~3,600 rows). Its
  coefficients and p-values are genuine multivariate evidence about which
  specific features matter controlling for the others, a real step up from
  04's one-at-a-time univariate screening. If the standard MLE fit hits a
  singular Hessian (near-collinear curated features, or quasi-complete
  separation on a small training set), it falls back to a regularized fit
  automatically -- verified against a synthetic edge case -- though that
  fallback path doesn't produce p-values, only coefficients.
- **XGBoost**, with much lighter regularization (`XGB_PARAMS_CURATED`)
  than 05 needed, since the small-n/large-p overfitting risk mostly
  doesn't apply at 15 features.

Verified against synthetic data with two planted real predictors among 13
pure-noise columns: the logistic regression correctly flagged the two real
predictors at p<0.001 and correctly reported the 13 noise columns as
non-significant, and both models beat the naive baselines on the holdout.

The script prints an explicit verdict either way: if at least one model
beats the base-rate baseline on holdout log loss, it says this is worth
taking to proper multi-season walk-forward validation before trusting it
(one 2-season holdout isn't enough to rule out a lucky split). If neither
does, it says so plainly -- a second, independent result agreeing with 05
that this feature set isn't showing reliable edge would be a real finding,
not a setup problem.

Output:
- `curated_model_metrics.json` -- both models vs. both baselines.
- `curated_logit_summary.txt` -- full regression table (coefficients,
  std errors, z-scores, p-values, confidence intervals).
- `curated_xgb_importance.csv` -- feature importance from the XGBoost side.

## Walk-forward validation (`07_walk_forward_validation.py`)

Run after `02_build_training_set.py` (same input as `06`, the full
`training_set.csv`):

```
python scripts/07_walk_forward_validation.py
```

This exists because `06`'s single 2024-2025 holdout (~550 games) gave
fragile results -- the consensus-pick z-score swung between ~2.0 and ~1.1
depending on whether turnover features were in the curated list, which is
itself a sign the earlier result was more noise than signal. The fix:
freeze `config.CURATED_FEATURES` (currently 12 features, matchup-based, no turnovers, decided BEFORE
looking at this script's output) and test it across every season the data
supports, not just one.

**Method: expanding-window, season by season.** Fold 1 trains on
2010-2017 and tests on 2018; fold 2 trains on 2010-2018 and tests on 2019;
and so on through training on 2010-2024 and testing on 2025
(`config.WALK_FORWARD_FIRST_TEST_SEASON` controls where folds start --
default 2018, chosen to guarantee at least 8 seasons of training history
before the first fold is scored). Both models are refit from scratch every
fold on only the data that would have actually existed at that point in
time -- same no-leakage discipline as the chronological split elsewhere,
just repeated across every available season.

**Why this is more trustworthy than `06`'s single holdout:** every fold's
consensus picks (games where the logistic regression and XGBoost predict
the same side) get pooled into one combined sample before computing
significance. Instead of ~550 games in one 2-season slice, this is every
test-season game across all 8 folds -- a much harder sample to fool with a
lucky split, and the feature set was fixed before any of these results were
seen.

Verified with a synthetic run: fully random data (no real signal by
construction) correctly produced pooled consensus accuracy of ~50.2% with
z=0.16 -- confirming the pooling and significance-testing logic doesn't
manufacture false positives out of noise.

Output:
- `walk_forward_fold_summary.csv` -- per-fold metrics (train/test size,
  logit and XGBoost AUC/accuracy/log loss, consensus-pick count and
  accuracy for that season alone).
- `walk_forward_consensus_picks.csv` -- every consensus pick from every
  fold, pooled, game by game -- includes `odds`/`profit` columns (actual
  market price for the picked side, and units won/lost) alongside
  `correct`, so this file works directly with
  `10_season_backtest_report.py` even though 07's selection rule itself
  never looks at price. Games missing a market price get `profit = NaN`
  (excluded from unit totals, not counted as a loss) rather than assuming
  a price that wasn't actually offered.
- `walk_forward_metrics.json` -- pooled consensus accuracy, standard error,
  z-score vs. 50%, and pooled units/ROI across the whole walk-forward
  window, plus mean AUC per model.

Read the printed verdict at the end literally: if the pooled z-score
doesn't clear ~1.96, that's the strongest evidence yet in this build that
this feature set doesn't have a reliable edge against the closing spread --
worth trusting over any single holdout's result, precisely because it's
harder to get by chance across 8 independent seasons than across one.

## Market-edge evaluation (`08_edge_based_evaluation.py`)

Run after `02_build_training_set.py`:

```
python scripts/08_edge_based_evaluation.py
```

This is a different, more realistic pick-selection methodology than `07`'s
"consensus when both models agree" -- it's closer to what this model was
actually built to do (find value against the market), rather than just
measuring raw accuracy against `spread_line`.

For each game and each side (home and away separately):
1. Convert the actual market price for that side (`home_spread_odds` /
   `away_spread_odds` -- American odds, e.g. `-110`, already present in
   `training_set.csv` via `build_features.pivot_to_game_level`, sourced
   from nflreadpy's `load_schedules()`) to an implied probability.
2. Blend the model's own probability for that side with the market's
   implied probability, weighted mostly toward the market
   (`config.EDGE_MODEL_WEIGHT`, default 0.35 on the model / 0.65 on the
   market).
3. `edge = blended_prob - market_implied_prob`, which simplifies to
   `EDGE_MODEL_WEIGHT * (model_prob - market_implied_prob)` -- the raw
   model/market disagreement, scaled down to reflect how much the model's
   opinion should be trusted against an efficient market.
4. Take whichever side has the higher edge as the pick, and only "give out"
   that pick if its edge clears a threshold -- swept across
   `config.EDGE_THRESHOLDS` (0% to 10%) rather than checked at a single cutoff,
   since 3% was historically just one point used to decide whether to
   release a pick, not the only threshold worth seeing.

Reported three ways, as requested: **logistic regression alone**, **XGBoost
alone**, and **combined** (the two models' raw probabilities averaged
*before* the market-blend step, then blended and edge-computed the same
way). Uses the same expanding-window walk-forward folds as `07` (not a
single holdout), pooling every out-of-sample season into one sample per
model for the same statistical-power reason `07` does.

**Games missing market odds are dropped from this analysis specifically**
(there's no price to compare the model to) -- a coverage table is printed
first showing exactly which seasons have odds and what fraction of games.
nflverse's odds coverage is known to be sparser in older seasons; if a real
run shows early seasons mostly missing, `config.SEASONS` may be worth
narrowing for this analysis even if the rest of the pipeline keeps the full
history.

**Read the ROI column, not just accuracy.** Because real odds vary pick to
pick (not always exactly -110/-110), a given win rate isn't always
profitable -- `roi` uses each pick's actual price to report average units
won per unit staked (0% = break-even; negative means losing money even
above 50% accuracy, since -110 needs ~52.4% just to break even). Also watch
`n_picks` at each threshold: it shrinks fast as the threshold rises, and
accuracy at very low n (single digits or teens) swings wildly by chance --
don't read a high accuracy at edge >= 7-10% as meaningful unless the pick
count behind it is large enough to trust.

Verified with a synthetic run (random target, uncorrelated with both
features and simulated odds, some seasons with odds intentionally left
missing): the coverage report correctly flagged the missing seasons, and
accuracy/ROI at every threshold came back noisy and inconsistent in sign as
expected for data with no real signal -- confirming the pipeline doesn't
manufacture an edge out of nothing.

Output:
- `edge_eval_picks.csv` -- every pick from every model, every fold, with
  its side, edge, price, and outcome.
- `edge_eval_threshold_summary.csv` -- accuracy, standard error, and ROI
  per model per threshold.
- `edge_eval_metrics.json` -- the same summary plus run metadata (weight
  used, thresholds swept, seasons covered).

## Consensus + edge walk-forward (`09_consensus_edge_walk_forward.py`)

Run after `02_build_training_set.py`:

```
python scripts/09_consensus_edge_walk_forward.py
python scripts/09_consensus_edge_walk_forward.py --threshold 0.0   # override config.CONSENSUS_EDGE_THRESHOLD without editing config.py
```

Combines `07`'s agreement gate with `08`'s price gate into one requirement:

1. Logistic regression and XGBoost must predict the **same side** --
   `logit_prob >= 0.5` and `xgb_prob >= 0.5` land on the same side. This is
   now IDENTICAL to `07`'s agreement test (see fix note below); the only
   difference between `07` and `09` is condition 2.
2. The **combined** model's edge on that agreed side (probabilities
   averaged before the market-blend step, same "combined" definition as
   `08`) must be `>= threshold` (`--threshold`, or `config.CONSENSUS_EDGE_THRESHOLD`
   if not passed -- default 2%).

**Fix worth knowing about:** the original version of this script defined
"agreement" differently from `07` -- it compared each model's EDGE-implied
side (whichever side has the higher edge once blended toward the market)
rather than the raw `>=0.5` side. Those two definitions can disagree
whenever the home/away market prices aren't symmetric, meaning `07` and
`09` were silently answering slightly different questions beyond just the
added edge filter. This was caught by `11_edge_calibration_check.py`'s
result (edge size showed no relationship to accuracy) prompting a closer
look at why `09` scored so much worse than `07` on the same underlying
models -- part of the answer turned out to be this definitional mismatch,
not just the edge threshold. Fixed so `09` is now a clean superset test:
"everything `07` would pick, further filtered by edge."

The `--threshold` flag exists because of this: with the definitions now
aligned, testing `--threshold 0.0` (agreement plus a merely non-negative
edge, as opposed to the default 2% bar) isolates whether ANY edge
requirement helps or hurts relative to `07`'s no-filter baseline, without
needing to hand-edit `config.py` for every threshold tried.

Both conditions have to hold -- this produces a smaller, more selective set
of picks than `07` alone (how much smaller depends on the threshold), which
is worth watching closely: the fold-by-fold output prints `n_test` /
`n_agree` / `n_selected` explicitly so it's clear how much the pool shrinks
at each gate, and how thin `n_selected` gets in any single season.

Picks are pooled across the same expanding-window walk-forward folds as
`07`/`08` before computing overall accuracy, standard error, z-score vs.
50%, and ROI (using each pick's actual price). If `n_total` picks comes
back small (<30), the printed verdict flags that explicitly rather than
reporting a headline accuracy number that a small sample can't actually
support.

One thing worth being upfront about: by the time this script runs, several
other selection rules have already been tested against this same feature
set and data (`06`'s single holdout, `07`'s consensus-only, `08`'s
edge-only by three model constructions). If THIS rule happens to come back
significant after the others didn't, that's the multiple-comparisons
problem in a very literal form -- the script's printed verdict says so
explicitly rather than presenting a late "hit" as a confirmed result.

Output:
- `consensus_edge_picks.csv` -- every selected pick, with both models'
  individual edges alongside the combined edge that gated it.
- `consensus_edge_metrics.json` -- pooled accuracy, ROI, and z-score,
  plus run metadata.

## Season backtest report (`10_season_backtest_report.py`)

Run after `09_consensus_edge_walk_forward.py` (or any script that produces
a picks CSV with `season`/`week`/`correct`/`profit` columns):

```
python scripts/10_season_backtest_report.py --season 2025
```

The pooled multi-season stats from `07`/`08`/`09` answer "is there a
statistically detectable edge at all" -- a fair question, but a different
one from "would this have been a strategy worth running last season."
This renders one season as an actual week-by-week table: record and units
for that week, plus a running season total, so a decent-looking overall
number can't quietly be hiding one huge week propping up a lot of average
or losing ones.

Defaults to `09`'s output (`config.CONSENSUS_EDGE_PICKS_CSV`) and the most
recent season in that file, but takes `--input` to point at a different
picks file (e.g. `08`'s `edge_eval_picks.csv`, which mixes all three models
together and needs `--model logit`/`xgb`/`combined` to filter to one) and
`--season` to pick a specific year.

Output: prints the table, and also writes it to
`season_report_{season}.csv` for reuse (e.g. in the eventual Google Sheet
dashboard).

## Edge calibration check (`11_edge_calibration_check.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/11_edge_calibration_check.py
```

Prompted by a real, concrete discrepancy: `07`'s raw consensus rule (agree
on side, no price filter) scored z=3.25 pooled across 8 seasons -- the
strongest result in this build. `09`'s consensus+edge rule, using the exact
same two models and feature set but ALSO requiring the edge to clear 2%,
scored z=0.60 on a stricter subset of those same picks -- meaningfully
worse. If "edge" (model probability vs. market implied probability) were
doing what it's supposed to -- flagging the picks worth the most
confidence -- filtering to bigger edges should improve accuracy, not tank
it. This checks that relationship directly instead of just picking
whichever headline number looks better.

Takes every consensus pick from `07`'s walk-forward run (every game where
logit and XGBoost agreed -- NOT pre-filtered by edge), computes each pick's
edge with the same formula `08`/`09` use, and splits into deciles by edge
size. If bigger edge really means more trustworthy, accuracy should trend
up from decile 1 (lowest edge) to decile 10 (highest edge). A Spearman
correlation between edge and correctness gives a formal answer (positive +
significant = edge is informative; flat or negative = it isn't, at least
not with this feature set).

Verified on synthetic (random-target) data: correctly returned a
near-zero, non-significant correlation (r=-0.006, p=0.84) with a flat
decile table -- confirming the diagnostic doesn't manufacture a trend out
of noise.

**Reading the real result matters more than the mechanics here.** If the
correlation comes back flat/non-significant, that doesn't mean the
underlying consensus signal (`07`) is untrustworthy -- it means edge size
specifically isn't adding information on top of raw agreement *yet*, so
`09`'s extra filter is just shrinking the sample without concentrating the
good picks. If it comes back positive and significant, `09` scoring worse
needs a different explanation (an unlucky threshold, most likely) and is
worth re-testing at other cutoffs. If it comes back negative and
significant, that's a genuine red flag -- the model being most confidently
wrong exactly when it disagrees most with the market -- worth treating any
edge-based filter with real suspicion until understood.

Accepts `--input` to run against any other picks file, not just `07`'s
default -- e.g. `data/processed/three_way_edge_aligned_picks.csv` (`23`'s
output). If the input file already has its own `edge` column, that column
is used AS-IS instead of being recomputed from `logit_prob`/`xgb_prob` --
`23`'s picks carry an edge computed from the 3-model (logit+xgb+nb)
average, and recomputing it here from just 2 of those 3 models would
silently analyze a different number than the one that actually selected
those picks. Output filenames get a suffix matching the input file's stem
when `--input` is used, same convention as `14`.

Output:
- `edge_calibration_deciles.csv` -- accuracy, units, ROI, and standard
  error for each of 10 edge deciles.
- `edge_calibration_metrics.json` -- Spearman correlation, p-value, and the
  decile table.
- Or `edge_calibration_deciles_<input-stem>.csv` /
  `edge_calibration_metrics_<input-stem>.json` when run with `--input`.

## Confidence calibration check (`12_confidence_calibration_check.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/12_confidence_calibration_check.py
```

Same diagnostic as `11`, but a different question: forget the market
entirely -- when the model itself is more confident (its probability
further from 50%), is it actually more often right? Confidence is defined
as `|combined_prob - 0.5|`, symmetric across home and away picks (a
combined_prob of 0.30 and 0.70 both give confidence 0.20, since one is a
confident away pick and the other a confident home pick -- see the
docstring for the derivation).

This was prompted directly by wanting a volume-reducing, ROI-raising
selection rule after market-based edge (`11`) failed to provide one. If the
model's own confidence tracks accuracy, that's a legitimate lever
market-edge wasn't; it's also the necessary first check before any
probability calibration work, since there's no reason to assume "the model
says 65%" corresponds to being right 65% of the time without checking.

Same decile + Spearman correlation structure as `11`. Also prints the
most-confident decile's accuracy and ROI on its own, since the practical
question ("is just betting the top 10% of confident picks good enough to
matter") is separate from whether the correlation is significant across
the whole range.

Verified on synthetic (random-target) data: correctly returned a
near-zero, non-significant correlation with no decile trend.

Output:
- `confidence_calibration_deciles.csv`
- `confidence_calibration_metrics.json`

## Pick-type breakdown (`13_pick_type_breakdown.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/13_pick_type_breakdown.py
```

`11` and `12` both checked probability-derived quantities (market edge,
model self-confidence) as ways to cut volume and raise ROI, and both came
back null. This checks a different, football-motivated axis instead:
whether the pick was on the home or away side, and whether that side was
the market favorite or underdog (from `spread_line`'s sign -- positive =
home favored, matching the convention used everywhere else in this
pipeline). These are pre-registered categories a football person would
think to check, not another round of threshold fishing.

Reports four combinations plus the two marginal splits: home vs. away
picks, favorite vs. underdog picks, and all four home/away x favorite/dog
combinations. Pick'em games (`spread_line == 0`, no favorite) are broken
out separately rather than forced into either bucket.

Defaults to `07`'s unfiltered consensus picks, but accepts `--input` for
any other picks file that carries `predicted_home_cover` (or `picked_side`)
and `spread_line` -- e.g. `21`'s or `23`'s output. `09`'s picks file
doesn't include `spread_line`, so `--input` won't work against that one
without adding the column there first. Output filenames get a suffix
matching the input file's stem when `--input` is used, same convention as
`11`/`14`.

Output: `pick_type_breakdown.csv` -- n picks, wins/losses, accuracy, units,
and ROI for every category (or `pick_type_breakdown_<input-stem>.csv` when
run with `--input`).

## Pick-type x edge breakdown (`14_pick_type_edge_breakdown.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/14_pick_type_edge_breakdown.py
python scripts/14_pick_type_edge_breakdown.py --input data/processed/three_way_consensus_picks.csv
```

`11`'s edge-calibration check found no relationship between edge and
accuracy pooling ALL consensus picks together. `13` then found a real
split by favorite/underdog. This checks whether edge was actually
informative all along, just masked by averaging favorites and underdogs
into one pooled correlation -- e.g. if edge helps within underdog picks
specifically but is pure noise (or inverted) within favorite picks, pooling
both groups could wash out a real, usable pattern.

Splits into favorite / underdog (same logic as `13`), then computes edge
(same formula as `08`/`09`/`11`/`19`/`20`:
`edge = EDGE_MODEL_WEIGHT * model_prob + (1 - EDGE_MODEL_WEIGHT) * market_prob
- market_prob`, which simplifies to `EDGE_MODEL_WEIGHT * (model_prob -
market_prob)`) and buckets into quintiles independently WITHIN each group,
with its own Spearman correlation per group.

Accepts `--input` to run the same breakdown against any other picks file
with the same schema, not just `07`'s default 2-way consensus output --
e.g. `21_three_way_consensus.py`'s 3-way picks or
`23_three_way_edge_aligned.py`'s edge-aligned picks. If the input file
already has its own `edge` column (like `23`'s output does, computed from
the 3-model average), that column is used as-is instead of being
recomputed from `logit_prob`/`xgb_prob` -- same reasoning as `11`'s
`--input` support: recomputing from only 2 of 3 models would silently
analyze a different number than the one that actually produced those
picks. Output filenames get a suffix matching the input file's stem when
`--input` is used, so different runs don't overwrite each other.

Output: `pick_type_edge_breakdown.csv` (bucket-level accuracy/units/ROI for
both groups) and `pick_type_edge_metrics.json` (the two correlations) --
or `pick_type_edge_breakdown_<input-stem>.csv`/`..._<input-stem>.json` when
run with `--input`.

## Combined favorite/underdog + edge rule (`15_combined_rule.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/15_combined_rule.py
```

Built directly from two real findings, not a new fishing expedition:
`13` found favorite picks are break-even (51.3% accuracy) while underdog
picks carry virtually all the profit (56.0%); `14` found that WITHIN
favorite picks specifically, bigger edge is significantly associated with
WORSE accuracy (p=0.027) while underdog picks showed no edge relationship
at all. Rule: keep every underdog consensus pick, and only favorite
consensus picks whose edge is `<= config.FAVORITE_EDGE_MAX_THRESHOLD`
(default 0.7%, from where `14`'s quintiles were still strong before the
accuracy cliff at higher edges). Pick'em games are excluded, same as
`13`/`14`.

**Read the printed caveat before trusting the result.** The 0.7% cutoff
was chosen by inspecting quintile boundaries computed on this exact
backtest -- running that same threshold against the same data is a
plausible-first-pass check ("does combining what 13+14 found produce a
better backtest"), not a clean out-of-sample validation of that specific
number. A rigorous version would re-derive the threshold inside each
walk-forward fold using only prior seasons' data, so it's never chosen
with knowledge of the fold it's applied to -- worth doing before trusting
this for real money.

Output uses the same schema as `07`'s picks file, so it plugs directly
into `10_season_backtest_report.py`:
```
python scripts/10_season_backtest_report.py --input data/processed/combined_rule_picks.csv --all-seasons
```

Output:
- `combined_rule_picks.csv` -- every selected pick.
- `combined_rule_metrics.json` -- accuracy/units/ROI for the combined rule
  AND for `07`'s unfiltered baseline side by side, plus how many favorites
  survived the filter.

## Market favorite/underdog base rate (`16_market_base_rate_check.py`)

Run any time after `02_build_training_set.py` (doesn't depend on any model
output -- this is a completely model-free check):

```
python scripts/16_market_base_rate_check.py
```

Prompted by wanting to understand WHY the model's consensus picks skew
toward underdogs (`13`'s finding). Checks whether that's a model artifact
or a real, pre-existing pattern in the market: computes the raw historical
cover rate for favorites vs. underdogs across every game in the FULL
2010-2025 dataset (not just consensus picks, not just the walk-forward
window, and with no model prediction involved at all). If favorites cover
measurably less than 50% of the time and underdogs measurably more, that
confirms the model learned something that was already true in the data it
trained on, rather than inventing a bias of its own.

Pick'em games (`spread_line == 0`) and pushes (`margin == spread_line`
exactly) are excluded from the rate calculation and reported separately,
since neither side "covers" in either case.

Output: `market_base_rate.csv` -- favorite/underdog cover rate and a
z-score vs. 50%, for all games, home-favorite games, and away-favorite
games separately.

**Also breaks the same rate out by season**, prompted by a real concern
raised about `15`/`18`'s combined rules: with 80%+ of their picks on the
underdog side, both are a large, concentrated structural bet that the
historical underdog tilt keeps holding. If that tilt weakens or reverses
in a given year, a strategy this lopsided has little in it to cushion
that. `market_base_rate_by_season.csv` is the model-free half of checking
that risk directly -- compare it against
`10_season_backtest_report.py --input data/processed/combined_rule_picks.csv --all-seasons`
(or `_v2_picks.csv`) to see whether the rule's worst seasons line up with
years the market-wide base rate favored favorites more than usual. If they
do, that confirms the exposure. If the rule holds up even in those years,
that's evidence of real selection skill beyond just riding the tilt.

## Home/away favorite edge breakdown (`17_home_favorite_edge_breakdown.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/17_home_favorite_edge_breakdown.py
```

`16`'s model-free base-rate check found the market's tilt away from
favorites concentrates in HOME favorites (48.3% cover, z=-1.68) rather
than away favorites (49.8%, z=-0.15). This checks whether `14`'s
"high-edge favorites do worse" finding (pooling home and away favorites
together) is actually a home-favorite-specific phenomenon -- splits that
same edge-bucket check into home-favorite-only and away-favorite-only
groups, with their own accuracy/units/ROI and Spearman correlation.

Home favorite picks are a much smaller group, so fewer buckets are used
for that side (`config.HOME_FAVORITE_EDGE_N_BUCKETS`, default 3 vs. 5 for
away favorites) to keep per-bucket samples large enough to read.

If home favorites turn out to show a much stronger negative edge-accuracy
relationship than away favorites, that would suggest sharpening
`15_combined_rule.py` further -- a stricter edge threshold for home
favorites than for away favorites, rather than the one shared 0.7% cutoff
currently used for all favorites.

Output: `home_away_favorite_edge_breakdown.csv` and
`home_away_favorite_edge_metrics.json`.

## Combined rule v2: separate home/away favorite thresholds (`18_combined_rule_v2.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/18_combined_rule_v2.py
```

Refines `15`'s single favorite-edge threshold (0.7% shared by home and
away favorites) into two separate thresholds, following `17`'s finding
that the two behave differently: home favorites only looked reliable below
~0.4% edge (the next bucket up was already near break-even), while away
favorites held up through a wider range, to about ~1.3%. Rule: underdogs
unfiltered (same as `15`), home favorites kept only if edge `<=
config.HOME_FAVORITE_EDGE_MAX_THRESHOLD` (default 0.4%), away favorites
kept if edge `<= config.AWAY_FAVORITE_EDGE_MAX_THRESHOLD` (default 1.3%).

**Even stronger version of the caveat already stated for `15`:** these two
thresholds came from bucket boundaries computed on an even smaller,
further-split sample (89 home favorite picks across only 3 buckets) than
`15`'s single cutoff was. More granularity is not automatically more
trustworthy -- it's more opportunity to fit this specific backtest's
noise. This script exists to see whether the refinement helps here, and
explicitly says to compare against `15`'s result rather than assuming v2
is better just because it's more targeted.

Output uses the same schema as `07`'s/`15`'s picks files:
```
python scripts/10_season_backtest_report.py --input data/processed/combined_rule_v2_picks.csv --all-seasons
```

Output:
- `combined_rule_v2_picks.csv`
- `combined_rule_v2_metrics.json` -- includes both thresholds, how many
  home/away favorites survived each, and the same accuracy/units/ROI
  comparison against `07`'s baseline that `15` reports.

## Positive-edge rule (`19_positive_edge_rule.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/19_positive_edge_rule.py                   # sweep only
python scripts/19_positive_edge_rule.py --threshold 0.01   # sweep + full picks file at 1%
```

`15`/`18` only filtered FAVORITE picks by edge -- underdog picks were taken
unconditionally, with no edge floor at all. This replaces that with a
single edge floor applied to EVERY pick, favorite or underdog, prompted by
three concerns raised together:

1. A real operation should only give out +EV picks by the model's own
   estimate, on both sides, not "all underdogs regardless of price plus
   selective favorites."
2. Volume under `15`/`18` (~7-10 picks/week) was too high.
3. The favorite/underdog split under `15`/`18` (~80/20) was more lopsided
   than desired -- filtering underdogs by edge too, not just favorites,
   should pull the split back toward center.

**Important note on the odds used here:** edge is computed against this
dataset's historical spread odds (nflverse, close to standard -110), NOT
Novig's actual live prices. If Novig's odds are more favorable in general
(more +money on spreads, as expected), this backtest's edge/ROI numbers at
every threshold are likely a CONSERVATIVE estimate of live performance,
not an inflated one -- real edge against Novig's prices should generally
be at least as good as what's shown here, plausibly better.

Rather than pick one threshold, this sweeps
`config.POSITIVE_EDGE_THRESHOLDS_SWEEP` (0% to 3%) and reports, at each
level: total picks, favorite/underdog split and favorite %, accuracy,
z-score, units, and ROI, and what % of the unfiltered baseline survives --
the full tradeoff curve to choose a threshold from, rather than guessing
one number. Pick'em games are excluded, consistent with every other
favorite/underdog script in this build.

Pass `--threshold` to also write the full picks file at one specific
threshold, ready for `10_season_backtest_report.py`.

On the real walk-forward data (2018-2025, 1,286 consensus picks), the sweep
solved concerns (1) and (2) -- e.g. at a 1.0% floor: 702 picks (~5/week,
down from ~8.2/week), 53.8% accuracy, z=2.04, +5.1% ROI. But it did NOT
solve (3): favorite_pct barely moved (31.1% -> 23-29%) as the threshold
rose, since a single shared floor cuts favorites and underdogs roughly
proportionally rather than rebalancing the mix. That gap is what
`20_independent_edge_rule.py` addresses directly.

## Independent favorite/underdog edge rule (`20_independent_edge_rule.py`)

Run after `07_walk_forward_validation.py`:

```
python scripts/20_independent_edge_rule.py                                              # grid only
python scripts/20_independent_edge_rule.py --favorite-threshold 0.0 --underdog-threshold 0.02  # grid + full picks file at that cell
```

Fixes the gap `19` left open: instead of one shared edge floor, this
applies TWO independent floors -- one for favorite picks, one for underdog
picks -- and sweeps every combination of
`config.FAVORITE_EDGE_THRESHOLDS_2D` x `config.UNDERDOG_EDGE_THRESHOLDS_2D`
(default 5x5 = 25 cells). Each cell reports n_picks, favorite/underdog
split and favorite %, accuracy, z-score, units, and ROI.

The idea: `13`/`14` found favorites have real (if smaller-sample) edge
structure at low edge specifically, while underdogs showed no edge
relationship in the pooled check -- so a LOWER bar for favorites and a
HIGHER bar for underdogs is the most football-motivated way to thin
underdogs preferentially, rebalancing favorite_pct toward center rather
than just shrinking both sides together the way `19`'s shared floor did.

When reading the grid, look for a cell where favorite_pct has moved
meaningfully toward 50% (not just where n_picks is smaller) while z_score
and ROI are still acceptable -- that's evidence of real rebalancing, not
just volume cutting.

Same caveat as everything downstream of `13`/`14`/`15`/`18`: still an
in-sample exploration on one backtest, not a nested walk-forward
validation -- use this to pick a promising combination to sanity-check
further, not as a provably robust final rule.

Pass `--favorite-threshold` and `--underdog-threshold` together to also
write the full picks file at one specific grid cell, ready for
`10_season_backtest_report.py`.

Output:
- `positive_edge_rule_sweep.csv` -- the full threshold sweep table.
- `positive_edge_rule_metrics.json` -- same data as JSON.
- `positive_edge_rule_picks.csv` -- only written if `--threshold` is
  passed; every pick at that one threshold.

On the real walk-forward data, the grid showed a clean pattern: raising the
underdog threshold monotonically hurt accuracy/z/ROI in every row (matches
14's earlier null finding for underdog edge), while raising the favorite
threshold was roughly neutral-to-positive. Best balance-improving cell:
favorite=0.0%/underdog=1.0% (849 picks, favorite_pct 31%->41%, z=2.38,
ROI+5.3%, basically unchanged from baseline). Best raw quality: favorite
=2.0%/underdog=0.0% (863 picks, z=3.18, ROI+7.5%, but favorite_pct drops to
11% -- the imbalance gets WORSE, not better). Read together, the grid
suggests the underdog concentration is real, not an edge-filterable
artifact -- forcing it toward 50/50 trades away real edge to do it.

## Three-way consensus (`21_three_way_consensus.py`)

Run after `07_walk_forward_validation.py` (only needs `training_set.csv`,
not 07's output directly, but 07 is worth running first for the 2-way
comparison point):

```
python scripts/21_three_way_consensus.py
```

Prompted directly by: "could a third model help smooth all these issues?"
(volume, class imbalance, needing positive edge everywhere). Every edge
threshold tried so far (`08`/`09`/`19`/`20`) has shown null or backwards
relationships to accuracy -- the one mechanism that HAS reliably worked all
build is agreement between models with different inductive biases (`07`'s
2-way consensus, z=3.25 pooled, the strongest result in this project).
This tightens that same mechanism from 2-way to 3-way instead of adding
another hand-picked edge cutoff.

Adds a Gaussian Naive Bayes model, trained on the SAME curated 12-feature
set as logit/XGBoost (deliberately NOT a different/broader feature set --
`05_train_baseline_model.py` already showed that overfits badly at this
sample size: train AUC 0.97, holdout AUC 0.48). Naive Bayes makes a
different assumption than either existing model (features are
conditionally independent given the outcome, vs. logit's linear
combination or XGBoost's learned interactions), so its errors should be
less correlated with the other two -- a genuinely more independent third
vote, not just another version of the same one.

Refits all three models from scratch in every walk-forward fold, same
expanding-window design as `07`. Reports 2-way (logit+xgb) and 3-way (all
three agree) consensus pooled across every fold side by side, so the
volume/accuracy tradeoff of tightening to 3-way is visible directly.

Output:
- `three_way_fold_summary.csv` -- per-fold accuracy/AUC for all three
  models plus both consensus definitions.
- `three_way_consensus_picks.csv` -- the 3-way consensus picks only
  (2-way's picks/stats are printed and included in the metrics JSON, but
  not written to their own file -- that's `07`'s output), ready for
  `10_season_backtest_report.py`, `13_pick_type_breakdown.py`, or as input
  to `19`/`20`'s edge rules.
- `three_way_metrics.json` -- both consensus definitions' pooled
  accuracy/z-score/units/ROI.

Compare the 3-way accuracy/z/ROI change against what `19`/`20`'s edge
floors gave up for a similar volume cut -- if 3-way holds up better per
pick removed, that's evidence agreement-based filtering is the more
informative lever, consistent with edge showing null/backwards
relationships everywhere else this build has checked. Worth also running
`13_pick_type_breakdown.py` against `three_way_consensus_picks.csv` to see
whether the favorite/underdog split moved at all -- this wasn't targeted
at the imbalance issue the way `19`/`20` were, so any shift would be
incidental, not designed.

## Edge-only pick selection, no consensus gate (`22_edge_only_breakdown.py`)

```
python scripts/22_edge_only_breakdown.py
```

Every edge diagnostic before this one (`11`, `14`) only checked edge's
relationship to accuracy WITHIN the consensus-gated subsample (games where
logit and XGBoost already agreed). This asks a different question: ignore
agreement entirely, and for EVERY game, bet whichever side -- home or away
-- has the higher edge (same formula as everywhere else:
`edge = EDGE_MODEL_WEIGHT * combined_prob + (1 - EDGE_MODEL_WEIGHT) *
market_prob - market_prob`, where `combined_prob` is the averaged
logit+xgb probability, computed per side against that side's
market-implied probability from its American odds). Does edge size predict
accuracy across the FULL slate, not just the self-selected subset where
both models already agreed?

`08_edge_based_evaluation.py` already builds this same "no consensus gate,
pick the higher-edge side" population, but only reports cumulative
threshold sweeps (accuracy for edge >= X). This reports the identical
underlying picks as edge deciles with a Spearman correlation instead,
matching `11`'s format so it's directly comparable to `11`'s
consensus-gated result.

Also prints the "bet the model's favorite side on literally every game, no
threshold at all" baseline accuracy/z-score/units/ROI -- worth comparing
directly against `07`'s consensus accuracy to see how much of `07`'s edge
comes from the agreement filter itself versus the model's raw ability to
pick a side.

Output:
- `edge_only_breakdown_picks.csv` -- every game, one row per game, with its
  picked side, edge, and outcome.
- `edge_only_breakdown_deciles.csv` -- accuracy/units/ROI by edge decile.
- `edge_only_breakdown_metrics.json` -- overall baseline stats plus the
  Spearman correlation and decile table.

## Three-way consensus + edge-direction alignment (`23_three_way_edge_aligned.py`)

```
python scripts/23_three_way_edge_aligned.py
```

Prompted directly by: "is there any way to combine these? all three have to
agree and the average edge has to agree?" Stacks two independently-checked
filters: `21`'s 3-way raw-threshold agreement (logit, xgb, and Naive Bayes
all >= 0.5 on the same side), AND the market-adjusted edge (using the
3-model average probability) also has to point to that same side.

Why these can actually disagree even though it sounds like they shouldn't:
raw threshold agreement only checks what the models themselves think.
Whether a side has the higher edge ALSO depends on how asymmetric that
game's actual home/away market prices are -- `compute_edges()` compares
`home_edge` vs `away_edge`, and those aren't mirror images of each other
unless `home_spread_odds` and `away_spread_odds` happen to be symmetric
(e.g. both -110). This is the same distinction that caused a real bug,
already found and fixed once in this build: `09_consensus_edge_walk_forward.py`
originally conflated "the side implied by a raw >=0.5 threshold" with "the
side implied by comparing edges," and they are NOT always the same game.
Requiring both to agree here is therefore a stricter, more specific claim
than 3-way agreement alone: not just "all three models like this side" but
"...and that preference survives being checked against the actual price
you'd bet at."

Reports 3-way-alone and 3-way+edge-aligned pooled across every fold side by
side, plus how many picks the edge-alignment check actually removes (how
often the discrepancy triggers in practice).

Output:
- `three_way_edge_aligned_fold_summary.csv` -- per-fold pick counts for
  both rules.
- `three_way_edge_aligned_picks.csv` -- the stricter (3-way + edge-aligned)
  picks only, ready for `10_season_backtest_report.py` or
  `13_pick_type_breakdown.py`.
- `three_way_edge_aligned_metrics.json` -- both rules' pooled
  accuracy/z-score/units/ROI.

## Final combined rule (`24_final_combined_rule.py`)

```
python scripts/24_final_combined_rule.py
```

Consolidates everything this build found into one rule, built on `21`'s
3-way consensus (NOT `23`'s edge-aligned version -- edge-direction
alignment was tested and added complexity without adding value, same
Occam's-razor reasoning that favored `15` over `18`):

- Every 3-way-consensus underdog pick, kept unfiltered (edge has shown no
  relationship to underdog accuracy in every check this build has run --
  `11`, `14`, `20`, and twice more on the 3-way picks).
- Every 3-way-consensus favorite pick, kept only if its edge (computed
  from the 3-model average probability) is <=
  `config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD` (default 0.2%) -- this
  favorite-side edge relationship has now replicated THREE times across
  this build (2-way consensus, 3-way consensus, 3-way+edge-aligned), each
  time more significant than the last (p=0.027 -> p=0.055 -> p=0.005).

**Same caveat as 15/18/19/20, stated again because it matters:** the 0.2%
cutoff was chosen from a decile boundary computed on this same backtest.
This is a reasonable first pass at combining everything found so far, not
proof the exact number holds going forward. A rigorous version re-derives
the threshold inside each walk-forward fold using only prior seasons
(nested walk-forward) -- not yet built, and worth doing before this rule
is trusted with real money.

Output:
- `final_combined_rule_picks.csv` -- the selected picks, ready for
  `10_season_backtest_report.py`.
- `final_combined_rule_metrics.json` -- accuracy/z-score/units/ROI/
  favorite% for both this rule and `21`'s unfiltered 3-way baseline, side
  by side.

## Not yet wired up (left as TODOs)

- **Novig integration** (`sheets/novig_client.py`): needs real endpoint,
  auth, and a team-code / spread-sign-convention check against nflverse's
  format before it's safe to feed into the same feature slot as `spread_line`.
- **Live scoring script**: doesn't exist yet — will look like
  `01_fetch_historical.py` + `novig_client.py` feeding straight into
  `build_features.build_full_dataset()`, then `sheet_sync.push_dataframe()`.
- **Features discussed but not yet in the pipeline**: rest days / short
  week, travel distance, QB starter changes, weather. `load_schedules()`
  already returns rest days and roof/surface/temp/wind — those just need to
  be pulled into `build_features.py`. QB starter changes would need
  `load_players`/`load_rosters_weekly` or injury reports.
- **Opponent/strength-of-schedule adjustment**: current `_allowed` stats are
  raw, not adjusted for opponent quality. Worth revisiting if the model
  starts overrating teams that faced weak offenses/defenses.

## A note on testing

This scaffold was written and reviewed but not run end-to-end against live
nflreadpy data — the sandbox this was built in doesn't have network access
to nflverse's data host. Run `01_fetch_historical.py` first and check the
printed column list against what `build_features.py` expects (`team` vs.
`recent_team`, presence of `spread_line`, etc.) before trusting the output.
