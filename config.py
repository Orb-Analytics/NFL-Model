"""
Central configuration for the Orb NFL model data pipeline.

Edit these values as your model evolves. Nothing else in the pipeline
should need season/window numbers hardcoded elsewhere -- keep it all here.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_SCHEDULES_PATH = RAW_DIR / "schedules.parquet"
RAW_TEAM_STATS_PATH = RAW_DIR / "team_stats.parquet"
RAW_PFR_ADVSTATS_PATH = RAW_DIR / "pfr_advstats_{stat_type}.parquet"

TRAINING_SET_CSV = PROCESSED_DIR / "training_set.csv"
TRAINING_SET_XLSX = PROCESSED_DIR / "training_set_snapshot.xlsx"
FEATURE_MANIFEST_CSV = PROCESSED_DIR / "feature_manifest.csv"

TRAINING_SET_PRUNED_CSV = PROCESSED_DIR / "training_set_pruned.csv"
PRUNE_LOG_CSV = PROCESSED_DIR / "prune_log.csv"

# --------------------------------------------------------------------------
# Seasons to pull. `True` = all available seasons in nflverse (goes back to
# 1999 for schedules, ~1999+ for team stats depending on stat). Narrow this
# to whatever window is actually representative of the current NFL (rule
# changes, pace of play, etc. make very old seasons less useful as training
# signal for a lot of stats).
#
# Includes the current season (CURRENT_SEASON below) deliberately -- for
# live scoring, load_schedules() returns the current season's completed AND
# upcoming/unplayed games together (unplayed games have null scores). This
# is what lets build_features.py produce feature rows for next week's games
# using the exact same rolling/EWMA pipeline as training, with no separate
# "live" code path (see build_features.py's module docstring and
# 25_live_weekly_scoring.py).
# --------------------------------------------------------------------------
CURRENT_SEASON = 2026
SEASONS = list(range(2010, CURRENT_SEASON + 1))

# Include playoff games in the training set? Playoffs behave differently
# (higher stakes, short rest weirdness less common, etc.) -- default is to
# exclude them so the model trains on regular-season dynamics only.
INCLUDE_POSTSEASON = False

# --------------------------------------------------------------------------
# Rolling / recency-weighting parameters
# --------------------------------------------------------------------------
# Simple trailing-window average, in games. Provided as an alternative /
# sanity-check alongside the EWMA features.
ROLLING_WINDOW_GAMES = 5

# EWMA half-life, in games. A game this many games back has half the weight
# of the most recent game. This is computed on each team's continuous game
# log across season boundaries -- see build_features.py for why that
# naturally handles the "blend in prior season, taper it out" problem
# without needing a separate manual blending formula.
EWMA_HALFLIFE_GAMES = 4

# --------------------------------------------------------------------------
# Columns that identify a game/team-week rather than being a statistic to
# roll. Extend this if nflreadpy's schema includes additional metadata
# columns you don't want treated as a numeric feature.
# --------------------------------------------------------------------------
ID_COLUMNS = {
    "season",
    "week",
    "game_id",
    "team",
    "recent_team",
    "opponent_team",
    "opponent",
    "is_home",
    "season_type",
    "game_type",
    "player_id",
    "pfr_id",
    "gsis_id",
}

# --------------------------------------------------------------------------
# Extra data sources merged into team_stats before rolling. The philosophy
# for this pipeline is "bring in everything nflreadpy exposes at team-week
# grain, prune later" -- add to this list rather than hand-picking columns
# up front. Each entry is aggregated to (season, week, team) and merged in,
# so every numeric column it contributes automatically gets rolled/EWMA'd
# and pruned like any other stat.
#
# PFR advanced stats (pressure rate, time to throw, broken tackles, yards
# before/after contact, etc.) are player-level box-score stats -- only
# available 2018+. They get summed to team-week before merging.
# --------------------------------------------------------------------------
INCLUDE_PFR_ADVSTATS = True
PFR_ADVSTATS_TYPES = ["pass", "rush", "rec", "def"]
PFR_ADVSTATS_MIN_SEASON = 2018

# --------------------------------------------------------------------------
# Rate stats -- computed at the team-game level BEFORE rolling, from raw
# counting stats already in team_stats. Almost every raw counting stat in
# team_stats scales with how many plays got run that game (pace/game script),
# not just EPA -- a team with 45 dropbacks racks up more total yards, TDs,
# and EPA than an efficient 25-play day even if the second team was better
# per play. Rolling a raw per-game total doesn't fix this: it's a pace
# confound baked into every single game before rolling ever sees it. Rate
# stats (per attempt/carry/target) isolate efficiency instead.
#
# Each entry: numerator/denominator are lists of raw column names (summed if
# more than one). Definitions are skipped gracefully (with a printed
# warning) if the underlying raw columns aren't present.
#
# "pairing" controls which suffix the denominator uses:
#   - "same" (default): offense-side stats. A team's own passing_yards,
#     rushing_yards, etc. divide by that SAME team's own attempts/carries/
#     targets that game. When viewed from the "_allowed" side (what the
#     opponent did), both numerator and denominator are the opponent's own
#     numbers, so the suffix stays matched on both sides.
#   - "cross": defense-side stats (def_sacks, def_interceptions, etc.).
#     These describe THIS team's defensive production, but the meaningful
#     denominator is the OPPONENT's snap count that game (how many
#     dropbacks/carries did the opponent's offense have for this defense to
#     work with). Getting this backwards -- dividing by the same team's own
#     offensive attempts instead of the opponent's -- would silently produce
#     a nonsense rate, so it's made an explicit, separate case rather than
#     assumed.
# --------------------------------------------------------------------------
RATE_STAT_DEFINITIONS = [
    # -- passing offense (same-suffix: numerator and denominator are the same team's own game) --
    {"name": "passing_epa_per_play", "numerator": ["passing_epa"], "denominator": ["attempts"]},
    {"name": "completion_pct", "numerator": ["completions"], "denominator": ["attempts"]},
    {"name": "yards_per_pass_attempt", "numerator": ["passing_yards"], "denominator": ["attempts"]},
    {"name": "passing_td_rate", "numerator": ["passing_tds"], "denominator": ["attempts"]},
    {"name": "passing_interception_rate", "numerator": ["passing_interceptions"], "denominator": ["attempts"]},
    {"name": "sack_rate_suffered", "numerator": ["sacks_suffered"], "denominator": ["attempts"]},
    {"name": "passing_air_yards_per_attempt", "numerator": ["passing_air_yards"], "denominator": ["attempts"]},
    {"name": "passing_yac_per_completion", "numerator": ["passing_yards_after_catch"], "denominator": ["completions"]},
    {"name": "passing_first_down_rate", "numerator": ["passing_first_downs"], "denominator": ["attempts"]},

    # -- rushing offense (same-suffix) --
    {"name": "rushing_epa_per_play", "numerator": ["rushing_epa"], "denominator": ["carries"]},
    {"name": "yards_per_carry", "numerator": ["rushing_yards"], "denominator": ["carries"]},
    {"name": "rushing_td_rate", "numerator": ["rushing_tds"], "denominator": ["carries"]},
    {"name": "rushing_first_down_rate", "numerator": ["rushing_first_downs"], "denominator": ["carries"]},
    {"name": "rushing_fumble_rate", "numerator": ["rushing_fumbles"], "denominator": ["carries"]},

    # -- receiving offense (same-suffix) --
    {"name": "receiving_epa_per_target", "numerator": ["receiving_epa"], "denominator": ["targets"]},
    {"name": "catch_rate", "numerator": ["receptions"], "denominator": ["targets"]},
    {"name": "receiving_air_yards_per_target", "numerator": ["receiving_air_yards"], "denominator": ["targets"]},
    {"name": "receiving_yac_per_reception", "numerator": ["receiving_yards_after_catch"], "denominator": ["receptions"]},
    {"name": "receiving_first_down_rate", "numerator": ["receiving_first_downs"], "denominator": ["targets"]},

    # -- combined offense (same-suffix) --
    {
        "name": "total_offense_epa_per_play",
        "numerator": ["passing_epa", "rushing_epa"],
        "denominator": ["attempts", "carries"],
    },

    # -- defense (CROSS-suffix: denominator is the opponent's snaps that game) --
    {"name": "def_sack_rate", "numerator": ["def_sacks"], "denominator": ["attempts"], "pairing": "cross"},
    {"name": "def_interception_rate", "numerator": ["def_interceptions"], "denominator": ["attempts"], "pairing": "cross"},
    {
        "name": "def_tackles_for_loss_rate",
        "numerator": ["def_tackles_for_loss"],
        "denominator": ["attempts", "carries"],
        "pairing": "cross",
    },
    {"name": "def_qb_hit_rate", "numerator": ["def_qb_hits"], "denominator": ["attempts"], "pairing": "cross"},
    {"name": "def_pass_defended_rate", "numerator": ["def_pass_defended"], "denominator": ["attempts"], "pairing": "cross"},
]

# --------------------------------------------------------------------------
# Feature engineering toggles (see scripts/feature_engineering.py)
# --------------------------------------------------------------------------
# For every home_X / away_X rolled-feature pair, add diff_X = home_X - away_X.
# Relative advantage is often more predictive than either team's raw level,
# and most classifiers benefit from having it spelled out directly.
ADD_DIFFERENTIAL_FEATURES = True

# For every stat that has both a "_produced" and "_allowed" version, pair a
# team's offensive tendency with the opponent's tendency to allow that same
# stat, to approximate "what should happen when these two collide" rather
# than just looking at each side in isolation.
ADD_MATCHUP_FEATURES = True

# Pull rest days, roof/surface, temp/wind, and other schedules-level context
# nflreadpy already returns but the base pipeline wasn't using.
ADD_SCHEDULE_CONTEXT_FEATURES = True

# --------------------------------------------------------------------------
# Pruning thresholds (see scripts/03_prune_features.py)
# --------------------------------------------------------------------------
# A column with more than this % null gets flagged. PFR-derived columns are
# expected to be null before PFR_ADVSTATS_MIN_SEASON (2018) -- that's a
# known, accepted coverage gap (see README), not a defect. The pruning
# script re-checks null % restricted to season >= PFR_ADVSTATS_MIN_SEASON
# for any column with "pfr" in its name before deciding whether it's
# genuinely sparse or just pre-dates PFR coverage.
SPARSE_NULL_THRESHOLD_PCT = 50.0

# Within the PFR-coverage era, a PFR column still needs at least this much
# non-null data to be considered usable rather than genuinely broken/sparse.
SPARSE_NULL_THRESHOLD_PCT_WITHIN_PFR_ERA = 50.0

# --------------------------------------------------------------------------
# Feature analysis (see scripts/04_feature_analysis.py)
# --------------------------------------------------------------------------
UNIVARIATE_STATS_CSV = PROCESSED_DIR / "feature_univariate_stats.csv"
HIGH_CORRELATION_PAIRS_CSV = PROCESSED_DIR / "feature_high_correlation_pairs.csv"

TARGET_COLUMN = "home_cover"

# Columns that are alternate representations of the outcome, or straight-up
# leak it (margin, home_win, away_win, away_cover are all deterministic
# functions of home_cover / the score). These must NEVER be treated as
# predictors -- they get excluded from the analysis regardless of how
# strongly they'd otherwise "predict" the target.
OUTCOME_COLUMNS = {
    "home_score", "away_score", "margin",
    "home_win", "away_win", "home_cover", "away_cover",
}

# For the pairwise-correlation multicollinearity check: only run it across
# the top N columns by univariate significance, not the full predictor set.
# Running a full pairwise matrix across ~2,600 columns is mostly noise and
# structural duplication (diff_X / matchup_X are exact linear functions of
# the raw produced/allowed columns -- see README) rather than a meaningful
# VIF-style check. Narrowing to the columns that already showed *some*
# individual association with the target keeps this both fast and useful.
TOP_N_FOR_CORRELATION_CHECK = 300
HIGH_CORRELATION_THRESHOLD = 0.85

# --------------------------------------------------------------------------
# Baseline model (see scripts/05_train_baseline_model.py)
# --------------------------------------------------------------------------
MODEL_PATH = PROCESSED_DIR / "baseline_model.json"
MODEL_METRICS_JSON = PROCESSED_DIR / "baseline_model_metrics.json"
MODEL_FEATURE_IMPORTANCE_CSV = PROCESSED_DIR / "baseline_feature_importance.csv"
MODEL_KEPT_FEATURES_CSV = PROCESSED_DIR / "baseline_kept_features.csv"

# Chronological holdout, NOT a random split -- the model is being asked to
# predict games it has never seen in time, same as it will have to do live.
# A random split would let the model train on games from the SAME week as
# ones it's tested on, which leaks information a real deployment wouldn't
# have (e.g. two teams' rolling stats derived from overlapping game windows).
TEST_SEASONS = [2024, 2025]

# Pre-filter by relevance to the target BEFORE dedup. This is the fix for a
# concrete failure mode that showed up in practice: with ~3,000 candidate
# columns against ~3,600 training rows, correlation-dedup alone (which only
# removes near-DUPLICATE columns, not irrelevant ones) left 1,800+ columns
# in play -- still enough for XGBoost to memorize the training seasons
# perfectly (train AUC 0.97) while doing WORSE than a coin flip on the
# holdout (test AUC 0.48). That's overfitting, not signal. Ranking by
# |correlation| with the target on TRAIN ONLY and keeping just the top N
# forces the feature count into a range the sample size can actually
# support. This runs BEFORE correlation_dedup, which then only has to work
# within this much smaller, already-relevant set.
MAX_FEATURES_BEFORE_DEDUP = 150

# Greedy correlation-based dedup, applied AFTER the relevance pre-filter
# above. Columns are ranked by |correlation| with the target, then added to
# the kept set one at a time, skipping any column above this threshold
# correlated with a column already kept. This is what actually thins out
# the diff_X/matchup_X/produced-allowed structural duplication (e.g.
# sack_rate_suffered and def_sack_rate measuring the identical real-world
# sacks from two different box-score angles) before the model ever sees it.
CORRELATION_DEDUP_THRESHOLD = 0.90

# XGBoost hyperparameters. Deliberately conservative (shallow trees, heavy
# regularization, row/column subsampling) because after the chronological
# split there are ~3,000-3,500 training rows against what could still be
# several hundred features even after dedup -- a classic small-n/large-p
# setup where an unregularized model will happily memorize noise. Revisit
# these once you have a sense of how much the model is over/underfitting.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 5.0,
    "reg_alpha": 1.0,
    "min_child_weight": 10,
}

# --------------------------------------------------------------------------
# Curated model (see scripts/06_train_curated_model.py)
# --------------------------------------------------------------------------
# A deliberate contrast to the algorithmic top-150-then-dedup approach above.
# 05's correlation-based selection, run against ~3,000 columns with near-zero
# true effect sizes (per 04_feature_analysis.py), ended up picking features
# that fit the training seasons without holding up on the holdout (AUC 0.51,
# essentially chance). This is the opposite approach: a short, football-
# intuition-driven list -- EPA/play (offense and defense, passing/rushing/
# receiving), completion %, trench performance (sack rates), rest, dome, and
# spread magnitude -- rather than letting a correlation ranking fish through
# thousands of noisy columns.
#
# Turnover-rate features (interceptions, fumbles) were removed in one
# version of this list (turnovers are widely understood to be largely
# random/hard to predict -- a fumble recovery in particular is close to a
# coin flip) and then added back alongside the extra EPA differentials
# below. Worth being explicit about why that's a methodological yellow
# flag: they went back in largely because their absence made a borderline
# holdout result (consensus-pick accuracy at z=2.0, right at the edge of
# significance) disappear. Removing a feature for a principled reason
# (turnovers are ~random) and then reversing that decision because it
# changed a p-value is a step toward tuning the feature set to hit a
# threshold rather than testing a fixed hypothesis -- if this combination
# produces a "significant" result, that's a reason for MORE scrutiny, not
# less. Walk-forward validation on a feature set decided in advance, not
# iteratively adjusted, is what actually resolves this.
#
# abs_spread_line captures spread MAGNITUDE independent of which team is
# favored -- a plain linear term on spread_line alone can't represent big
# favorites behaving differently than small ones (e.g. backdoor covers in
# blowouts); this gives the logistic regression a direct way to test that
# with its own coefficient, rather than relying on XGBoost to maybe find
# the nonlinearity on its own.
#
# Deliberately built from base team_stats-derived rate stats only (NOT PFR
# columns), so every feature has full coverage across all of 2010-2025 with
# no pre-2018 NaN gaps -- keeps the logistic regression path simple (no
# missing-value handling needed) and makes this a genuinely independent test
# of a different feature-selection philosophy, not just a smaller random
# subset of the same PFR-heavy columns that didn't generalize.
#
# MATCHUP-BASED, not produced/allowed-based (changed after every produced/
# allowed-pair version of this list failed to show a walk-forward edge).
# The original version paired each team's own offense trend with its own
# defense trend as two SEPARATE features (diff_X_produced_ewma = home
# offense vs away offense, diff_X_allowed_ewma = home defense vs away
# defense). The intuition this switched to: what should actually matter for
# THIS game is home offense vs away DEFENSE and away offense vs home
# DEFENSE -- the specific matchup, not each side's stats in isolation.
# add_matchup_features() (feature_engineering.py) already builds exactly
# that: home_X_matchup_ewma = (home_X_produced_ewma + away_X_allowed_ewma) / 2,
# and symmetrically for away. diff_X_matchup_ewma = home_X_matchup_ewma -
# away_X_matchup_ewma is now auto-generated the same way every other diff_
# column is (engineer_all() runs matchup before differential specifically
# so this works).
#
# Worth being precise about what this change actually does mathematically,
# since it's easy to overstate: diff_X_matchup_ewma = 0.5 * (diff_X_produced_ewma
# - diff_X_allowed_ewma) -- an EXACT linear combination of the two features
# the old list already had. For the logistic regression, that means this
# isn't new information the model couldn't already see; it's a real change
# in what the model is ALLOWED to fit. The old list let the regression find
# its own weight on offense vs. defense (whatever the training data
# supported, including noise); this version hard-codes a 50/50 split a
# priori, using football judgment instead of the fit. Fewer effective
# parameters (9 matchup diffs replacing what was 13 separate produced/
# allowed diffs) is also a real, independent reason this could generalize
# better given how much this build's evidence points to overfitting on
# noise as the dominant risk, not underfitting. For XGBoost it's a genuine
# change either way -- a fixed 50/50 blend as its own feature vs. two raw
# ingredients it has to learn to combine via splits.
CURATED_FEATURES = [
    "spread_line",
    "abs_spread_line",
    "diff_total_offense_epa_per_play_matchup_ewma",
    "diff_passing_epa_per_play_matchup_ewma",
    "diff_rushing_epa_per_play_matchup_ewma",
    "diff_receiving_epa_per_target_matchup_ewma",
    "diff_completion_pct_matchup_ewma",
    "diff_sack_rate_suffered_matchup_ewma",
    "diff_def_sack_rate_matchup_ewma",
    "diff_yards_per_carry_matchup_ewma",
    "rest_diff",
    "is_dome",
]

CURATED_MODEL_METRICS_JSON = PROCESSED_DIR / "curated_model_metrics.json"
CURATED_LOGIT_SUMMARY_TXT = PROCESSED_DIR / "curated_logit_summary.txt"
CURATED_XGB_IMPORTANCE_CSV = PROCESSED_DIR / "curated_xgb_importance.csv"
CURATED_CONSENSUS_PICKS_CSV = PROCESSED_DIR / "curated_consensus_picks.csv"

# Much lighter regularization than XGB_PARAMS -- with only ~15 features
# against a few thousand rows, the small-n/large-p overfitting risk that
# justified the heavy regularization above mostly doesn't apply here.
XGB_PARAMS_CURATED = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_weight": 5,
}

# --------------------------------------------------------------------------
# Walk-forward validation (see scripts/07_walk_forward_validation.py)
# --------------------------------------------------------------------------
# 06's single 2024-2025 holdout is only ~550 games -- one slice of history,
# and results on it swung between "borderline significant" and "not" based
# on minor feature-set tweaks (see the CURATED_FEATURES comment above). That
# fragility is itself informative: a real, generalizable edge shouldn't
# hinge on whether one turnover feature is in or out. Walk-forward testing
# fixes the feature set in advance (no more tweaking after seeing results)
# and expands the effective test set to every season it can, which is both
# more honest and statistically more powerful.
#
# Design: expanding-window, season-by-season. For each test season s, train
# on every prior season (2010..s-1) and test on season s only, then advance.
# This mimics deployment exactly -- the model only ever sees data that would
# have actually been available at the time -- and produces one holdout
# season's worth of predictions per fold, which are pooled across all folds
# afterward for a much bigger combined sample than any single holdout.
#
# First test season chosen to guarantee a reasonable minimum training
# history (2010-2017 = 8 seasons) before the first fold is scored.
WALK_FORWARD_FIRST_TEST_SEASON = 2018

WALK_FORWARD_FOLD_SUMMARY_CSV = PROCESSED_DIR / "walk_forward_fold_summary.csv"
WALK_FORWARD_CONSENSUS_PICKS_CSV = PROCESSED_DIR / "walk_forward_consensus_picks.csv"
WALK_FORWARD_METRICS_JSON = PROCESSED_DIR / "walk_forward_metrics.json"

# --------------------------------------------------------------------------
# Market-edge evaluation (see scripts/08_edge_based_evaluation.py)
# --------------------------------------------------------------------------
# The walk-forward "consensus pick" methodology above (07) only asks "did
# logit and XGBoost predict the same side" -- it never looks at price. This
# is a different, more realistic methodology: convert the actual market
# price for each side of the spread bet (home_spread_odds / away_spread_odds,
# American odds, already pulled into training_set.csv by
# build_features.pivot_to_game_level) to an implied probability, regress the
# model's own probability toward that market probability, and only bet when
# the regressed probability clears the market by more than a threshold.
#
# The regression step matters: an untrained eye should not fully trust a
# model's raw probability against an efficient market (that market price
# already reflects a huge amount of information the model doesn't have --
# injuries, weather, sharp money, etc.). Blending the model's view with the
# market's own view, weighted mostly toward the market, produces a more
# honest probability estimate than either alone -- and the "edge" is then
# just how far that honest estimate sits from the price being offered.
#
# EDGE_MODEL_WEIGHT is the weight given to the model's own probability in
# that blend (market gets 1 - EDGE_MODEL_WEIGHT). At the default of 0.35,
# for example, a home edge of:
#   home_edge = EDGE_MODEL_WEIGHT * model_home_prob
#             + (1 - EDGE_MODEL_WEIGHT) * market_home_implied_prob
#             - market_home_implied_prob
# simplifies to EDGE_MODEL_WEIGHT * (model_home_prob - market_home_implied_prob)
# -- i.e. the raw model/market disagreement, scaled down to reflect how much
# the model's opinion should be trusted relative to the market's.
EDGE_MODEL_WEIGHT = 0.35

# Edge thresholds to sweep when reporting accuracy/ROI -- 0.03 (3%) was the
# threshold historically used to decide whether to release a pick at all.
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

EDGE_EVAL_PICKS_CSV = PROCESSED_DIR / "edge_eval_picks.csv"
EDGE_EVAL_THRESHOLD_SUMMARY_CSV = PROCESSED_DIR / "edge_eval_threshold_summary.csv"
EDGE_EVAL_METRICS_JSON = PROCESSED_DIR / "edge_eval_metrics.json"

# --------------------------------------------------------------------------
# Consensus + edge walk-forward (see scripts/09_consensus_edge_walk_forward.py)
# --------------------------------------------------------------------------
# Combines 07's agreement gate (logit and XGBoost must predict the same
# side) with 08's market-edge gate (the combined model's edge on that side
# must clear a minimum) into a single, stricter selection rule -- the idea
# being that a pick both models agree on AND that clears a real price
# discrepancy is a stronger claim than either condition alone.
CONSENSUS_EDGE_THRESHOLD = 0.02

CONSENSUS_EDGE_PICKS_CSV = PROCESSED_DIR / "consensus_edge_picks.csv"
CONSENSUS_EDGE_METRICS_JSON = PROCESSED_DIR / "consensus_edge_metrics.json"

# --------------------------------------------------------------------------
# Season backtest report (see scripts/10_season_backtest_report.py)
# --------------------------------------------------------------------------
# Renders one season's worth of a picks file (default: the consensus+edge
# rule's output) as a week-by-week record/units table with a running season
# total -- the actual shape a season rollout decision gets made from, not
# just a single pooled accuracy number.
SEASON_REPORT_CSV_TEMPLATE = str(PROCESSED_DIR / "season_report_{season}.csv")

# --------------------------------------------------------------------------
# Edge calibration check (see scripts/11_edge_calibration_check.py)
# --------------------------------------------------------------------------
# Diagnostic prompted by 07 (raw consensus, no price filter) scoring z=3.25
# while 09 (same models, same feature set, PLUS a >=2% edge requirement)
# scored z=0.60 -- the same underlying picks, filtered further, performing
# WORSE. That's backwards from what "edge" is supposed to mean (bigger
# edge = model and market disagree more = should be MORE informative, not
# less), and worth checking directly rather than just picking whichever
# result looks better.
EDGE_CALIBRATION_N_BUCKETS = 10
EDGE_CALIBRATION_DECILE_CSV = PROCESSED_DIR / "edge_calibration_deciles.csv"
EDGE_CALIBRATION_METRICS_JSON = PROCESSED_DIR / "edge_calibration_metrics.json"

# --------------------------------------------------------------------------
# Confidence calibration check (see scripts/12_confidence_calibration_check.py)
# --------------------------------------------------------------------------
# Same diagnostic as the edge calibration check, but measuring the MODEL's
# own raw confidence (|combined_prob - 0.5|, no market comparison at all)
# instead of edge-vs-market. Market-based edge already showed no
# relationship to accuracy (see 11's results) -- this checks whether the
# model being more confident in itself (independent of what the market
# thinks) is a more promising lever for cutting volume and raising ROI.
CONFIDENCE_CALIBRATION_N_BUCKETS = 10
CONFIDENCE_CALIBRATION_DECILE_CSV = PROCESSED_DIR / "confidence_calibration_deciles.csv"
CONFIDENCE_CALIBRATION_METRICS_JSON = PROCESSED_DIR / "confidence_calibration_metrics.json"

# --------------------------------------------------------------------------
# Pick-type breakdown (see scripts/13_pick_type_breakdown.py)
# --------------------------------------------------------------------------
# Prompted by both probability-based volume-reduction levers (market edge,
# self-confidence) coming back null -- checking a different kind of split:
# not "how confident is the model" but "in which KINDS of games does the
# consensus rule actually work." Home vs. away picks, and favorite vs.
# underdog picks (using spread_line's sign to determine which side was
# favored), are football-motivated, pre-registered categories rather than
# another round of threshold fishing.
PICK_TYPE_BREAKDOWN_CSV = PROCESSED_DIR / "pick_type_breakdown.csv"

# --------------------------------------------------------------------------
# Pick-type x edge breakdown (see scripts/14_pick_type_edge_breakdown.py)
# --------------------------------------------------------------------------
# 11's edge-calibration check found no relationship between edge and
# accuracy across ALL consensus picks pooled together. 13 then found a real
# split by favorite/underdog. This checks whether edge was actually
# informative all along, just masked by pooling favorites and underdogs
# together (e.g. if edge works for one group and is noise/inverted for the
# other, averaging them could wash out a real effect). Uses fewer buckets
# than 11 (quintiles, not deciles) since each favorite/underdog group is
# roughly a third to two-thirds the size of the full pooled set.
PICK_TYPE_EDGE_N_BUCKETS = 5
PICK_TYPE_EDGE_BREAKDOWN_CSV = PROCESSED_DIR / "pick_type_edge_breakdown.csv"

# --------------------------------------------------------------------------
# Combined favorite/underdog + edge rule (see scripts/15_combined_rule.py)
# --------------------------------------------------------------------------
# Built from two real findings, in order:
#   13: favorite picks are break-even (51.3% accuracy), underdog picks
#       carry virtually all the profit (56.0%) -- bet underdogs broadly.
#   14: WITHIN favorite picks specifically, edge and accuracy are
#       significantly NEGATIVELY correlated (p=0.027) -- low-edge favorites
#       hit ~57-59%, high-edge favorites collapse to ~40-48%. Underdog
#       picks showed no edge relationship (p=0.91), so underdogs are left
#       unfiltered by edge.
# Rule: take every underdog consensus pick, and every favorite consensus
# pick whose edge is <= FAVORITE_EDGE_MAX_THRESHOLD (chosen from the
# quintile boundary in 14's results where favorite accuracy was still
# strong, before the cliff at higher edges).
#
# IMPORTANT CAVEAT, stated plainly: this threshold was chosen by looking at
# quintile boundaries computed on THIS SAME backtest data. Evaluating the
# resulting rule on that same data is not a clean out-of-sample test of the
# threshold itself -- it's a reasonable first pass (does combining what
# 13+14 found produce a plausible improved backtest), not proof the exact
# 0.7% cutoff will hold going forward. A properly rigorous version would
# re-derive the threshold inside each walk-forward fold using only prior
# seasons' data -- worth doing before trusting this for real money, not
# before looking at what it shows here.
FAVORITE_EDGE_MAX_THRESHOLD = 0.007

COMBINED_RULE_PICKS_CSV = PROCESSED_DIR / "combined_rule_picks.csv"
COMBINED_RULE_METRICS_JSON = PROCESSED_DIR / "combined_rule_metrics.json"

# --------------------------------------------------------------------------
# Combined rule v2: separate home/away favorite thresholds (see scripts/18_combined_rule_v2.py)
# --------------------------------------------------------------------------
# Refines 15's single FAVORITE_EDGE_MAX_THRESHOLD (0.7% for all favorites)
# into two separate thresholds, following 17's finding that home favorites
# and away favorites behave differently: home favorites only look reliable
# in the lowest edge bucket (below ~0.4%; the next bucket up is already
# roughly break-even), while away favorites hold up through a wider range
# (up to ~1.3-1.4% edge) before collapsing.
#
# EVEN MORE of a look-ahead-derived threshold than 15's already-flagged
# cutoff: these two numbers came from bucket boundaries on an even smaller,
# further-split sample (89 home favorite picks in 3 buckets, ~30 each) --
# more prone to overfitting to this specific backtest than 15's single
# threshold was. Treat this as an exploratory refinement to compare against
# 15's version, not a more-trustworthy final answer just because it's more
# granular.
HOME_FAVORITE_EDGE_MAX_THRESHOLD = 0.004
AWAY_FAVORITE_EDGE_MAX_THRESHOLD = 0.013

COMBINED_RULE_V2_PICKS_CSV = PROCESSED_DIR / "combined_rule_v2_picks.csv"
COMBINED_RULE_V2_METRICS_JSON = PROCESSED_DIR / "combined_rule_v2_metrics.json"

# --------------------------------------------------------------------------
# Positive-edge rule (see scripts/19_positive_edge_rule.py)
# --------------------------------------------------------------------------
# 15/18 only filtered FAVORITE picks by edge -- underdog picks were taken
# unconditionally, with no edge floor at all. That's a real gap raised
# directly: (1) a real betting operation should require positive edge on
# EVERY pick, dog or favorite, not just favorites; (2) this build's edge
# numbers are computed against nflverse's historical odds (~standard -110),
# not Novig's actual live prices, which are expected to be more favorable
# in general -- meaning this backtest likely UNDERSTATES true live edge,
# not overstates it; (3) volume (~7-10 picks/week under 15/18) is too high,
# and the favorite/underdog split (~80/20) is more lopsided than desired.
# A single edge floor applied to EVERY pick addresses all three at once:
# it enforces "only bet +EV picks by the model's own estimate" everywhere,
# cuts volume, and should rebalance the favorite/underdog split back
# toward center since the cut is no longer favorite-only.
POSITIVE_EDGE_THRESHOLDS_SWEEP = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
POSITIVE_EDGE_RULE_SWEEP_CSV = PROCESSED_DIR / "positive_edge_rule_sweep.csv"
POSITIVE_EDGE_RULE_PICKS_CSV = PROCESSED_DIR / "positive_edge_rule_picks.csv"
POSITIVE_EDGE_RULE_METRICS_JSON = PROCESSED_DIR / "positive_edge_rule_metrics.json"

# --------------------------------------------------------------------------
# Market favorite/underdog base rate (see scripts/16_market_base_rate_check.py)
# --------------------------------------------------------------------------
# Model-free sanity check on WHY the model's picks skew toward underdogs
# (13's finding): computes the raw historical cover rate for favorites vs.
# underdogs across every game in the full dataset (not just consensus
# picks, not just the walk-forward test seasons) -- if underdogs genuinely
# cover more than 50% of the time in the real 2010-2025 data independent of
# any model, that confirms the skew reflects an actual market pattern the
# model picked up on, not an artifact of how this particular model or
# feature set was built.
MARKET_BASE_RATE_CSV = PROCESSED_DIR / "market_base_rate.csv"

# --------------------------------------------------------------------------
# Home/away favorite edge breakdown (see scripts/17_home_favorite_edge_breakdown.py)
# --------------------------------------------------------------------------
# 16 found the model-free market tilt away from favorites is concentrated
# in HOME favorites (48.3% cover, z=-1.68) rather than away favorites
# (49.8%, z=-0.15) -- a real, football-plausible pattern (public/market may
# specifically overvalue home favorites more than road favorites) worth
# checking against the model's own favorite picks and 14's edge finding
# specifically, rather than lumping all favorites together as 13/14 did.
# Home favorite picks are a much smaller group than away favorite picks, so
# fewer buckets are used for the home side to keep per-bucket samples large
# enough to read.
HOME_FAVORITE_EDGE_N_BUCKETS = 3
AWAY_FAVORITE_EDGE_N_BUCKETS = 5
HOME_AWAY_FAVORITE_EDGE_BREAKDOWN_CSV = PROCESSED_DIR / "home_away_favorite_edge_breakdown.csv"

# --------------------------------------------------------------------------
# Independent favorite/underdog edge floors (see scripts/20_independent_edge_rule.py)
# --------------------------------------------------------------------------
# 19's single shared edge floor solved concerns (1) positive edge everywhere
# and (2) volume, but NOT (3) class imbalance -- favorite_pct barely moved
# (31.1% -> 23-29%) as the shared threshold rose, because a uniform floor
# cuts favorites and underdogs roughly proportionally rather than
# rebalancing the split. This is the direct fix: sweep a favorite-side
# threshold and an underdog-side threshold INDEPENDENTLY (a grid, not one
# shared number), so a combination can be chosen that pulls favorite_pct
# toward center while still hitting a volume/accuracy target -- e.g. a
# lower bar for favorites (there are fewer of them and 13/14 showed real
# signal there at low edge) and a higher bar for underdogs (there are many
# more of them, so a tighter filter is needed to meaningfully thin them out
# and shift the mix).
FAVORITE_EDGE_THRESHOLDS_2D = [0.000, 0.005, 0.010, 0.015, 0.020]
UNDERDOG_EDGE_THRESHOLDS_2D = [0.000, 0.010, 0.020, 0.030, 0.040]
INDEPENDENT_EDGE_RULE_GRID_CSV = PROCESSED_DIR / "independent_edge_rule_grid.csv"
INDEPENDENT_EDGE_RULE_PICKS_CSV = PROCESSED_DIR / "independent_edge_rule_picks.csv"
INDEPENDENT_EDGE_RULE_METRICS_JSON = PROCESSED_DIR / "independent_edge_rule_metrics.json"

# --------------------------------------------------------------------------
# Three-way consensus (see scripts/21_three_way_consensus.py)
# --------------------------------------------------------------------------
# Prompted by: "could a third model help smooth all these issues?" Edge-based
# volume filters (19/20) have repeatedly shown null or backwards
# relationships to accuracy; the one thing that HAS worked all build is
# agreement between independently-biased models (07's 2-way consensus,
# z=3.25 pooled). This adds a third model -- Gaussian Naive Bayes, on the
# SAME curated feature set as logit/XGBoost -- and requires all three to
# agree, instead of adding a new feature set (which 05's baseline model
# already showed overfits badly at this sample size: train AUC 0.97, holdout
# AUC 0.48) or a new edge threshold.
#
# Why Naive Bayes specifically: logit is linear/additive, XGBoost is
# tree-based/nonlinear with learned interactions -- both good models, but
# correlated in HOW they can be wrong (whatever one framework's blind spot
# is, both were fit to the same 12 columns). Naive Bayes makes a completely
# different assumption (each feature is conditionally independent given the
# outcome), so its errors should be less correlated with the other two --
# closer to a genuinely independent third vote instead of another version of
# the same vote. It's also about as hard to overfit as logistic regression
# at this sample size (no interaction terms, no tree depth to tune).
THREE_WAY_FOLD_SUMMARY_CSV = PROCESSED_DIR / "three_way_fold_summary.csv"
THREE_WAY_CONSENSUS_PICKS_CSV = PROCESSED_DIR / "three_way_consensus_picks.csv"
THREE_WAY_METRICS_JSON = PROCESSED_DIR / "three_way_metrics.json"

# --------------------------------------------------------------------------
# Edge-only pick selection, no consensus gate (see scripts/22_edge_only_breakdown.py)
# --------------------------------------------------------------------------
# Every edge diagnostic so far (11, 14) checked edge's relationship to
# accuracy only WITHIN the consensus-gated subsample (games where logit and
# XGBoost already agreed). This checks a genuinely different selection rule:
# skip the agreement gate entirely, and for EVERY game, pick whichever side
# (home or away) has the higher edge (combined logit+xgb probability
# regressed toward the market per EDGE_MODEL_WEIGHT, same formula as
# 08/09/11/14/19/20) -- then bucket ALL of those picks by edge size to see
# whether edge alone, unfiltered by agreement, tracks accuracy.
EDGE_ONLY_N_BUCKETS = 10
EDGE_ONLY_BREAKDOWN_PICKS_CSV = PROCESSED_DIR / "edge_only_breakdown_picks.csv"
EDGE_ONLY_BREAKDOWN_DECILE_CSV = PROCESSED_DIR / "edge_only_breakdown_deciles.csv"
EDGE_ONLY_BREAKDOWN_METRICS_JSON = PROCESSED_DIR / "edge_only_breakdown_metrics.json"

# --------------------------------------------------------------------------
# Three-way consensus + edge-direction alignment (see scripts/23_three_way_edge_aligned.py)
# --------------------------------------------------------------------------
# Prompted directly by: "is there any way to combine these? all three have
# to agree and the average edge has to agree?" Two independently-validated
# filters, stacked:
#   21: raw >=0.5 classification agreement across all three models
#       (logit, xgb, nb) -- z=2.83 pooled, only modestly below 07's 2-way
#       z=3.25.
#   22: picking the side with the higher market-adjusted edge, on every
#       game (no agreement gate) -- z=1.97 pooled, a real but much weaker
#       signal on its own.
# These are NOT guaranteed to always agree even when 21's condition is met:
# raw threshold agreement (all three probabilities >= 0.5 for the same
# side) only checks the model's own view, while the edge-implied side
# (compute_edges, comparing home_edge vs away_edge) ALSO depends on how
# asymmetric the market's home/away prices are for that game -- the same
# distinction that caused the real bug fixed in 09 (see 09's docstring):
# the raw-threshold side and the edge-maximizing side can differ whenever
# home_spread_odds and away_spread_odds aren't priced symmetrically. This
# script keeps only picks where BOTH conditions -- 3-way raw agreement AND
# the edge-implied side (computed from the 3-model average probability)
# -- point to the same side, which is a stricter, more specific claim than
# either filter alone: not just "all three models like this side" but
# "all three models like this side AND that preference survives being
# checked against the actual asymmetric market price."
THREE_WAY_EDGE_ALIGNED_FOLD_SUMMARY_CSV = PROCESSED_DIR / "three_way_edge_aligned_fold_summary.csv"
THREE_WAY_EDGE_ALIGNED_PICKS_CSV = PROCESSED_DIR / "three_way_edge_aligned_picks.csv"
THREE_WAY_EDGE_ALIGNED_METRICS_JSON = PROCESSED_DIR / "three_way_edge_aligned_metrics.json"

# --------------------------------------------------------------------------
# Final combined rule: 3-way consensus + low-edge favorite filter (see scripts/24_final_combined_rule.py)
# --------------------------------------------------------------------------
# Consolidates everything found this build into one rule:
#   21: 3-way agreement (logit, xgb, nb) is the strongest, most-replicated
#       volume/quality lever available -- stronger than any edge threshold
#       tried (z=2.82 pooled vs. 07's 2-way z=3.25, a small quality give-up
#       for real volume reduction).
#   23: requiring edge-direction alignment on TOP of 3-way agreement added
#       complexity without adding value (removed only 1.4% of picks,
#       accuracy/z/ROI slightly WORSE, not better) -- left out of this rule
#       per the same Occam's-razor reasoning that favored 15 over 18.
#   13/14/16/17, replicated a third time on the 3-way-edge-aligned picks:
#       underdogs (especially HOME underdogs) carry essentially all the
#       profit; favorites are break-even to negative in aggregate, but
#       LOW-edge favorites specifically are strong (68.9% accuracy, +30.2%
#       ROI in the lowest edge bucket) while high-edge favorites collapse
#       (~42% accuracy, -16 to -18% ROI) -- now a p=0.005 result, the
#       strongest replication of this finding all build.
# Rule: take every 3-way-consensus underdog pick unfiltered, and every
# 3-way-consensus favorite pick whose edge (computed from the 3-model
# average probability) is <= THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD.
#
# SAME CAVEAT AS 15/18/19/20, stated plainly again: this threshold was
# chosen from a decile boundary computed on THIS SAME backtest. A properly
# rigorous version re-derives it inside each walk-forward fold using only
# prior seasons -- worth doing before real money, not before seeing what
# this shows.
THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD = 0.002

FINAL_COMBINED_RULE_PICKS_CSV = PROCESSED_DIR / "final_combined_rule_picks.csv"
FINAL_COMBINED_RULE_METRICS_JSON = PROCESSED_DIR / "final_combined_rule_metrics.json"

# --------------------------------------------------------------------------
# Live weekly scoring (see scripts/novig_client.py, scripts/25_live_weekly_scoring.py)
# --------------------------------------------------------------------------
# Everything above this point is the backtesting/research pipeline --
# scripts 01-24 all validate the model against seasons that already
# happened. This section is for the live path: score whatever NFL games
# haven't been played yet and publish picks the site can read.
#
# NOVIG_LEAGUE_FILTER must match however Novig tags NFL events in their
# GraphQL schema (confirmed "MLB" works for the MLB model; "NFL" is the
# expected analogous value but hasn't been confirmed against a live NFL
# slate yet -- run novig_client.py in --debug mode once games are on the
# board and check the printed event descriptions before trusting it).
NOVIG_GRAPHQL_URL = "https://gql.novig.us/v1/graphql"
NOVIG_LEAGUE_FILTER = "NFL"

# Team abbreviation mapping: Novig's team codes are NOT guaranteed to match
# nflverse's (the MLB client needed a _NOVIG_ABBR_MAP for exactly this
# reason -- KAN/CWS/WAS all differed). Populate this once real NFL odds are
# available and mismatches are visible in novig_client.py's --debug output.
# Empty by default -- most NFL team codes are expected to match directly.
NOVIG_TEAM_ABBR_MAP = {}

LIVE_NOVIG_ODDS_CSV = PROCESSED_DIR / "live_novig_odds.csv"
LIVE_SCORING_INPUT_CSV = PROCESSED_DIR / "live_scoring_input.csv"
LIVE_PICKS_CSV = PROCESSED_DIR / "live_picks.csv"
PREDICTIONS_JSON = ROOT_DIR / "predictions.json"

# Model/version string written into predictions.json's "version" field --
# bump this manually whenever CURATED_FEATURES, the model set (21's 3-way
# consensus), or THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD change, so the site
# (and anyone debugging a bad week) can tell which model logic produced a
# given predictions.json.
MODEL_VERSION = "v1.0-3way-consensus-low-edge-favorite"
