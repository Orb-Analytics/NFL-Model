"""
Live weekly scoring: score the NEXT unplayed week of NFL games using
everything validated in the backtesting pipeline (01-33), and write
data/processed/live_picks.csv -- the input to 26_write_predictions_json.py.

This is the first script in this repo that scores games that haven't been
played yet, rather than backtesting games that already have. It reuses the
exact same feature pipeline as training (build_features.py,
feature_engineering.py) -- deliberately, per build_features.py's module
docstring: "same functions, same logic... the single best defense against
train/serve skew." The only new logic here is (1) overriding
schedules.parquet's spread_line/odds with Novig's live line for the
upcoming week before features are built, and (2) fitting FINAL production
models on ALL available history instead of a walk-forward split (there's
no "future fold" to hold out for live scoring -- the walk-forward folds in
07/21 exist purely to validate the METHOD; this step trusts that method and
applies it for real).

RULE v2 (replaces the v1 3-way-consensus + raw-edge rule that used to live
here -- see config.py's "Live production rule v2" comment for the full
derivation and caveats):
  - Fit logit + XGBoost on config.CURATED_FEATURES (NO Gaussian Naive
    Bayes, NO 3-way agreement gate -- this is the 2-model population
    28_devigged_edge_breakdown.py backtested), using every played game in
    training_set.csv (all of config.SEASONS through the most recently
    completed week).
  - Score the upcoming week. For each game, compute the DE-VIGGED edge
    (feature_utils.compute_edges_devigged) for both sides and pick whichever
    side has the higher edge.
  - Keep the pick if it's an underdog and edge >= config.LIVE_UNDERDOG_EDGE_MIN;
    keep it if it's a favorite and edge >= config.LIVE_FAVORITE_EDGE_MIN.
  - Pick'em games (spread_line == 0) are excluded, consistent with every
    backtest script.

CAVEATS carried over from config.py: both edge-min thresholds were chosen
by eyeballing volume/quintile boundaries on backtest data, not a nested
walk-forward re-derivation, and the 8-season backtest that validated this
rule leaned heavily on one strong season (2021). Watch real-world results
accordingly -- update config.py if/when a more rigorous threshold is derived.

Prerequisites:
  1. python scripts/01_fetch_historical.py   (pulls current season too, per config.SEASONS)
  2. python scripts/novig_client.py --debug  (confirm the spread-market parsing format
     against a real slate BEFORE trusting this script's odds -- see novig_client.py's docstring)

Run:
    python scripts/25_live_weekly_scoring.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

import config
from build_features import build_full_dataset
from feature_engineering import engineer_all
from feature_utils import american_odds_to_implied_prob, compute_edges_devigged
from novig_client import fetch_current_lines


def load_extra_sources() -> dict[str, pd.DataFrame]:
    sources = {}
    if config.INCLUDE_PFR_ADVSTATS:
        for stat_type in config.PFR_ADVSTATS_TYPES:
            path = Path(str(config.RAW_PFR_ADVSTATS_PATH).format(stat_type=stat_type))
            if path.exists():
                sources[f"pfr_{stat_type}"] = pd.read_parquet(path)
    return sources


def fit_logit(X_train_const, y_train):
    # Last-line-of-defense: force plain numpy float64 regardless of current
    # dtype, rather than only checking for `== object` (confirmed
    # insufficient -- a real failure got past that check silently because
    # the actual culprit was a pandas nullable extension dtype, not
    # classic `object`). If astype itself fails, name the exact column
    # instead of letting statsmodels' generic "cast to numpy dtype of
    # object" error through with no hint where to look.
    try:
        X_train_const = X_train_const.astype("float64")
    except (ValueError, TypeError) as e:
        bad = {}
        for c in X_train_const.columns:
            try:
                X_train_const[c].astype("float64")
            except (ValueError, TypeError) as col_e:
                bad[c] = str(col_e)
        raise TypeError(
            f"Could not convert column(s) to float64 right before fitting: {list(bad.keys())}. "
            f"Per-column errors: {bad}. Original: {e}"
        ) from e
    try:
        return sm.Logit(y_train, X_train_const).fit(disp=0)
    except np.linalg.LinAlgError:
        return sm.Logit(y_train, X_train_const).fit_regularized(alpha=1.0, disp=0)


def main():
    if not config.RAW_SCHEDULES_PATH.exists() or not config.RAW_TEAM_STATS_PATH.exists():
        raise FileNotFoundError(
            f"{config.RAW_SCHEDULES_PATH} / {config.RAW_TEAM_STATS_PATH} not found. "
            "Run 01_fetch_historical.py first (config.SEASONS includes the current season)."
        )

    schedules = pd.read_parquet(config.RAW_SCHEDULES_PATH)
    team_stats = pd.read_parquet(config.RAW_TEAM_STATS_PATH)

    if not config.INCLUDE_POSTSEASON and "game_type" in schedules.columns:
        schedules_reg = schedules[schedules["game_type"] == "REG"]
    else:
        schedules_reg = schedules

    # --- Find the next unplayed week for the current season -------------
    current = schedules_reg[schedules_reg["season"] == config.CURRENT_SEASON]
    unplayed = current[current["home_score"].isna()]
    if unplayed.empty:
        print(f"No unplayed {config.CURRENT_SEASON} games found in the schedule -- "
              "either the season is over, or 01_fetch_historical.py needs a re-run.")
        return

    next_week = int(unplayed["week"].min())
    upcoming = unplayed[unplayed["week"] == next_week].copy()
    upcoming_game_ids = set(upcoming["game_id"])
    print(f"Scoring {config.CURRENT_SEASON} week {next_week}: {len(upcoming)} games.\n")

    # --- Override this week's spread_line / odds with Novig's live line -
    # BEFORE building features, so abs_spread_line and every downstream
    # column derived from spread_line reflect the real price being bet,
    # not whatever (possibly stale/placeholder) odds nflreadpy's schedules
    # pull happens to carry for a future game.
    novig_odds_raw = fetch_current_lines(upcoming[["game_id", "home_team", "away_team"]])
    # Write the raw fetch (including unmatched games with null odds) to
    # disk immediately -- this is the file the weekly workflow commits as
    # an audit trail of what was actually bet against. Previously this was
    # only ever written when running novig_client.py standalone with
    # --out; calling fetch_current_lines() directly from here (as this
    # script does) never persisted it, so the workflow's `git add` step
    # for data/processed/live_novig_odds.csv failed every single run with
    # "did not match any files" since the file never existed. Writing it
    # here, and BEFORE the early-return-on-zero-matches path below, so the
    # file always exists once this script gets this far regardless of
    # match rate.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    novig_odds_raw.to_csv(config.LIVE_NOVIG_ODDS_CSV, index=False)
    print(f"  -> {config.LIVE_NOVIG_ODDS_CSV}")

    novig_odds = novig_odds_raw.dropna(subset=["spread_line"])
    n_matched = len(novig_odds)
    print(f"Novig matched odds for {n_matched} of {len(upcoming)} games in week {next_week}.")
    if n_matched == 0:
        print("No Novig odds matched -- nothing to score this week (can't compute edge without a "
              "market price). Run scripts/novig_client.py --debug to diagnose.")
        return

    schedules = schedules.set_index("game_id")
    for _, row in novig_odds.iterrows():
        gid = row["game_id"]
        if gid not in schedules.index:
            continue
        schedules.loc[gid, "spread_line"] = row["spread_line"]
        schedules.loc[gid, "home_spread_odds"] = row["home_spread_odds"]
        schedules.loc[gid, "away_spread_odds"] = row["away_spread_odds"]
    schedules = schedules.reset_index()

    matched_game_ids = set(novig_odds["game_id"])

    # --- Build features the exact same way training does ----------------
    extra_sources = load_extra_sources()
    full = build_full_dataset(schedules, team_stats, extra_sources=extra_sources)
    full = engineer_all(full)

    curated_cols = [c for c in config.CURATED_FEATURES if c in full.columns]
    missing = [c for c in config.CURATED_FEATURES if c not in full.columns]
    if missing:
        print(f"WARNING: {len(missing)} curated features missing from the live feature set: {missing}")

    # Force every curated column to plain numpy float64, unconditionally --
    # NOT gated on an `== object` dtype check. Confirmed from a real failed
    # run: a curated column can come through as a pandas NULLABLE extension
    # dtype (Float64/Int64/boolean -- capital letters, distinct from numpy's
    # int64/float64), which nflreadpy's polars-backed `.to_pandas()` can
    # produce anywhere in the merge/feature chain. Extension dtypes are NOT
    # reported as `object` by pandas (an `== object` check silently misses
    # them, as happened here -- no warning printed, yet statsmodels still
    # failed converting the whole DataFrame block to one numpy array).
    # astype("float64") normalizes any of {extension dtype, object, plain
    # numpy} to one guaranteed-safe dtype in a single step, so detection
    # doesn't need to be exhaustive -- just always do it.
    non_standard = {c: str(full[c].dtype) for c in curated_cols
                    if str(full[c].dtype) not in ("float64", "int64", "float32", "int32")}
    if non_standard:
        print(f"\nNormalizing {len(non_standard)} curated feature column(s) with non-plain-numpy "
              f"dtype to float64 (pandas nullable extension dtypes from polars->pandas conversion "
              f"are a known cause -- these don't show up as 'object' dtype but still break "
              f"statsmodels' array conversion the same way): {non_standard}")
    try:
        full[curated_cols] = full[curated_cols].astype("float64")
    except (ValueError, TypeError) as e:
        bad = {}
        for c in curated_cols:
            try:
                full[c].astype("float64")
            except (ValueError, TypeError) as col_e:
                bad[c] = str(col_e)
        raise TypeError(
            f"Could not normalize curated column(s) to float64: {list(bad.keys())}. "
            f"Per-column errors: {bad}. Original: {e}"
        ) from e

    training = full[full["home_score"].notna() & full["away_score"].notna()].copy()
    training = training.dropna(subset=curated_cols + [config.TARGET_COLUMN])
    print(f"Fitting final production models on {len(training)} completed games "
          f"(all of {config.SEASONS[0]}-{config.CURRENT_SEASON} played so far).")

    scoring = full[full["game_id"].isin(matched_game_ids)].copy()
    scoring_before = len(scoring)
    scoring = scoring.dropna(subset=curated_cols)
    if len(scoring) < scoring_before:
        print(f"Dropped {scoring_before - len(scoring)} of this week's games -- missing curated "
              f"feature values (likely a team with too little game history, e.g. season opener "
              f"with no prior-season data available).")
    if scoring.empty:
        print("No scoreable games remain this week after feature completeness check.")
        return

    # --- Fit logit + xgb on ALL history, predict this week --------------
    # 2-model average only (v2 rule) -- no Gaussian Naive Bayes, no 3-way
    # agreement gate. See config.py's "Live production rule v2" comment.
    X_train = training[curated_cols]
    y_train = training[config.TARGET_COLUMN]
    means = X_train.mean()
    stds = X_train.std().replace(0, 1)
    X_train_std = (X_train - means) / stds
    X_score_std = (scoring[curated_cols] - means) / stds

    X_train_const = sm.add_constant(X_train_std)
    X_score_const = sm.add_constant(X_score_std, has_constant="add")
    logit_model = fit_logit(X_train_const, y_train.to_numpy())
    logit_pred = np.asarray(logit_model.predict(X_score_const))

    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS_CURATED, missing=np.nan)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(scoring[curated_cols])[:, 1]

    scoring = scoring.reset_index(drop=True)
    scoring["logit_prob"] = logit_pred
    scoring["xgb_prob"] = xgb_pred
    scoring["avg_prob"] = (logit_pred + xgb_pred) / 2

    # --- Edge, de-vigged, exactly as backtested in 28/31/32/33 -----------
    # feature_utils.compute_edges_devigged normalizes home/away implied
    # probability to sum to 1 BEFORE comparing against the model, then picks
    # whichever side has the higher edge. This is now both the SELECTION
    # basis (which games get picked) and the DISPLAY basis (what
    # 26_write_predictions_json.py shows as "Win Probability"/"Edge") --
    # unlike the old v1 rule, there is no longer a separate raw-vig
    # selection edge and de-vigged display edge; de-vigging is load-bearing
    # for selection now, confirmed in config.py's comment (raw edge at the
    # 2% threshold was NOT statistically significant; de-vigged edge was).
    home_implied_raw = american_odds_to_implied_prob(scoring["home_spread_odds"])
    away_implied_raw = american_odds_to_implied_prob(scoring["away_spread_odds"])
    edges = compute_edges_devigged(
        pd.Series(scoring["avg_prob"].to_numpy(), index=scoring.index),
        home_implied_raw, away_implied_raw,
    )
    is_home = edges["picked_side"].to_numpy() == "home"
    scoring["predicted_home_cover"] = is_home.astype(int)
    scoring["edge"] = edges["picked_edge"].to_numpy()
    scoring["market_implied_prob_devigged"] = edges["picked_market_implied"].to_numpy()
    scoring["confidence"] = edges["picked_confidence"].to_numpy()
    picked_odds = np.where(is_home, scoring["home_spread_odds"], scoring["away_spread_odds"])
    scoring["picked_odds"] = picked_odds

    picked_em = scoring["spread_line"] == 0
    picked_favorite = np.where(is_home, scoring["spread_line"] > 0, scoring["spread_line"] < 0)
    scoring["picked_favorite"] = picked_favorite

    keep_underdog = ~picked_favorite & ~picked_em & (scoring["edge"] >= config.LIVE_UNDERDOG_EDGE_MIN)
    keep_favorite = picked_favorite & ~picked_em & (scoring["edge"] >= config.LIVE_FAVORITE_EDGE_MIN)
    selected = keep_underdog | keep_favorite

    picks = scoring[selected].copy()
    print(f"Final rule picks this week: {len(picks)} of {len(scoring)} scored games "
          f"({int(keep_underdog.sum())} underdog >= {config.LIVE_UNDERDOG_EDGE_MIN:.0%} edge, "
          f"{int(keep_favorite.sum())} favorite >= {config.LIVE_FAVORITE_EDGE_MIN:.0%} edge).")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scoring.to_csv(config.LIVE_SCORING_INPUT_CSV, index=False)
    picks.to_csv(config.LIVE_PICKS_CSV, index=False)

    print(f"\n  -> {config.LIVE_SCORING_INPUT_CSV} (every scored game this week, picked or not)")
    print(f"  -> {config.LIVE_PICKS_CSV} (this week's final picks)")
    print(f"\nNext: python scripts/26_write_predictions_json.py")


if __name__ == "__main__":
    main()
