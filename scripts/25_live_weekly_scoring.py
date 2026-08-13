"""
Live weekly scoring: score the NEXT unplayed week of NFL games using
everything validated in the backtesting pipeline (01-24), and write
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

Rule applied, matching 24_final_combined_rule.py exactly:
  - Fit logit + XGBoost + Gaussian Naive Bayes on config.CURATED_FEATURES,
    using every played game in training_set.csv (all of config.SEASONS
    through the most recently completed week).
  - Score the upcoming week. Keep a game only if all three models agree on
    a side via the raw >=0.5 threshold (21's 3-way consensus).
  - Within that agreed set: keep every underdog pick; keep favorite picks
    only if edge (3-model average probability vs. Novig's market-implied
    probability) is <= config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD.
  - Pick'em games (spread_line == 0) are excluded, consistent with every
    backtest script.

CAVEAT carried over from 24: THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD was
derived from decile boundaries on backtest data, not a nested walk-forward
re-derivation. This live script applies that same fixed threshold -- update
config.py if/when a more rigorous threshold is derived.

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
from sklearn.naive_bayes import GaussianNB

import config
from build_features import build_full_dataset
from feature_engineering import engineer_all
from feature_utils import american_odds_to_implied_prob
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
    # Last-line-of-defense dtype check: if any column is still object dtype
    # here despite the earlier coercion (e.g. something introduced object
    # dtype AFTER curated_cols was coerced, such as sm.add_constant or the
    # standardization arithmetic), fail with a message that names the
    # actual offending column instead of statsmodels' generic "cast to
    # numpy dtype of object" error, which gives no hint where to look.
    bad_cols = [c for c in X_train_const.columns if X_train_const[c].dtype == object]
    if bad_cols:
        raise TypeError(
            f"X_train_const has {len(bad_cols)} object-dtype column(s) right before fitting: "
            f"{bad_cols}. This means dtype drift survived the earlier coercion in main() -- "
            f"inspect these columns' source data directly."
        )
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
    novig_odds = fetch_current_lines(upcoming[["game_id", "home_team", "away_team"]])
    novig_odds = novig_odds.dropna(subset=["spread_line"])
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

    # Defensive dtype check + coercion, applied right before any of these
    # columns get used. build_features.py/feature_engineering.py runs a
    # long chain of merges (extra PFR sources) and derived-column passes
    # (add_matchup_features: 796 cols, add_differential_features: 1194
    # cols) on top of raw data that's now fetched one season at a time --
    # if dtype drift creeps in ANYWHERE upstream of a curated column, the
    # arithmetic in those derived-feature passes can silently produce an
    # object-dtype column (no error at the time) that only surfaces much
    # later as statsmodels' cryptic "cast to numpy dtype of object" error.
    # Coercing here, with diagnostics, means a real run either fixes itself
    # or tells us exactly which column and why instead of failing blind.
    obj_curated = [c for c in curated_cols if full[c].dtype == object]
    if obj_curated:
        print(f"\nWARNING: {len(obj_curated)} curated feature column(s) came through as object "
              f"dtype (upstream dtype drift somewhere in build_features/feature_engineering or "
              f"a merged source): {obj_curated}")
        for c in obj_curated:
            sample = full[c].dropna()
            sample_types = sorted({type(v).__name__ for v in sample.head(20)})
            before_nan = full[c].isna().sum()
            full[c] = pd.to_numeric(full[c], errors="coerce")
            after_nan = full[c].isna().sum()
            print(f"  {c}: sample python types before coercion={sample_types}, "
                  f"n_nan before={before_nan}, after={after_nan}"
                  + (f"  !! {after_nan - before_nan} values weren't actually numeric -- "
                     f"inspect this column's source." if after_nan > before_nan else ""))

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

    # --- Fit logit + xgb + nb on ALL history, predict this week ---------
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

    nb_model = GaussianNB()
    nb_model.fit(X_train_std, y_train)
    nb_pred = nb_model.predict_proba(X_score_std)[:, 1]

    scoring = scoring.reset_index(drop=True)
    scoring["logit_prob"] = logit_pred
    scoring["xgb_prob"] = xgb_pred
    scoring["nb_prob"] = nb_pred
    scoring["avg_prob_3"] = (logit_pred + xgb_pred + nb_pred) / 3

    logit_class = (logit_pred >= 0.5).astype(int)
    xgb_class = (xgb_pred >= 0.5).astype(int)
    nb_class = (nb_pred >= 0.5).astype(int)
    agree_3way = (logit_class == xgb_class) & (logit_class == nb_class)
    scoring["predicted_home_cover"] = logit_class
    scoring["three_way_agree"] = agree_3way

    print(f"\n3-way agreement: {int(agree_3way.sum())} of {len(scoring)} games.")

    # --- Edge, exactly as in 24_final_combined_rule.py -------------------
    is_home = scoring["predicted_home_cover"] == 1
    model_prob_for_pick = np.where(is_home, scoring["avg_prob_3"], 1 - scoring["avg_prob_3"])
    picked_odds = np.where(is_home, scoring["home_spread_odds"], scoring["away_spread_odds"])
    implied_prob_for_pick = american_odds_to_implied_prob(pd.Series(picked_odds)).to_numpy()
    scoring["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)
    scoring["picked_odds"] = picked_odds

    picked_em = scoring["spread_line"] == 0
    picked_favorite = np.where(is_home, scoring["spread_line"] > 0, scoring["spread_line"] < 0)
    scoring["picked_favorite"] = picked_favorite

    keep_underdog = agree_3way & ~picked_favorite & ~picked_em
    keep_favorite = agree_3way & picked_favorite & ~picked_em & \
        (scoring["edge"] <= config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD)
    selected = keep_underdog | keep_favorite

    picks = scoring[selected].copy()
    print(f"Final rule picks this week: {len(picks)} of {len(scoring)} scored games "
          f"({int(keep_underdog.sum())} underdog, {int((keep_favorite).sum())} low-edge favorite).")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scoring.to_csv(config.LIVE_SCORING_INPUT_CSV, index=False)
    picks.to_csv(config.LIVE_PICKS_CSV, index=False)

    print(f"\n  -> {config.LIVE_SCORING_INPUT_CSV} (every scored game this week, picked or not)")
    print(f"  -> {config.LIVE_PICKS_CSV} (this week's final picks)")
    print(f"\nNext: python scripts/26_write_predictions_json.py")


if __name__ == "__main__":
    main()
