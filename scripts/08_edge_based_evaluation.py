"""
Market-edge evaluation: a different, more realistic way to turn model
probabilities into picks than 07's "consensus when both models agree."

07_walk_forward_validation.py's consensus-pick approach never looks at
price -- it just checks whether logit and XGBoost predict the same side.
This script does what the model was actually built to do (per the very
first design conversation for this project): compare the model's
probability to the market's own implied probability, in a form that
regresses the model toward the market rather than trusting it outright,
and only surface a pick when that regressed probability clears the market
by a real margin.

Method, per game, per side (home and away separately):
  1. Convert the market's actual price for that side (home_spread_odds /
     away_spread_odds -- American odds, e.g. -110) to an implied probability.
  2. Blend the model's own probability for that side with the market's
     implied probability, weighted mostly toward the market
     (config.EDGE_MODEL_WEIGHT, default 0.35 on the model / 0.65 on the
     market) -- this tempers the model's confidence with the fact that an
     efficient market price already encodes information the model doesn't
     have.
  3. edge = blended_prob - market_implied_prob. This simplifies to
     EDGE_MODEL_WEIGHT * (model_prob - market_implied_prob), i.e. the raw
     model/market disagreement scaled down by how much the model's opinion
     should be trusted.
  4. Take whichever side (home or away) has the higher edge as the pick for
     that game. A pick is only "given out" if its edge clears
     config.EDGE_THRESHOLDS (swept, not just checked at one cutoff -- the
     3% threshold historically used is one point on this curve, not the
     only one worth seeing).

This is evaluated three ways, exactly as requested: the logistic regression
alone, XGBoost alone, and a combined model (the two models' raw
probabilities averaged BEFORE the market-blend step, then blended and
edge-computed the same way as the individual models).

Uses the same expanding-window walk-forward folds as 07 (not a single
holdout) so this is evaluated across every out-of-sample season the data
supports, pooled into one sample per model -- same statistical-power
reasoning as 07.

Requires home_spread_odds / away_spread_odds to be present in
training_set.csv (build_features.pivot_to_game_level already pulls these
from nflreadpy's schedules if present -- re-run 02_build_training_set.py if
your local copy predates that). Games missing either price are dropped from
this analysis specifically (there's no market to compare the model to), with
a coverage report printed so it's clear how much of the data that affects.

Run:
    python scripts/08_edge_based_evaluation.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

import config
from feature_utils import american_odds_to_implied_prob, american_odds_to_profit_if_win, compute_edges


def fit_logit(X_train_const, y_train):
    try:
        model = sm.Logit(y_train, X_train_const).fit(disp=0)
        return model
    except np.linalg.LinAlgError:
        return sm.Logit(y_train, X_train_const).fit_regularized(alpha=1.0, disp=0)


def run_fold(train_df, test_df, curated_cols):
    target_train = train_df[config.TARGET_COLUMN]

    means = train_df[curated_cols].mean()
    stds = train_df[curated_cols].std().replace(0, 1)
    X_train_std = (train_df[curated_cols] - means) / stds
    X_test_std = (test_df[curated_cols] - means) / stds

    X_train_const = sm.add_constant(X_train_std)
    X_test_const = sm.add_constant(X_test_std, has_constant="add")
    logit_model = fit_logit(X_train_const, target_train.to_numpy())
    logit_test_pred = np.asarray(logit_model.predict(X_test_const))

    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS_CURATED, missing=np.nan)
    xgb_model.fit(train_df[curated_cols], target_train)
    xgb_test_pred = xgb_model.predict_proba(test_df[curated_cols])[:, 1]

    return logit_test_pred, xgb_test_pred


def main():
    df = pd.read_csv(config.TRAINING_SET_CSV, low_memory=False)
    df = df[df[config.TARGET_COLUMN].notna()].copy()
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    curated_cols = [c for c in config.CURATED_FEATURES if c in df.columns]
    missing = [c for c in config.CURATED_FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} curated features not found in training_set.csv: {missing}")
    print(f"Using {len(curated_cols)} curated features: {curated_cols}\n")

    before_dropna = len(df)
    df = df.dropna(subset=curated_cols + [config.TARGET_COLUMN])
    dropped = before_dropna - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in the curated set.")

    # --- market-odds coverage check, BEFORE dropping anything for it -- this
    # is worth seeing explicitly since it directly limits how much of the
    # dataset this analysis can even run on. ---
    odds_cols = ["home_spread_odds", "away_spread_odds"]
    missing_odds_cols = [c for c in odds_cols if c not in df.columns]
    if missing_odds_cols:
        raise ValueError(
            f"Missing column(s) {missing_odds_cols} in training_set.csv. These come from "
            "nflreadpy's load_schedules() via build_features.pivot_to_game_level() -- re-run "
            "02_build_training_set.py against a current build_features.py if this is stale, and "
            "confirm 01_fetch_historical.py's schedules pull actually includes these columns."
        )

    has_odds = df["home_spread_odds"].notna() & df["away_spread_odds"].notna()
    coverage_by_season = df.groupby("season")["game_id"].count().to_frame("n_games")
    coverage_by_season["n_with_odds"] = df[has_odds].groupby("season")["game_id"].count()
    coverage_by_season["n_with_odds"] = coverage_by_season["n_with_odds"].fillna(0).astype(int)
    coverage_by_season["pct_with_odds"] = (
        coverage_by_season["n_with_odds"] / coverage_by_season["n_games"] * 100
    ).round(1)
    print("\n--- Market spread-odds coverage by season ---")
    print(coverage_by_season.to_string())

    df = df[has_odds].copy()
    print(f"\n{len(df)} games have both home_spread_odds and away_spread_odds -- "
          f"this analysis is scoped to those games only.")
    if len(df) == 0:
        raise ValueError(
            "No games have market spread-odds data. This methodology can't run at all without "
            "it -- check that nflreadpy's schedules actually populate these columns for the "
            "seasons in config.SEASONS (odds coverage in nflverse's data is known to be sparser "
            "for older seasons; consider narrowing config.SEASONS if this analysis specifically "
            "needs to run)."
        )

    df["home_implied_prob"] = american_odds_to_implied_prob(df["home_spread_odds"])
    df["away_implied_prob"] = american_odds_to_implied_prob(df["away_spread_odds"])

    all_seasons = sorted(int(s) for s in df["season"].unique())
    test_seasons = [s for s in all_seasons if s >= config.WALK_FORWARD_FIRST_TEST_SEASON]
    if not test_seasons:
        raise ValueError(
            f"No seasons >= WALK_FORWARD_FIRST_TEST_SEASON ({config.WALK_FORWARD_FIRST_TEST_SEASON}) "
            f"with market odds. Seasons present: {all_seasons}"
        )
    print(f"\nWalk-forward folds (odds-covered seasons only): {test_seasons}\n")

    all_picks = []
    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        logit_pred, xgb_pred = run_fold(train_df, test_df, curated_cols)
        logit_pred = pd.Series(logit_pred, index=test_df.index)
        xgb_pred = pd.Series(xgb_pred, index=test_df.index)
        combined_pred = (logit_pred + xgb_pred) / 2

        home_implied = test_df["home_implied_prob"]
        away_implied = test_df["away_implied_prob"]
        actual_home_cover = test_df[config.TARGET_COLUMN]

        for model_name, model_pred in [("logit", logit_pred), ("xgb", xgb_pred), ("combined", combined_pred)]:
            edges = compute_edges(model_pred, home_implied, away_implied)
            picked_odds = np.where(
                edges["picked_side"].to_numpy() == "home",
                test_df["home_spread_odds"].to_numpy(),
                test_df["away_spread_odds"].to_numpy(),
            )
            correct = np.where(
                edges["picked_side"].to_numpy() == "home",
                actual_home_cover.to_numpy() == 1,
                actual_home_cover.to_numpy() == 0,
            )
            fold_picks = pd.DataFrame({
                "model": model_name,
                "season": test_season,
                "week": test_df["week"].to_numpy(),
                "game_id": test_df["game_id"].to_numpy() if "game_id" in test_df.columns else np.arange(len(test_df)),
                "picked_side": edges["picked_side"].to_numpy(),
                "edge": edges["picked_edge"].to_numpy(),
                "odds": picked_odds,
                "correct": correct.astype(int),
            })
            all_picks.append(fold_picks)

        print(f"Fold {test_season}: {len(test_df)} games scored for logit, xgb, and combined")

    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df["profit_if_win"] = american_odds_to_profit_if_win(picks_df["odds"])
    picks_df["profit"] = np.where(picks_df["correct"] == 1, picks_df["profit_if_win"], -1.0)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(config.EDGE_EVAL_PICKS_CSV, index=False)

    print(f"\n{len(picks_df) // 3} games x 3 models = {len(picks_df)} scored picks pooled across "
          f"{len(test_seasons)} walk-forward folds.")

    # --- accuracy + ROI at every edge threshold, per model ---
    summary_rows = []
    for model_name in ["logit", "xgb", "combined"]:
        model_picks = picks_df[picks_df["model"] == model_name]
        for threshold in config.EDGE_THRESHOLDS:
            eligible = model_picks[model_picks["edge"] >= threshold]
            n = len(eligible)
            if n == 0:
                summary_rows.append({
                    "model": model_name, "edge_threshold": threshold, "n_picks": 0,
                    "accuracy": float("nan"), "accuracy_se": float("nan"), "roi": float("nan"),
                })
                continue
            accuracy = eligible["correct"].mean()
            se = (accuracy * (1 - accuracy) / n) ** 0.5
            roi = eligible["profit"].mean()
            summary_rows.append({
                "model": model_name,
                "edge_threshold": threshold,
                "n_picks": n,
                "accuracy": accuracy,
                "accuracy_se": se,
                "roi": roi,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.EDGE_EVAL_THRESHOLD_SUMMARY_CSV, index=False)

    print("\n--- Accuracy and ROI by model and edge threshold ---")
    print("(accuracy = win rate on picks with edge >= threshold; roi = average units won per unit")
    print(" staked, using each pick's ACTUAL price -- 0.0 is break-even, negative means losing money")
    print(" even if accuracy is > 50%, since -110 requires ~52.4% just to break even)\n")
    for model_name in ["logit", "xgb", "combined"]:
        print(f"{model_name}:")
        sub = summary_df[summary_df["model"] == model_name]
        for _, row in sub.iterrows():
            if row["n_picks"] == 0:
                print(f"  edge >= {row['edge_threshold']*100:.0f}%: no picks")
                continue
            print(
                f"  edge >= {row['edge_threshold']*100:.0f}%: n={int(row['n_picks']):4d}  "
                f"accuracy={row['accuracy']*100:5.1f}% (+/-{row['accuracy_se']*100:.1f}pt)  "
                f"roi={row['roi']*100:+6.1f}%"
            )
        print()

    with open(config.EDGE_EVAL_METRICS_JSON, "w") as f:
        json.dump({
            "edge_model_weight": config.EDGE_MODEL_WEIGHT,
            "edge_thresholds": config.EDGE_THRESHOLDS,
            "test_seasons": test_seasons,
            "n_games_with_odds": len(df),
            "threshold_summary": summary_rows,
        }, f, indent=2)

    print(f"  -> {config.EDGE_EVAL_PICKS_CSV}")
    print(f"  -> {config.EDGE_EVAL_THRESHOLD_SUMMARY_CSV}")
    print(f"  -> {config.EDGE_EVAL_METRICS_JSON}")


if __name__ == "__main__":
    main()
