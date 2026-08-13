"""
Consensus + market-edge walk-forward: the strictest pick-selection rule
tested yet in this build, combining 07's agreement gate with 08's price
gate into one rule.

07_walk_forward_validation.py asks only "do logit and XGBoost predict the
same side" (agreement, no price check). 08_edge_based_evaluation.py asks
only "does this side's edge against the market clear a threshold" (price
check, no agreement requirement, and evaluated per-model rather than
requiring the models to agree). This script requires BOTH at once:

  1. Logistic regression and XGBoost must predict the SAME side (the
     agreement test from 07).
  2. The COMBINED model's edge on that agreed side -- combined meaning the
     two models' raw probabilities averaged BEFORE the market-blend step,
     same definition as "combined" in 08 -- must be >=
     config.CONSENSUS_EDGE_THRESHOLD (default 2%).

The logic for why this is a reasonable thing to test: a pick where two
independently-trained models land on the same side AND that side clears a
real price discrepancy against the market is a stronger claim than either
condition alone. It's also a MUCH smaller, more selective set of games than
either 07 or 08 individually produce -- watch n_picks per fold closely,
since this rule can plausibly produce very few (or zero) picks in some
seasons, and small-n accuracy claims are exactly what earlier scripts in
this build have shown to be unreliable.

Same expanding-window walk-forward folds as 07/08, same 12-feature
CURATED_FEATURES set (fixed in advance, not tuned against these results),
same requirement that games have both home_spread_odds and away_spread_odds
to be eligible at all.

IMPORTANT fix (post the original version of this script): "agree" here now
means exactly what it means in 07 -- logit_prob >= 0.5 and xgb_prob >= 0.5
land on the same side. The original version of this script instead compared
each model's EDGE-implied side (from compute_edges, which can differ from
the raw >=0.5 side whenever home/away market prices aren't symmetric) --
meaning 07 and 09 were silently using two different definitions of
"agreement," which was a real, separate reason they could diverge beyond
just the edge filter itself. Fixed so the only difference between 07 and
09 now is the added edge requirement, as documented above.

Run:
    python scripts/09_consensus_edge_walk_forward.py
    python scripts/09_consensus_edge_walk_forward.py --threshold 0.0   # override config.CONSENSUS_EDGE_THRESHOLD
"""

import argparse
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
from feature_utils import american_odds_to_implied_prob, american_odds_to_profit_if_win


def fit_logit(X_train_const, y_train):
    try:
        return sm.Logit(y_train, X_train_const).fit(disp=0)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override config.CONSENSUS_EDGE_THRESHOLD (as a fraction, e.g. 0.0 for >=0%% edge, "
             "0.02 for the default >=2%%).",
    )
    args = parser.parse_args()
    edge_threshold = args.threshold if args.threshold is not None else config.CONSENSUS_EDGE_THRESHOLD

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

    odds_cols = ["home_spread_odds", "away_spread_odds"]
    missing_odds_cols = [c for c in odds_cols if c not in df.columns]
    if missing_odds_cols:
        raise ValueError(
            f"Missing column(s) {missing_odds_cols} in training_set.csv -- re-run "
            "02_build_training_set.py against a current build_features.py."
        )

    has_odds = df["home_spread_odds"].notna() & df["away_spread_odds"].notna()
    df = df[has_odds].copy()
    print(f"{len(df)} games have both home_spread_odds and away_spread_odds -- "
          f"scoped to those games only.\n")
    if len(df) == 0:
        raise ValueError("No games have market spread-odds data -- this methodology can't run without it.")

    df["home_implied_prob"] = american_odds_to_implied_prob(df["home_spread_odds"])
    df["away_implied_prob"] = american_odds_to_implied_prob(df["away_spread_odds"])

    all_seasons = sorted(int(s) for s in df["season"].unique())
    test_seasons = [s for s in all_seasons if s >= config.WALK_FORWARD_FIRST_TEST_SEASON]
    if not test_seasons:
        raise ValueError(
            f"No seasons >= WALK_FORWARD_FIRST_TEST_SEASON ({config.WALK_FORWARD_FIRST_TEST_SEASON}) "
            f"with market odds. Seasons present: {all_seasons}"
        )
    print(f"Walk-forward folds: {test_seasons}")
    print(f"Selection rule: logit and xgb must agree on side (raw >=0.5 threshold, same "
          f"definition as 07), AND combined edge on that side >= {edge_threshold*100:.0f}%\n")

    all_picks = []
    fold_rows = []
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

        # Agreement uses the exact same raw >=0.5 threshold as 07 -- NOT each
        # model's edge-implied side (compute_edges picks whichever side has
        # the higher edge, which can differ from the raw threshold side when
        # home/away market prices aren't symmetric). Fixing this so 07 and
        # 09 use an identical definition of "agree," and the ONLY difference
        # between the two scripts is the added edge requirement.
        logit_side_home = logit_pred.to_numpy() >= 0.5
        xgb_side_home = xgb_pred.to_numpy() >= 0.5
        agree_mask = logit_side_home == xgb_side_home
        agreed_side_home = logit_side_home  # == xgb_side_home wherever agree_mask is True

        home_implied_arr = home_implied.to_numpy()
        away_implied_arr = away_implied.to_numpy()

        def edge_for_agreed_side(prob_home_arr):
            model_prob = np.where(agreed_side_home, prob_home_arr, 1 - prob_home_arr)
            implied = np.where(agreed_side_home, home_implied_arr, away_implied_arr)
            return config.EDGE_MODEL_WEIGHT * (model_prob - implied)

        combined_edge = edge_for_agreed_side(combined_pred.to_numpy())
        logit_edge = edge_for_agreed_side(logit_pred.to_numpy())
        xgb_edge = edge_for_agreed_side(xgb_pred.to_numpy())

        clears_edge = combined_edge >= edge_threshold
        selected = agree_mask & clears_edge

        n_test = len(test_df)
        n_agree = int(agree_mask.sum())
        n_selected = int(selected.sum())

        if n_selected > 0:
            picked_side = np.where(agreed_side_home[selected], "home", "away")
            picked_odds = np.where(
                picked_side == "home",
                test_df["home_spread_odds"].to_numpy()[selected],
                test_df["away_spread_odds"].to_numpy()[selected],
            )
            actual = actual_home_cover.to_numpy()[selected]
            correct = np.where(picked_side == "home", actual == 1, actual == 0)

            fold_picks = pd.DataFrame({
                "season": test_season,
                "week": test_df["week"].to_numpy()[selected],
                "game_id": (
                    test_df["game_id"].to_numpy()[selected]
                    if "game_id" in test_df.columns
                    else np.arange(n_selected)
                ),
                "picked_side": picked_side,
                "combined_edge": combined_edge[selected],
                "logit_edge": logit_edge[selected],
                "xgb_edge": xgb_edge[selected],
                "odds": picked_odds,
                "correct": correct.astype(int),
            })
            all_picks.append(fold_picks)
            fold_accuracy = correct.mean()
        else:
            fold_accuracy = float("nan")

        fold_rows.append({
            "test_season": test_season,
            "n_test": n_test,
            "n_agree": n_agree,
            "n_selected": n_selected,
            "fold_accuracy": fold_accuracy,
        })
        print(
            f"Fold {test_season}: {n_test} games, {n_agree} agreed on side, "
            f"{n_selected} cleared the {edge_threshold*100:.0f}% edge bar"
            + (f" (accuracy {fold_accuracy*100:.1f}%)" if n_selected > 0 else "")
        )

    fold_df = pd.DataFrame(fold_rows)
    print("\n--- Fold-by-fold ---")
    print(fold_df.to_string(index=False))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not all_picks:
        print(
            "\nNo picks cleared BOTH the agreement gate and the edge threshold in any fold -- "
            "this selection rule is too strict to produce any picks at all on this data. That's "
            "itself informative: it means the games where the models agree essentially never "
            "coincide with a large enough market disagreement, at least at this threshold."
        )
        with open(config.CONSENSUS_EDGE_METRICS_JSON, "w") as f:
            json.dump({
                "consensus_edge_threshold": edge_threshold,
                "test_seasons": test_seasons,
                "n_total_picks": 0,
            }, f, indent=2)
        print(f"\n  -> {config.CONSENSUS_EDGE_METRICS_JSON}")
        return

    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df["profit_if_win"] = american_odds_to_profit_if_win(picks_df["odds"])
    picks_df["profit"] = np.where(picks_df["correct"] == 1, picks_df["profit_if_win"], -1.0)
    picks_df.to_csv(config.CONSENSUS_EDGE_PICKS_CSV, index=False)

    n_total = len(picks_df)
    n_correct = int(picks_df["correct"].sum())
    accuracy = n_correct / n_total
    se = (accuracy * (1 - accuracy) / n_total) ** 0.5 if n_total > 0 else float("nan")
    z_score = (accuracy - 0.5) / se if se > 0 else float("nan")
    roi = picks_df["profit"].mean()
    roi_se = picks_df["profit"].std(ddof=1) / (n_total ** 0.5) if n_total > 1 else float("nan")

    total_test_games = int(fold_df["n_test"].sum())
    print(f"\n--- Pooled across all {len(fold_rows)} folds ---")
    print(f"Total picks (agreed AND edge >= {edge_threshold*100:.0f}%): "
          f"{n_total} out of {total_test_games} total test games "
          f"({n_total / total_test_games * 100:.1f}% of games produced a pick)")
    print(f"Accuracy: {accuracy*100:.1f}% (standard error +/-{se*100:.1f} points), z-score vs 50% = {z_score:.2f}")
    print(f"ROI: {roi*100:+.1f}% per unit staked (standard error +/-{roi_se*100:.1f} points)")

    if n_total < 30:
        print(
            f"\nn={n_total} is a small sample by any standard -- earlier scripts in this build "
            "showed accuracy at similarly small n swinging wildly by chance (e.g. single-digit-n "
            "cells hitting 100% or 0%). Don't treat this result as reliable regardless of which "
            "way it points; a rule this strict may simply not produce enough picks per season to "
            "be testable this way."
        )
    elif abs(z_score) < 1.96:
        print(
            "\nNot statistically distinguishable from a coin flip. Combined with every other "
            "evaluation run in this build (raw consensus, edge-sweep by model), this is more "
            "evidence pointing the same direction: no reliable edge detected."
        )
    else:
        print(
            f"\nClears the conventional significance bar ({abs(z_score):.2f} SEs from 50%) -- but "
            "given how many different selection rules have been tried on this same underlying "
            "feature set and data (consensus alone, edge alone by three model constructions, and "
            "now consensus+edge combined), treat this as one more data point subject to the "
            "multiple-comparisons problem, not a standalone confirmed result. A truly convincing "
            "case would need this exact rule re-tested on new data going forward, not just found "
            "significant after several other rules were tried and didn't clear the bar."
        )

    with open(config.CONSENSUS_EDGE_METRICS_JSON, "w") as f:
        json.dump({
            "consensus_edge_threshold": config.CONSENSUS_EDGE_THRESHOLD,
            "test_seasons": test_seasons,
            "n_total_picks": n_total,
            "n_total_test_games": total_test_games,
            "accuracy": accuracy,
            "accuracy_se": se,
            "z_score": z_score,
            "roi": roi,
            "roi_se": roi_se,
        }, f, indent=2)

    print(f"\n  -> {config.CONSENSUS_EDGE_PICKS_CSV}")
    print(f"  -> {config.CONSENSUS_EDGE_METRICS_JSON}")


if __name__ == "__main__":
    main()
