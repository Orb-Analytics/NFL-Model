"""
Baseline classifier: P(home team covers spread_line).

Given what 04_feature_analysis.py found -- essentially no individual
predictor beats what pure chance would produce across ~2,600 univariate
tests -- this script's job is to check whether a model that can combine
features (rather than looking at them one at a time) finds anything a
single-feature test structurally can't. It's also the first point where
"beating the spread" gets measured honestly: against a real chronological
holdout, compared to naive baselines, not just fit statistics on data the
model has already seen.

Four things this script does deliberately, each for a reason discussed
along the way:

  1. CHRONOLOGICAL split, not random. Train on seasons before TEST_SEASONS,
     test on TEST_SEASONS. A random split would let the model train on
     games from the same stretch of the season as ones it's tested on --
     leaking information a real weekly deployment would never have.

  2. RELEVANCE pre-filter BEFORE dedup (filter_by_relevance). This exists
     because of a concrete failure observed in practice: correlation-dedup
     alone only removes near-DUPLICATE columns, not irrelevant ones. Run
     against ~3,000 candidate columns, dedup left 1,800+ columns in play --
     still enough for XGBoost to hit 0.97 train AUC while scoring 0.48 AUC
     (worse than a coin flip) on the holdout. That's overfitting, not
     signal. Ranking by |correlation| with the target on TRAIN ONLY and
     keeping just the top MAX_FEATURES_BEFORE_DEDUP forces the feature
     count into a range the sample size can actually support.

  3. Correlation-based dedup, applied AFTER the relevance filter. diff_X,
     matchup_X, and the raw produced/allowed columns are structurally
     redundant by construction -- greedily dropping near-duplicates here
     means the model (and its feature importances) aren't diluted across
     3-4 copies of what's functionally the same signal.

  4. Deliberately conservative XGBoost hyperparameters. Even after both
     filters, there are still more features than you'd want relative to a
     few thousand training rows -- an unregularized model will happily
     memorize noise rather than learn anything that generalizes.

Evaluation is against TWO naive baselines, not just "what's the accuracy":
  - always predict the majority class (whichever of home-covers /
    away-covers was more common in the TRAINING data)
  - always predict the training data's base rate as a constant probability
A model that can't beat both of these under log loss/AUC on the holdout
isn't finding real signal, no matter how it looks in isolation. Given the
univariate results, don't be surprised if that's exactly what happens --
that's a legitimate finding about how hard this target is, not proof the
model or pipeline is broken.

Run:
    python scripts/05_train_baseline_model.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from feature_utils import get_predictor_columns, evaluate


def filter_by_relevance(df: pd.DataFrame, candidate_cols: list[str], target: pd.Series) -> list[str]:
    """Keep only the top config.MAX_FEATURES_BEFORE_DEDUP columns by |correlation|
    with the target. This runs BEFORE correlation_dedup and is the fix for a
    real failure mode: dedup alone only removes near-DUPLICATE columns, not
    irrelevant ones, so it can still leave thousands of columns in play
    against a few thousand training rows -- more than enough for a model to
    memorize the training seasons rather than learn anything that holds up
    out of sample. Computed on TRAIN data only, same reasoning as dedup:
    using the holdout to pick features would leak test information into
    feature selection even without touching test labels directly.
    """
    target_corr = df[candidate_cols].corrwith(target).abs().sort_values(ascending=False)
    kept = target_corr.dropna().head(config.MAX_FEATURES_BEFORE_DEDUP).index.tolist()
    print(f"Relevance pre-filter: {len(candidate_cols)} -> {len(kept)} columns (top {config.MAX_FEATURES_BEFORE_DEDUP} by |correlation| with target)")
    return kept


def correlation_dedup(df: pd.DataFrame, candidate_cols: list[str], target: pd.Series) -> list[str]:
    """Greedy correlation-based feature selection: rank columns by |correlation|
    with the target, then keep a column only if it isn't too correlated with
    something already kept. This is what actually thins out diff_X/matchup_X/
    produced-allowed duplication before training, rather than leaving the
    model to sort out near-identical columns on its own.
    """
    target_corr = df[candidate_cols].corrwith(target).abs().sort_values(ascending=False)
    ordered_cols = target_corr.dropna().index.tolist()

    print(f"Computing full pairwise correlation matrix across {len(ordered_cols)} columns for dedup...")
    corr_matrix = df[ordered_cols].corr().abs()

    kept: list[str] = []
    for col in ordered_cols:
        if not kept:
            kept.append(col)
            continue
        max_corr_with_kept = corr_matrix.loc[col, kept].max()
        if max_corr_with_kept < config.CORRELATION_DEDUP_THRESHOLD:
            kept.append(col)

    print(f"Correlation dedup: {len(ordered_cols)} -> {len(kept)} columns (threshold {config.CORRELATION_DEDUP_THRESHOLD})")
    return kept


def main():
    df = pd.read_csv(config.TRAINING_SET_PRUNED_CSV, low_memory=False)
    df = df[df[config.TARGET_COLUMN].notna()].copy()
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    is_test = df["season"].isin(config.TEST_SEASONS)
    train_df, test_df = df[~is_test], df[is_test]
    print(f"Train: {len(train_df)} games (seasons < {min(config.TEST_SEASONS)})")
    print(f"Test:  {len(test_df)} games (seasons {config.TEST_SEASONS})")
    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError(
            f"Empty train or test split -- check config.TEST_SEASONS ({config.TEST_SEASONS}) "
            f"against the seasons actually present: {sorted(df['season'].unique())}"
        )

    target_train = train_df[config.TARGET_COLUMN]
    target_test = test_df[config.TARGET_COLUMN]

    candidate_cols = get_predictor_columns(df)
    print(f"{len(candidate_cols)} candidate predictor columns before filtering")

    # both steps computed on TRAIN ONLY -- using test data to decide which
    # features to keep would leak information about the holdout into
    # feature selection, even though no labels are directly used.
    relevant_cols = filter_by_relevance(train_df, candidate_cols, target_train)
    kept_cols = correlation_dedup(train_df, relevant_cols, target_train)
    pd.DataFrame({"column": kept_cols}).to_csv(config.MODEL_KEPT_FEATURES_CSV, index=False)

    X_train, X_test = train_df[kept_cols], test_df[kept_cols]

    model = xgb.XGBClassifier(**config.XGB_PARAMS, missing=np.nan)
    model.fit(X_train, target_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    # naive baselines, computed from TRAIN data only (what you'd actually
    # know before seeing the test season)
    train_base_rate = target_train.mean()
    majority_class = int(train_base_rate >= 0.5)
    baseline_majority_pred = np.full(len(target_test), majority_class, dtype=float)
    baseline_base_rate_pred = np.full(len(target_test), train_base_rate, dtype=float)

    results = [
        evaluate(target_train.to_numpy(), train_pred, "model (train, in-sample)"),
        evaluate(target_test.to_numpy(), test_pred, "model (test, holdout)"),
        evaluate(target_test.to_numpy(), baseline_majority_pred, "baseline: always predict majority class"),
        evaluate(target_test.to_numpy(), baseline_base_rate_pred, "baseline: always predict train base rate"),
    ]
    results_df = pd.DataFrame(results)

    print(f"\nTrain base rate (home covers): {train_base_rate * 100:.1f}%")
    print("\n" + results_df.to_string(index=False))

    train_test_gap = results[0]["log_loss"] - results[1]["log_loss"]
    print(f"\nTrain vs. holdout log loss gap: {results[0]['log_loss']:.4f} vs {results[1]['log_loss']:.4f}")
    if results[1]["log_loss"] > results[3]["log_loss"]:
        print(
            "NOTE: the model's holdout log loss is WORSE than just predicting the training base rate.\n"
            "Given how weak the univariate signal looked in 04_feature_analysis.py, this is a plausible,\n"
            "honest outcome -- it would mean this feature set doesn't currently beat a naive baseline\n"
            "against the spread, not that something is broken."
        )
    elif results[1]["auc"] < 0.55:
        print(
            "NOTE: holdout AUC is only marginally above 0.5 (coin flip). Treat any edge here cautiously --\n"
            "confirm it holds up with a second holdout season or walk-forward validation before trusting it."
        )

    importance = pd.DataFrame({
        "column": kept_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(config.MODEL_FEATURE_IMPORTANCE_CSV, index=False)

    print("\nTop 20 features by importance:")
    print(importance.head(20).to_string(index=False))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(config.MODEL_PATH)
    with open(config.MODEL_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  -> {config.MODEL_PATH}")
    print(f"  -> {config.MODEL_METRICS_JSON}")
    print(f"  -> {config.MODEL_FEATURE_IMPORTANCE_CSV}")
    print(f"  -> {config.MODEL_KEPT_FEATURES_CSV}")


if __name__ == "__main__":
    main()
