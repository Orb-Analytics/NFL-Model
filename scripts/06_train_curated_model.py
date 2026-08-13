"""
Small, football-intuition-driven feature set: a deliberate contrast to
05_train_baseline_model.py's algorithmic top-150-then-dedup approach.

What happened in 05 explains why this script exists: correlation-based
selection run against ~3,000 columns picked features that fit the training
seasons (train AUC 0.81) but didn't generalize (holdout AUC 0.51, right at
chance). Given 04_feature_analysis.py already found effect sizes across
those columns were close to indistinguishable from pure noise, a purely
statistical selection process was likely fishing in that noise rather than
finding real structure. This script tests the opposite philosophy: a short
list of features a football person would actually expect to matter --
EPA/play, completion %, turnover rates, trench performance (sack rates),
rest, and dome -- defined in config.CURATED_FEATURES.

Two models are fit on the SAME curated set, for comparison:
  - Logistic regression (statsmodels) -- appropriate now that the
    features-to-rows ratio is sane (~15 features, ~3,600 rows). Its
    coefficients and p-values are real, multivariate evidence about which
    of these specific features matter controlling for the others -- a much
    better-controlled answer than 04's one-at-a-time univariate pass.
  - XGBoost with much lighter regularization than 05 needed, since the
    small-n/large-p overfitting risk mostly doesn't apply at 15 features.

Both are evaluated against the same naive baselines on the same
chronological holdout used in 05, so this is a genuine apples-to-apples
comparison of "curated 15 features" vs. "algorithmic top-150" -- not just
a vibes-based judgment call.

Run:
    python scripts/06_train_curated_model.py
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
from feature_utils import evaluate


def main():
    df = pd.read_csv(config.TRAINING_SET_CSV, low_memory=False)
    df = df[df[config.TARGET_COLUMN].notna()].copy()
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    curated_cols = [c for c in config.CURATED_FEATURES if c in df.columns]
    missing = [c for c in config.CURATED_FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} curated features not found in training_set.csv: {missing}")
        print("Check these were actually produced by build_features.py / feature_engineering.py --")
        print("a typo here would silently shrink the feature set without you noticing.")
    print(f"Using {len(curated_cols)} curated features: {curated_cols}")

    before_dropna = len(df)
    df = df.dropna(subset=curated_cols + [config.TARGET_COLUMN])
    dropped = before_dropna - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in the curated set (expected: mostly the")
        print(f"very first games of the earliest season, where there's no prior game to roll from).")

    is_test = df["season"].isin(config.TEST_SEASONS)
    train_df, test_df = df[~is_test], df[is_test]
    print(f"\nTrain: {len(train_df)} games (seasons < {min(config.TEST_SEASONS)})")
    print(f"Test:  {len(test_df)} games (seasons {config.TEST_SEASONS})")
    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError(
            f"Empty train or test split -- check config.TEST_SEASONS ({config.TEST_SEASONS}) "
            f"against the seasons actually present: {sorted(df['season'].unique())}"
        )

    target_train = train_df[config.TARGET_COLUMN]
    target_test = test_df[config.TARGET_COLUMN]

    # standardize for the logistic regression -- not required for validity,
    # but keeps coefficients on a comparable scale for reading at a glance,
    # and helps the optimizer converge cleanly. Fit on TRAIN only, applied
    # to test, same no-leakage discipline as everywhere else in this pipeline.
    means = train_df[curated_cols].mean()
    stds = train_df[curated_cols].std().replace(0, 1)
    X_train_std = (train_df[curated_cols] - means) / stds
    X_test_std = (test_df[curated_cols] - means) / stds

    print("\n--- Logistic regression ---")
    X_train_const = sm.add_constant(X_train_std)
    X_test_const = sm.add_constant(X_test_std, has_constant="add")
    try:
        logit_model = sm.Logit(target_train.to_numpy(), X_train_const).fit(disp=0)
        print(logit_model.summary())
    except np.linalg.LinAlgError:
        # Singular Hessian -- usually near-perfect collinearity between two
        # curated columns, or (with a small training set) quasi-complete
        # separation. Falling back to a lightly-regularized fit trades away
        # p-values for a model that actually converges; if this triggers on
        # the real data, it's worth checking config.CURATED_FEATURES for a
        # pair that's more collinear than expected.
        print("Standard MLE fit hit a singular matrix (likely near-collinear features or, with a small")
        print("training set, quasi-complete separation). Falling back to a regularized fit -- p-values")
        print("won't be available from this path, only coefficients.")
        logit_model = sm.Logit(target_train.to_numpy(), X_train_const).fit_regularized(
            alpha=1.0, disp=0
        )
        print(logit_model.params)

    try:
        summary_text = str(logit_model.summary())
    except (NotImplementedError, AttributeError):
        summary_text = f"(regularized fit -- no full summary available)\n\ncoefficients:\n{logit_model.params}"
    with open(config.CURATED_LOGIT_SUMMARY_TXT, "w") as f:
        f.write(summary_text)

    logit_train_pred = logit_model.predict(X_train_const)
    logit_test_pred = logit_model.predict(X_test_const)

    print("\n--- XGBoost (lightly regularized -- see config.XGB_PARAMS_CURATED) ---")
    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS_CURATED, missing=np.nan)
    xgb_model.fit(train_df[curated_cols], target_train)
    xgb_train_pred = xgb_model.predict_proba(train_df[curated_cols])[:, 1]
    xgb_test_pred = xgb_model.predict_proba(test_df[curated_cols])[:, 1]

    xgb_importance = pd.DataFrame({
        "column": curated_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    xgb_importance.to_csv(config.CURATED_XGB_IMPORTANCE_CSV, index=False)

    train_base_rate = target_train.mean()
    majority_class = int(train_base_rate >= 0.5)
    baseline_majority_pred = np.full(len(target_test), majority_class, dtype=float)
    baseline_base_rate_pred = np.full(len(target_test), train_base_rate, dtype=float)

    results = [
        evaluate(target_train.to_numpy(), logit_train_pred, "logistic regression (train, in-sample)"),
        evaluate(target_test.to_numpy(), logit_test_pred, "logistic regression (test, holdout)"),
        evaluate(target_train.to_numpy(), xgb_train_pred, "xgboost (train, in-sample)"),
        evaluate(target_test.to_numpy(), xgb_test_pred, "xgboost (test, holdout)"),
        evaluate(target_test.to_numpy(), baseline_majority_pred, "baseline: always predict majority class"),
        evaluate(target_test.to_numpy(), baseline_base_rate_pred, "baseline: always predict train base rate"),
    ]
    results_df = pd.DataFrame(results)

    print(f"\nTrain base rate (home covers): {train_base_rate * 100:.1f}%")
    print("\n" + results_df.to_string(index=False))

    print("\nTop features by XGBoost importance:")
    print(xgb_importance.to_string(index=False))

    # --- Consensus picks: only count it as a "pick" when both models predict
    # the same side. Disagreements are skipped entirely, same as a real bettor
    # would only act when independent signals line up. Note this is a
    # different (smaller, self-selected) sample than the holdout as a whole --
    # see the standard error printed below before reading too much into it. ---
    logit_test_class = (logit_test_pred.to_numpy() >= 0.5).astype(int)
    xgb_test_class = (xgb_test_pred >= 0.5).astype(int)
    actual_test = target_test.to_numpy()
    agree_mask = logit_test_class == xgb_test_class

    n_total = len(actual_test)
    n_agree = int(agree_mask.sum())
    print(f"\n--- Consensus picks (logistic regression and XGBoost predict the same side) ---")
    print(f"Models agree on {n_agree}/{n_total} holdout games ({n_agree / n_total * 100:.1f}%)")

    if n_agree > 0:
        consensus_pred = logit_test_class[agree_mask]  # identical to xgb_test_class[agree_mask] by definition
        consensus_actual = actual_test[agree_mask]
        consensus_accuracy = (consensus_pred == consensus_actual).mean()
        consensus_se = (consensus_accuracy * (1 - consensus_accuracy) / n_agree) ** 0.5

        disagree_actual = actual_test[~agree_mask]
        n_disagree = n_total - n_agree

        print(f"Accuracy on consensus picks: {consensus_accuracy * 100:.1f}% (n={n_agree}, "
              f"standard error ±{consensus_se * 100:.1f} points)")
        print(f"(for reference) overall holdout base rate (home covers): {actual_test.mean() * 100:.1f}%, "
              f"n={n_total}")
        if n_disagree > 0:
            print(f"{n_disagree} games had no consensus pick and were excluded from the accuracy above.")

        z_score = (consensus_accuracy - 0.5) / consensus_se if consensus_se > 0 else float("nan")
        if abs(z_score) < 1.96:
            print(
                f"Consensus accuracy is within {abs(z_score):.1f} standard errors of 50% -- NOT "
                "statistically distinguishable from a coin flip at this sample size. Don't treat "
                "this as confirmed edge; a bigger sample (more seasons, via walk-forward) is needed "
                "before this could be trusted."
            )
        else:
            print(
                f"Consensus accuracy is {abs(z_score):.1f} standard errors from 50% -- clears the "
                "conventional significance bar, but still just ONE holdout's worth of consensus "
                "picks. Confirm this holds up across multiple seasons via walk-forward before "
                "trusting it."
            )

        consensus_df = test_df.loc[agree_mask, [c for c in ["game_id", "season", "week", "home_team", "away_team", "spread_line"] if c in test_df.columns]].copy()
        consensus_df["actual_home_cover"] = consensus_actual
        consensus_df["predicted_home_cover"] = consensus_pred
        consensus_df["logit_prob"] = logit_test_pred.to_numpy()[agree_mask]
        consensus_df["xgb_prob"] = xgb_test_pred[agree_mask]
        consensus_df["correct"] = consensus_pred == consensus_actual
        consensus_df.to_csv(config.CURATED_CONSENSUS_PICKS_CSV, index=False)
        print(f"  -> {config.CURATED_CONSENSUS_PICKS_CSV} (every consensus pick, game by game)")
    else:
        print("No consensus picks at all -- the two models never agreed on a single holdout game.")

    holdout_beats_baseline = (
        results[1]["log_loss"] < results[5]["log_loss"] or results[3]["log_loss"] < results[5]["log_loss"]
    )
    if holdout_beats_baseline:
        print(
            "\nAt least one curated model beats the base-rate baseline on holdout log loss. Worth taking\n"
            "seriously enough to move to proper multi-season walk-forward validation before trusting it --\n"
            "one 2-season holdout isn't enough to confirm this is real edge rather than a lucky split."
        )
    else:
        print(
            "\nNeither curated model clearly beats the base-rate baseline on this holdout. Combined with\n"
            "05's algorithmic result, this is a second, independent signal pointing the same direction:\n"
            "this feature set (even hand-picked, even football-intuitive) isn't showing a reliable edge\n"
            "against spread_line on 2024-2025. Worth treating as a real finding, not a setup problem."
        )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CURATED_MODEL_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  -> {config.CURATED_MODEL_METRICS_JSON}")
    print(f"  -> {config.CURATED_LOGIT_SUMMARY_TXT}")
    print(f"  -> {config.CURATED_XGB_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
