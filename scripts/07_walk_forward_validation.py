"""
Multi-season walk-forward validation of the curated feature set
(config.CURATED_FEATURES -- 16 features, no turnover-rate predictors; see the
comment block above CURATED_FEATURES in config.py for why turnovers were
deliberately left out rather than re-added to chase a p-value).

Why this script exists, and why it comes after -- not alongside -- feature
selection: 06_train_curated_model.py's single 2024-2025 holdout is only
~550 games, and results on it were fragile -- they flipped between
borderline-significant and clearly-not-significant depending on minor,
after-the-fact feature tweaks (adding/removing turnover features changed the
consensus-pick z-score from ~2.0 to ~1.1). That fragility is itself evidence
against a robust edge: a real signal shouldn't depend on which of two very
similar feature lists you happened to test. The fix is exactly what was
planned from the start -- fix the feature set BEFORE looking at more
results, then validate across many seasons instead of one.

Method: expanding-window walk-forward, one season at a time.
  fold 1: train on 2010-2017, test on 2018
  fold 2: train on 2010-2018, test on 2019
  ...
  fold N: train on 2010-2024, test on 2025
Each fold only ever uses data that would have actually existed at the time
-- the same no-leakage discipline as the chronological split elsewhere in
this pipeline, just repeated across every available season instead of one.

Both models (logistic regression + XGBoost, same as 06) are refit from
scratch in every fold. Per-fold metrics are recorded, and -- more
importantly -- every fold's predictions are pooled into one combined sample
before computing consensus-pick accuracy. Pooling is what actually buys
statistical power here: instead of one holdout's ~550 games, this is every
test-season game across however many folds exist, which is what turns
"maybe significant, maybe not" into an answer you can trust.

Run:
    python scripts/07_walk_forward_validation.py
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
from feature_utils import evaluate, american_odds_to_profit_if_win


def fit_logit(X_train_const, y_train):
    """Same singular-Hessian fallback as 06 -- a fold with an unlucky small
    training slice or near-collinear columns can hit this even if most
    folds don't."""
    try:
        model = sm.Logit(y_train, X_train_const).fit(disp=0)
        return model, True
    except np.linalg.LinAlgError:
        model = sm.Logit(y_train, X_train_const).fit_regularized(alpha=1.0, disp=0)
        return model, False


def run_fold(train_df, test_df, curated_cols):
    target_train = train_df[config.TARGET_COLUMN]
    target_test = test_df[config.TARGET_COLUMN]

    means = train_df[curated_cols].mean()
    stds = train_df[curated_cols].std().replace(0, 1)
    X_train_std = (train_df[curated_cols] - means) / stds
    X_test_std = (test_df[curated_cols] - means) / stds

    X_train_const = sm.add_constant(X_train_std)
    X_test_const = sm.add_constant(X_test_std, has_constant="add")
    logit_model, converged_cleanly = fit_logit(X_train_const, target_train.to_numpy())
    logit_test_pred = np.asarray(logit_model.predict(X_test_const))

    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS_CURATED, missing=np.nan)
    xgb_model.fit(train_df[curated_cols], target_train)
    xgb_test_pred = xgb_model.predict_proba(test_df[curated_cols])[:, 1]

    train_base_rate = target_train.mean()
    baseline_base_rate_pred = np.full(len(target_test), train_base_rate, dtype=float)

    logit_metrics = evaluate(target_test.to_numpy(), logit_test_pred, "logit")
    xgb_metrics = evaluate(target_test.to_numpy(), xgb_test_pred, "xgb")
    baseline_metrics = evaluate(target_test.to_numpy(), baseline_base_rate_pred, "baseline")

    logit_class = (logit_test_pred >= 0.5).astype(int)
    xgb_class = (xgb_test_pred >= 0.5).astype(int)
    agree_mask = logit_class == xgb_class

    fold_consensus = test_df.loc[
        agree_mask,
        [c for c in ["game_id", "season", "week", "home_team", "away_team", "spread_line"] if c in test_df.columns],
    ].copy()
    if len(fold_consensus) > 0:
        fold_consensus["actual_home_cover"] = target_test.to_numpy()[agree_mask]
        fold_consensus["predicted_home_cover"] = logit_class[agree_mask]
        fold_consensus["logit_prob"] = logit_test_pred[agree_mask]
        fold_consensus["xgb_prob"] = xgb_test_pred[agree_mask]
        fold_consensus["correct"] = logit_class[agree_mask] == target_test.to_numpy()[agree_mask]

        # Units, using each pick's ACTUAL market price -- added so this
        # script's picks file is directly comparable to 08/09's (and usable
        # by 10_season_backtest_report.py) even though 07's selection rule
        # itself never looks at price. Games missing odds get NaN profit
        # (excluded from unit sums, not treated as a loss) rather than
        # silently assuming -110 for every pick.
        if {"home_spread_odds", "away_spread_odds"}.issubset(test_df.columns):
            picked_odds = np.where(
                fold_consensus["predicted_home_cover"].to_numpy() == 1,
                test_df["home_spread_odds"].to_numpy()[agree_mask],
                test_df["away_spread_odds"].to_numpy()[agree_mask],
            )
            fold_consensus["odds"] = picked_odds
            fold_consensus["profit_if_win"] = american_odds_to_profit_if_win(fold_consensus["odds"]).to_numpy()
            fold_consensus["profit"] = np.where(
                fold_consensus["correct"], fold_consensus["profit_if_win"], -1.0
            )
            # NaN odds (no market price for that game) -> NaN profit, not -1
            fold_consensus.loc[fold_consensus["odds"].isna(), "profit"] = np.nan

    fold_summary = {
        "test_season": int(test_df["season"].iloc[0]),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "logit_converged_cleanly": converged_cleanly,
        "logit_accuracy": logit_metrics["accuracy"],
        "logit_auc": logit_metrics["auc"],
        "logit_log_loss": logit_metrics["log_loss"],
        "xgb_accuracy": xgb_metrics["accuracy"],
        "xgb_auc": xgb_metrics["auc"],
        "xgb_log_loss": xgb_metrics["log_loss"],
        "baseline_log_loss": baseline_metrics["log_loss"],
        "n_consensus": int(agree_mask.sum()),
        "consensus_accuracy": (
            float((logit_class[agree_mask] == target_test.to_numpy()[agree_mask]).mean())
            if agree_mask.sum() > 0 else float("nan")
        ),
    }
    return fold_summary, fold_consensus


def main():
    df = pd.read_csv(config.TRAINING_SET_CSV, low_memory=False)
    df = df[df[config.TARGET_COLUMN].notna()].copy()
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    curated_cols = [c for c in config.CURATED_FEATURES if c in df.columns]
    missing = [c for c in config.CURATED_FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} curated features not found in training_set.csv: {missing}")
    print(f"Using {len(curated_cols)} curated features (fixed across every fold): {curated_cols}\n")

    before_dropna = len(df)
    df = df.dropna(subset=curated_cols + [config.TARGET_COLUMN])
    dropped = before_dropna - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in the curated set.\n")

    all_seasons = sorted(int(s) for s in df["season"].unique())
    test_seasons = [s for s in all_seasons if s >= config.WALK_FORWARD_FIRST_TEST_SEASON]
    if not test_seasons:
        raise ValueError(
            f"No seasons >= WALK_FORWARD_FIRST_TEST_SEASON ({config.WALK_FORWARD_FIRST_TEST_SEASON}) "
            f"found. Seasons present: {all_seasons}"
        )
    print(f"Walk-forward folds: {test_seasons}\n")

    fold_summaries = []
    all_consensus = []
    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        summary, consensus = run_fold(train_df, test_df, curated_cols)
        fold_summaries.append(summary)
        if len(consensus) > 0:
            all_consensus.append(consensus)
        base_line = (
            f"Fold {test_season}: train n={summary['n_train']}, test n={summary['n_test']}, "
            f"logit AUC={summary['logit_auc']:.3f}, xgb AUC={summary['xgb_auc']:.3f}"
        )
        if summary["n_consensus"] > 0:
            print(f"{base_line}, consensus n={summary['n_consensus']}, "
                  f"consensus acc={summary['consensus_accuracy']*100:.1f}%")
        else:
            print(f"{base_line}, no consensus picks")

    fold_summary_df = pd.DataFrame(fold_summaries)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary_df.to_csv(config.WALK_FORWARD_FOLD_SUMMARY_CSV, index=False)

    print("\n--- Fold-by-fold summary ---")
    print(fold_summary_df.to_string(index=False))

    avg_logit_auc = fold_summary_df["logit_auc"].mean()
    avg_xgb_auc = fold_summary_df["xgb_auc"].mean()
    print(f"\nMean logit AUC across folds: {avg_logit_auc:.3f}")
    print(f"Mean xgb AUC across folds:   {avg_xgb_auc:.3f}")
    print("(0.50 = no better than chance; each fold is a genuinely unseen future season)")

    # --- Pooled consensus-pick accuracy across every fold. This is the whole
    # point of walk-forward: instead of one holdout's ~550 games, this pools
    # every test-season game across all folds into one combined sample,
    # which is what actually gives this a fair shot at distinguishing real
    # edge from noise. ---
    if all_consensus:
        combined = pd.concat(all_consensus, ignore_index=True)
        combined.to_csv(config.WALK_FORWARD_CONSENSUS_PICKS_CSV, index=False)

        n_total_consensus = len(combined)
        n_correct = int(combined["correct"].sum())
        pooled_accuracy = n_correct / n_total_consensus
        pooled_se = (pooled_accuracy * (1 - pooled_accuracy) / n_total_consensus) ** 0.5
        z_score = (pooled_accuracy - 0.5) / pooled_se if pooled_se > 0 else float("nan")

        print(f"\n--- Pooled consensus picks across all {len(fold_summaries)} folds ---")
        print(f"Total consensus picks: {n_total_consensus} (out of "
              f"{fold_summary_df['n_test'].sum()} total test games across all folds)")
        print(f"Pooled consensus accuracy: {pooled_accuracy*100:.1f}% "
              f"(standard error ±{pooled_se*100:.1f} points)")
        print(f"z-score vs. 50%: {z_score:.2f}")

        if "profit" in combined.columns and combined["profit"].notna().any():
            units_df = combined[combined["profit"].notna()]
            pooled_units = units_df["profit"].sum()
            pooled_roi = units_df["profit"].mean()
            print(f"Pooled units: {pooled_units:+.2f} across {len(units_df)} priced picks "
                  f"(ROI {pooled_roi*100:+.1f}% per unit staked)")
        else:
            pooled_units = None

        if abs(z_score) < 1.96:
            verdict = (
                "NOT statistically distinguishable from a coin flip, even pooled across "
                f"{len(fold_summaries)} seasons. This is a materially larger, harder-to-fool sample "
                "than any single holdout used so far in this build -- treat this as the most "
                "trustworthy read yet on whether this feature set has real edge against the "
                "closing spread, and right now the honest answer is no."
            )
        else:
            verdict = (
                f"Clears the conventional significance bar ({abs(z_score):.2f} standard errors from "
                f"50%) pooled across {len(fold_summaries)} independent out-of-sample seasons -- this "
                "is meaningfully stronger evidence than the single 2024-2025 holdout, since the "
                "feature set was fixed in advance and never adjusted based on these results. Still "
                "worth remembering: consensus picks are a self-selected subsample (only games where "
                "both models agreed), so this measures 'accuracy when the model is confident enough "
                "that both approaches agree', not 'accuracy on every game' -- check n_consensus vs. "
                "total games per fold above to see how often that condition is met in practice."
            )
        print("\n" + verdict)

        metrics_out = {
            "n_folds": len(fold_summaries),
            "test_seasons": test_seasons,
            "mean_logit_auc": avg_logit_auc,
            "mean_xgb_auc": avg_xgb_auc,
            "pooled_consensus_n": n_total_consensus,
            "pooled_consensus_accuracy": pooled_accuracy,
            "pooled_consensus_se": pooled_se,
            "pooled_consensus_z_score": z_score,
            "pooled_units": pooled_units,
        }
    else:
        print("\nNo consensus picks in any fold -- the two models never agreed on a single game "
              "across the entire walk-forward window.")
        metrics_out = {
            "n_folds": len(fold_summaries),
            "test_seasons": test_seasons,
            "mean_logit_auc": avg_logit_auc,
            "mean_xgb_auc": avg_xgb_auc,
            "pooled_consensus_n": 0,
        }

    with open(config.WALK_FORWARD_METRICS_JSON, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  -> {config.WALK_FORWARD_FOLD_SUMMARY_CSV}")
    print(f"  -> {config.WALK_FORWARD_CONSENSUS_PICKS_CSV}")
    print(f"  -> {config.WALK_FORWARD_METRICS_JSON}")


if __name__ == "__main__":
    main()
