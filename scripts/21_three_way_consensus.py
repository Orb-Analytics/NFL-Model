"""
Three-way consensus: add a third model -- Gaussian Naive Bayes, trained on
the SAME curated feature set as logit/XGBoost -- and require all three to
agree, instead of two.

Prompted by: "could a third model help smooth all these issues?" (volume,
class imbalance, needing positive edge everywhere). Every edge-threshold
lever tried so far (08/09/19/20) has shown null or backwards relationships
to accuracy -- the one mechanism that HAS reliably worked all build is
agreement between models with different inductive biases (07's 2-way
consensus: z=3.25 pooled, the strongest result in this project). Tightening
that same mechanism from 2-way to 3-way is a more conservative volume lever
than another hand-picked edge cutoff.

Why the same predictors, a different algorithm (not different predictors):
05_train_baseline_model.py already showed what happens with a broader,
algorithmically-selected feature set at this sample size -- severe
overfitting (train AUC 0.97, holdout AUC 0.48). The curated 12-feature set
is the one with real, walk-forward-validated edge; swapping in new columns
for a third model reopens exactly that failure mode. Naive Bayes on the
SAME features instead adds algorithmic diversity: logit is linear/additive,
XGBoost is tree-based/nonlinear with learned interactions, and Naive Bayes
assumes features are conditionally independent given the outcome -- a
meaningfully different bias from both, so its errors shouldn't be
correlated with theirs in the same way logit and XGBoost's are (both
fit to the same 12 columns, just with different functional forms).

Reports 2-way (logit+xgb, same as 07) and 3-way (all three agree)
consensus side by side, pooled across every walk-forward fold, so the
volume/accuracy tradeoff of tightening to 3-way is visible directly rather
than assumed.

Run:
    python scripts/21_three_way_consensus.py
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
from sklearn.naive_bayes import GaussianNB

import config
from feature_utils import evaluate, american_odds_to_profit_if_win


def fit_logit(X_train_const, y_train):
    """Same singular-Hessian fallback as 07/06."""
    try:
        model = sm.Logit(y_train, X_train_const).fit(disp=0)
        return model, True
    except np.linalg.LinAlgError:
        model = sm.Logit(y_train, X_train_const).fit_regularized(alpha=1.0, disp=0)
        return model, False


def build_consensus_df(test_df, target_test, agree_mask, logit_pred, xgb_pred, nb_pred, logit_class):
    cols = [c for c in ["game_id", "season", "week", "home_team", "away_team", "spread_line"] if c in test_df.columns]
    out = test_df.loc[agree_mask, cols].copy()
    if len(out) == 0:
        return out
    out["actual_home_cover"] = target_test.to_numpy()[agree_mask]
    out["predicted_home_cover"] = logit_class[agree_mask]
    out["logit_prob"] = logit_pred[agree_mask]
    out["xgb_prob"] = xgb_pred[agree_mask]
    out["nb_prob"] = nb_pred[agree_mask]
    out["correct"] = logit_class[agree_mask] == target_test.to_numpy()[agree_mask]

    if {"home_spread_odds", "away_spread_odds"}.issubset(test_df.columns):
        picked_odds = np.where(
            out["predicted_home_cover"].to_numpy() == 1,
            test_df["home_spread_odds"].to_numpy()[agree_mask],
            test_df["away_spread_odds"].to_numpy()[agree_mask],
        )
        out["odds"] = picked_odds
        out["profit_if_win"] = american_odds_to_profit_if_win(out["odds"]).to_numpy()
        out["profit"] = np.where(out["correct"], out["profit_if_win"], -1.0)
        out.loc[out["odds"].isna(), "profit"] = np.nan
    return out


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

    # Naive Bayes on the SAME standardized features logit uses (standardizing
    # doesn't change NB's predictions in principle since it fits its own
    # per-class mean/variance either way, but keeps preprocessing consistent
    # and avoids any numerical-scale surprises).
    nb_model = GaussianNB()
    nb_model.fit(X_train_std, target_train)
    nb_test_pred = nb_model.predict_proba(X_test_std)[:, 1]

    train_base_rate = target_train.mean()
    baseline_base_rate_pred = np.full(len(target_test), train_base_rate, dtype=float)

    logit_metrics = evaluate(target_test.to_numpy(), logit_test_pred, "logit")
    xgb_metrics = evaluate(target_test.to_numpy(), xgb_test_pred, "xgb")
    nb_metrics = evaluate(target_test.to_numpy(), nb_test_pred, "nb")
    baseline_metrics = evaluate(target_test.to_numpy(), baseline_base_rate_pred, "baseline")

    logit_class = (logit_test_pred >= 0.5).astype(int)
    xgb_class = (xgb_test_pred >= 0.5).astype(int)
    nb_class = (nb_test_pred >= 0.5).astype(int)

    agree_2way = logit_class == xgb_class
    agree_3way = agree_2way & (logit_class == nb_class)

    fold_consensus_2way = build_consensus_df(
        test_df, target_test, agree_2way, logit_test_pred, xgb_test_pred, nb_test_pred, logit_class
    )
    fold_consensus_3way = build_consensus_df(
        test_df, target_test, agree_3way, logit_test_pred, xgb_test_pred, nb_test_pred, logit_class
    )

    fold_summary = {
        "test_season": int(test_df["season"].iloc[0]),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "logit_converged_cleanly": converged_cleanly,
        "logit_accuracy": logit_metrics["accuracy"],
        "logit_auc": logit_metrics["auc"],
        "xgb_accuracy": xgb_metrics["accuracy"],
        "xgb_auc": xgb_metrics["auc"],
        "nb_accuracy": nb_metrics["accuracy"],
        "nb_auc": nb_metrics["auc"],
        "baseline_log_loss": baseline_metrics["log_loss"],
        "n_consensus_2way": int(agree_2way.sum()),
        "consensus_2way_accuracy": (
            float((logit_class[agree_2way] == target_test.to_numpy()[agree_2way]).mean())
            if agree_2way.sum() > 0 else float("nan")
        ),
        "n_consensus_3way": int(agree_3way.sum()),
        "consensus_3way_accuracy": (
            float((logit_class[agree_3way] == target_test.to_numpy()[agree_3way]).mean())
            if agree_3way.sum() > 0 else float("nan")
        ),
    }
    return fold_summary, fold_consensus_2way, fold_consensus_3way


def pool_and_report(all_dfs, label, metrics_json_path=None, picks_csv_path=None):
    if not all_dfs:
        print(f"\nNo {label} consensus picks in any fold.")
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    if picks_csv_path is not None:
        combined.to_csv(picks_csv_path, index=False)

    n = len(combined)
    n_correct = int(combined["correct"].sum())
    accuracy = n_correct / n
    se = (accuracy * (1 - accuracy) / n) ** 0.5
    z = (accuracy - 0.5) / se if se > 0 else float("nan")

    print(f"\n--- Pooled {label} consensus picks ---")
    print(f"Total picks: {n}")
    print(f"Accuracy: {accuracy*100:.1f}% (SE ±{se*100:.1f}pt), z-score vs 50% = {z:.2f}")

    units = roi = None
    if "profit" in combined.columns and combined["profit"].notna().any():
        priced = combined[combined["profit"].notna()]
        units = priced["profit"].sum()
        roi = priced["profit"].mean()
        print(f"Units: {units:+.2f} across {len(priced)} priced picks (ROI {roi*100:+.1f}%)")

    result = {"n": n, "accuracy": accuracy, "se": se, "z_score": z, "units": units, "roi": roi}
    return result


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
    all_2way = []
    all_3way = []
    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        summary, consensus_2way, consensus_3way = run_fold(train_df, test_df, curated_cols)
        fold_summaries.append(summary)
        if len(consensus_2way) > 0:
            all_2way.append(consensus_2way)
        if len(consensus_3way) > 0:
            all_3way.append(consensus_3way)
        line = (
            f"Fold {test_season}: train n={summary['n_train']}, test n={summary['n_test']}, "
            f"logit AUC={summary['logit_auc']:.3f}, xgb AUC={summary['xgb_auc']:.3f}, "
            f"nb AUC={summary['nb_auc']:.3f} | "
            f"2-way n={summary['n_consensus_2way']} acc={summary['consensus_2way_accuracy']*100:.1f}% | "
        )
        if summary["n_consensus_3way"] > 0:
            line += f"3-way n={summary['n_consensus_3way']} acc={summary['consensus_3way_accuracy']*100:.1f}%"
        else:
            line += "3-way: no picks"
        print(line)

    fold_summary_df = pd.DataFrame(fold_summaries)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary_df.to_csv(config.THREE_WAY_FOLD_SUMMARY_CSV, index=False)

    print("\n--- Fold-by-fold summary ---")
    print(fold_summary_df.to_string(index=False))

    print(f"\nMean logit AUC: {fold_summary_df['logit_auc'].mean():.3f}  "
          f"Mean xgb AUC: {fold_summary_df['xgb_auc'].mean():.3f}  "
          f"Mean nb AUC: {fold_summary_df['nb_auc'].mean():.3f}")

    result_2way = pool_and_report(all_2way, "2-way (logit+xgb)")
    result_3way = pool_and_report(
        all_3way, "3-way (logit+xgb+nb)",
        picks_csv_path=config.THREE_WAY_CONSENSUS_PICKS_CSV,
    )

    if result_2way and result_3way:
        print(
            f"\n--- 2-way vs. 3-way ---\n"
            f"2-way: n={result_2way['n']}, accuracy={result_2way['accuracy']*100:.1f}%, "
            f"z={result_2way['z_score']:.2f}"
            + (f", ROI={result_2way['roi']*100:+.1f}%" if result_2way['roi'] is not None else "") + "\n"
            f"3-way: n={result_3way['n']}, accuracy={result_3way['accuracy']*100:.1f}%, "
            f"z={result_3way['z_score']:.2f}"
            + (f", ROI={result_3way['roi']*100:+.1f}%" if result_3way['roi'] is not None else "")
        )
        volume_cut_pct = (1 - result_3way['n'] / result_2way['n']) * 100
        print(
            f"\nRequiring Naive Bayes agreement too cuts volume by {volume_cut_pct:.1f}% "
            f"({result_2way['n']} -> {result_3way['n']} picks). Compare the accuracy/z/ROI change "
            f"above against what 19/20's edge floors gave up for similar volume cuts -- if 3-way "
            f"consensus holds accuracy/z better per pick removed than an edge floor did, that's "
            f"evidence agreement-based filtering is the more informative lever of the two, as "
            f"expected given edge has shown null/backwards relationships everywhere else this build "
            f"has checked. Also check whether 3-way's favorite/underdog split "
            f"(via 13_pick_type_breakdown.py on {config.THREE_WAY_CONSENSUS_PICKS_CSV}) moved at "
            f"all, since unlike 19/20 this wasn't targeted at the imbalance issue specifically."
        )

    with open(config.THREE_WAY_METRICS_JSON, "w") as f:
        json.dump({
            "n_folds": len(fold_summaries),
            "test_seasons": test_seasons,
            "two_way": result_2way,
            "three_way": result_3way,
        }, f, indent=2, default=str)

    print(f"\n  -> {config.THREE_WAY_FOLD_SUMMARY_CSV}")
    print(f"  -> {config.THREE_WAY_CONSENSUS_PICKS_CSV}")
    print(f"  -> {config.THREE_WAY_METRICS_JSON}")
    print(f"\nFor the week-by-week / season view:")
    print(f"  python scripts/10_season_backtest_report.py --input {config.THREE_WAY_CONSENSUS_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
