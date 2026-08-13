"""
Three-way consensus + edge-direction alignment: keep a pick only if BOTH
(a) logit, XGBoost, and Naive Bayes all agree on the same side via the raw
>=0.5 threshold (21's 3-way consensus), AND (b) that same side also has the
higher market-adjusted edge when checked against the actual home/away
prices (22's edge-only side selection, using the 3-model average
probability instead of 22's 2-model average).

Prompted directly by: "is there any way to combine these? all three have to
agree and the average edge has to agree?"

Why these two conditions aren't redundant: raw threshold agreement (all
three probabilities >= 0.5 for the same side) only checks what the models
themselves think. The edge-implied side (comparing home_edge vs away_edge,
same compute_edges() helper used in 08/22) ALSO depends on how asymmetric
the market's home/away prices are for that specific game -- home_edge and
away_edge aren't just mirror images of each other unless home_spread_odds
and away_spread_odds happen to be symmetric (e.g. both -110). This is the
exact distinction that caused a real, previously-fixed bug in
09_consensus_edge_walk_forward.py (see its docstring): the side a raw
threshold prefers and the side that maximizes market-adjusted edge can
differ. Requiring both to agree is therefore a genuinely stricter, more
specific claim than either filter alone -- not just "all three models like
this side" but "all three models like this side, AND that preference
survives being checked against the actual asymmetric price you'd bet at."

Reports three ways side by side, pooled across every walk-forward fold:
  - 3-way raw consensus alone (21's rule)
  - edge-only side selection alone, on the SAME 3-way-agreed games only
    (so this is an apples-to-apples subset, not 22's full ungated slate)
  - 3-way consensus AND edge-direction alignment (this script's rule)

Run:
    python scripts/23_three_way_edge_aligned.py
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
from feature_utils import (
    american_odds_to_implied_prob,
    american_odds_to_profit_if_win,
    compute_edges,
)


def fit_logit(X_train_const, y_train):
    try:
        model = sm.Logit(y_train, X_train_const).fit(disp=0)
        return model, True
    except np.linalg.LinAlgError:
        return sm.Logit(y_train, X_train_const).fit_regularized(alpha=1.0, disp=0), False


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

    nb_model = GaussianNB()
    nb_model.fit(X_train_std, target_train)
    nb_test_pred = nb_model.predict_proba(X_test_std)[:, 1]

    logit_class = (logit_test_pred >= 0.5).astype(int)
    xgb_class = (xgb_test_pred >= 0.5).astype(int)
    nb_class = (nb_test_pred >= 0.5).astype(int)
    agree_3way = (logit_class == xgb_class) & (logit_class == nb_class)

    avg_prob_3 = pd.Series((logit_test_pred + xgb_test_pred + nb_test_pred) / 3, index=test_df.index)
    home_implied = test_df["home_implied_prob"]
    away_implied = test_df["away_implied_prob"]
    edges = compute_edges(avg_prob_3, home_implied, away_implied)
    edge_side_home = (edges["picked_side"].to_numpy() == "home").astype(int)
    edge_matches_consensus = edge_side_home == logit_class  # logit_class stands in for "the agreed side" where agree_3way is True

    both_align = agree_3way & edge_matches_consensus

    cols = [c for c in ["game_id", "season", "week", "home_team", "away_team", "spread_line"] if c in test_df.columns]

    def build(mask):
        out = test_df.loc[mask, cols].copy()
        if len(out) == 0:
            return out
        out["actual_home_cover"] = target_test.to_numpy()[mask]
        out["predicted_home_cover"] = logit_class[mask]
        out["logit_prob"] = logit_test_pred[mask]
        out["xgb_prob"] = xgb_test_pred[mask]
        out["nb_prob"] = nb_test_pred[mask]
        out["avg_prob_3"] = avg_prob_3.to_numpy()[mask]
        out["edge"] = edges["picked_edge"].to_numpy()[mask]
        out["correct"] = logit_class[mask] == target_test.to_numpy()[mask]
        picked_odds = np.where(
            out["predicted_home_cover"].to_numpy() == 1,
            test_df["home_spread_odds"].to_numpy()[mask],
            test_df["away_spread_odds"].to_numpy()[mask],
        )
        out["odds"] = picked_odds
        out["profit_if_win"] = american_odds_to_profit_if_win(out["odds"]).to_numpy()
        out["profit"] = np.where(out["correct"], out["profit_if_win"], -1.0)
        out.loc[out["odds"].isna(), "profit"] = np.nan
        return out

    fold_3way = build(agree_3way)
    fold_both = build(both_align)

    fold_summary = {
        "test_season": int(test_df["season"].iloc[0]),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_3way": int(agree_3way.sum()),
        "n_3way_edge_aligned": int(both_align.sum()),
    }
    return fold_summary, fold_3way, fold_both


def pool_and_report(all_dfs, label, picks_csv_path=None):
    if not all_dfs:
        print(f"\nNo {label} picks in any fold.")
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    if picks_csv_path is not None:
        combined.to_csv(picks_csv_path, index=False)

    n = len(combined)
    n_correct = int(combined["correct"].sum())
    accuracy = n_correct / n
    se = (accuracy * (1 - accuracy) / n) ** 0.5
    z = (accuracy - 0.5) / se if se > 0 else float("nan")

    print(f"\n--- Pooled {label} ---")
    print(f"Total picks: {n}")
    print(f"Accuracy: {accuracy*100:.1f}% (SE ±{se*100:.1f}pt), z-score vs 50% = {z:.2f}")

    units = roi = None
    if "profit" in combined.columns and combined["profit"].notna().any():
        priced = combined[combined["profit"].notna()]
        units = priced["profit"].sum()
        roi = priced["profit"].mean()
        print(f"Units: {units:+.2f} across {len(priced)} priced picks (ROI {roi*100:+.1f}%)")

    return {"n": n, "accuracy": accuracy, "se": se, "z_score": z, "units": units, "roi": roi}


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
        print(f"Dropped {dropped} rows with missing values in the curated set.\n")

    odds_cols = ["home_spread_odds", "away_spread_odds"]
    missing_odds_cols = [c for c in odds_cols if c not in df.columns]
    if missing_odds_cols:
        raise ValueError(f"Missing column(s) {missing_odds_cols} in training_set.csv.")

    has_odds = df["home_spread_odds"].notna() & df["away_spread_odds"].notna()
    df = df[has_odds].copy()
    print(f"{len(df)} games have both home_spread_odds and away_spread_odds -- "
          f"this analysis is scoped to those games only.\n")

    df["home_implied_prob"] = american_odds_to_implied_prob(df["home_spread_odds"])
    df["away_implied_prob"] = american_odds_to_implied_prob(df["away_spread_odds"])

    all_seasons = sorted(int(s) for s in df["season"].unique())
    test_seasons = [s for s in all_seasons if s >= config.WALK_FORWARD_FIRST_TEST_SEASON]
    if not test_seasons:
        raise ValueError(
            f"No seasons >= WALK_FORWARD_FIRST_TEST_SEASON ({config.WALK_FORWARD_FIRST_TEST_SEASON}) "
            f"with market odds. Seasons present: {all_seasons}"
        )
    print(f"Walk-forward folds (odds-covered seasons only): {test_seasons}\n")

    fold_summaries = []
    all_3way = []
    all_both = []
    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        summary, fold_3way, fold_both = run_fold(train_df, test_df, curated_cols)
        fold_summaries.append(summary)
        if len(fold_3way) > 0:
            all_3way.append(fold_3way)
        if len(fold_both) > 0:
            all_both.append(fold_both)
        print(f"Fold {test_season}: test n={summary['n_test']}, "
              f"3-way n={summary['n_3way']}, 3-way+edge-aligned n={summary['n_3way_edge_aligned']}")

    fold_summary_df = pd.DataFrame(fold_summaries)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary_df.to_csv(config.THREE_WAY_EDGE_ALIGNED_FOLD_SUMMARY_CSV, index=False)

    print("\n--- Fold-by-fold summary ---")
    print(fold_summary_df.to_string(index=False))

    result_3way = pool_and_report(all_3way, "3-way consensus alone (21's rule)")
    result_both = pool_and_report(
        all_both, "3-way consensus + edge-direction alignment",
        picks_csv_path=config.THREE_WAY_EDGE_ALIGNED_PICKS_CSV,
    )

    if result_3way and result_both:
        volume_cut_pct = (1 - result_both['n'] / result_3way['n']) * 100
        print(
            f"\n--- 3-way alone vs. 3-way + edge-aligned ---\n"
            f"3-way alone:        n={result_3way['n']}, accuracy={result_3way['accuracy']*100:.1f}%, "
            f"z={result_3way['z_score']:.2f}"
            + (f", ROI={result_3way['roi']*100:+.1f}%" if result_3way['roi'] is not None else "") + "\n"
            f"3-way + edge-aligned: n={result_both['n']}, accuracy={result_both['accuracy']*100:.1f}%, "
            f"z={result_both['z_score']:.2f}"
            + (f", ROI={result_both['roi']*100:+.1f}%" if result_both['roi'] is not None else "")
        )
        print(
            f"\nRequiring edge-direction alignment on top of 3-way agreement removes "
            f"{volume_cut_pct:.1f}% of the 3-way picks -- these are cases where all three models "
            f"raced past the 0.5 threshold on one side, but the actual asymmetric home/away market "
            f"price meant the market-adjusted edge favored the OTHER side. If accuracy/z/ROI improve "
            f"here, that's evidence this asymmetric-price case is a real, previously-invisible source "
            f"of weaker picks inside the 3-way consensus set. If they're roughly flat, this "
            f"discrepancy is rare enough in your data not to matter much -- worth checking n_3way vs. "
            f"n_3way_edge_aligned per fold above to see how often it actually triggers."
        )

    with open(config.THREE_WAY_EDGE_ALIGNED_METRICS_JSON, "w") as f:
        json.dump({
            "n_folds": len(fold_summaries),
            "test_seasons": test_seasons,
            "three_way_alone": result_3way,
            "three_way_edge_aligned": result_both,
        }, f, indent=2, default=str)

    print(f"\n  -> {config.THREE_WAY_EDGE_ALIGNED_FOLD_SUMMARY_CSV}")
    print(f"  -> {config.THREE_WAY_EDGE_ALIGNED_PICKS_CSV}")
    print(f"  -> {config.THREE_WAY_EDGE_ALIGNED_METRICS_JSON}")
    print(f"\nFor the week-by-week / season view:")
    print(f"  python scripts/10_season_backtest_report.py --input {config.THREE_WAY_EDGE_ALIGNED_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
