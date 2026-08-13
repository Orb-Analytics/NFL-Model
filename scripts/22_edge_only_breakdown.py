"""
Edge-only pick selection, no consensus/agreement gate: for EVERY game (not
just games where logit and XGBoost happen to agree), pick whichever side --
home or away -- has the higher edge, then bucket ALL of those picks by edge
size to see whether edge alone tracks accuracy.

Every edge diagnostic run so far (11, 14) only checked edge's relationship
to accuracy WITHIN the consensus-gated subsample -- i.e. "among games the
two models already agreed on, does a bigger edge mean a better pick." This
is a genuinely different question: "if you ignore agreement entirely and
just bet whichever side the model likes best relative to the market, on
every single game, does edge size predict accuracy?" 08_edge_based_evaluation.py
already builds these edge-only picks but only reports cumulative
threshold sweeps (edge >= X); this reports the same underlying picks as
edge deciles with a Spearman correlation, matching 11's format, so it's
directly comparable to 11's consensus-gated result.

Method, per walk-forward fold (same expanding-window folds as everywhere
else in this build):
  1. Refit logit + XGBoost on the curated feature set, same as every other
     script here.
  2. combined_prob = average(logit_prob, xgb_prob) for home cover.
  3. For each game, compute home_edge and away_edge (EDGE_MODEL_WEIGHT-
     weighted blend of combined_prob against each side's market-implied
     probability, same formula as 08/09/11/14/19/20), and pick whichever
     side has the higher edge -- every game gets a pick, there's no
     "no consensus" case here.
  4. Pool picks across all folds, bucket into deciles by edge size, report
     accuracy/units/ROI per decile plus a Spearman correlation.

Run:
    python scripts/22_edge_only_breakdown.py
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
from scipy.stats import spearmanr

import config
from feature_utils import (
    american_odds_to_implied_prob,
    american_odds_to_profit_if_win,
    compute_edges,
)


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

    all_picks = []
    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        logit_pred, xgb_pred = run_fold(train_df, test_df, curated_cols)
        combined_pred = pd.Series((logit_pred + xgb_pred) / 2, index=test_df.index)

        home_implied = test_df["home_implied_prob"]
        away_implied = test_df["away_implied_prob"]
        actual_home_cover = test_df[config.TARGET_COLUMN]

        edges = compute_edges(combined_pred, home_implied, away_implied)
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
            "season": test_season,
            "week": test_df["week"].to_numpy(),
            "game_id": test_df["game_id"].to_numpy() if "game_id" in test_df.columns else np.arange(len(test_df)),
            "spread_line": test_df["spread_line"].to_numpy() if "spread_line" in test_df.columns else np.nan,
            "picked_side": edges["picked_side"].to_numpy(),
            "edge": edges["picked_edge"].to_numpy(),
            "odds": picked_odds,
            "correct": correct.astype(int),
        })
        all_picks.append(fold_picks)
        print(f"Fold {test_season}: {len(test_df)} games, every game picked (no consensus gate)")

    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df["profit_if_win"] = american_odds_to_profit_if_win(picks_df["odds"])
    picks_df["profit"] = np.where(picks_df["correct"] == 1, picks_df["profit_if_win"], -1.0)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(config.EDGE_ONLY_BREAKDOWN_PICKS_CSV, index=False)

    print(f"\n{len(picks_df)} total picks pooled across {len(test_seasons)} walk-forward folds "
          f"(one pick per game, every game included -- this is the FULL slate, not a filtered "
          f"subsample).")
    print(f"Edge range: {picks_df['edge'].min()*100:.1f}% to {picks_df['edge'].max()*100:.1f}%, "
          f"mean {picks_df['edge'].mean()*100:.1f}%\n")

    n_buckets = min(config.EDGE_ONLY_N_BUCKETS, picks_df["edge"].nunique())
    picks_df["edge_decile"] = pd.qcut(picks_df["edge"], n_buckets, labels=False, duplicates="drop") + 1

    deciles = (
        picks_df.groupby("edge_decile")
        .agg(
            n=("correct", "size"),
            min_edge=("edge", "min"),
            max_edge=("edge", "max"),
            mean_edge=("edge", "mean"),
            accuracy=("correct", "mean"),
            units=("profit", "sum"),
            roi=("profit", "mean"),
        )
        .reset_index()
    )
    deciles["accuracy_se"] = (deciles["accuracy"] * (1 - deciles["accuracy"]) / deciles["n"]) ** 0.5
    deciles.to_csv(config.EDGE_ONLY_BREAKDOWN_DECILE_CSV, index=False)

    print("--- Accuracy by edge decile, ALL games, no consensus gate (1 = lowest edge, "
          f"{n_buckets} = highest edge) ---")
    print(
        deciles.to_string(
            index=False,
            formatters={
                "min_edge": "{:+.1%}".format,
                "max_edge": "{:+.1%}".format,
                "mean_edge": "{:+.1%}".format,
                "accuracy": "{:.1%}".format,
                "accuracy_se": "±{:.1%}".format,
                "units": "{:+.2f}".format,
                "roi": "{:+.1%}".format,
            },
        )
    )

    corr, p_value = spearmanr(picks_df["edge"], picks_df["correct"])
    print(f"\nSpearman correlation between edge and correctness (all games, no gate): "
          f"{corr:+.3f} (p={p_value:.3f})")

    overall_accuracy = picks_df["correct"].mean()
    overall_n = len(picks_df)
    overall_se = (overall_accuracy * (1 - overall_accuracy) / overall_n) ** 0.5
    overall_z = (overall_accuracy - 0.5) / overall_se if overall_se > 0 else float("nan")
    overall_units = picks_df["profit"].sum()
    overall_roi = picks_df["profit"].mean()
    print(f"\nOverall (every game, no threshold): n={overall_n}, accuracy={overall_accuracy*100:.1f}% "
          f"(z={overall_z:.2f} vs 50%), units={overall_units:+.2f}, ROI={overall_roi*100:+.1f}%")
    print(
        "This is the honest 'bet the model's favorite side on literally every game' baseline -- "
        "compare it against 07's consensus accuracy (which only bets when models agree, a smaller, "
        "self-selected subsample) to see how much of 07's edge comes from the agreement filter "
        "itself versus the model's raw side-picking ability."
    )

    if p_value >= 0.05:
        verdict = (
            "Not statistically distinguishable from zero -- edge size doesn't predict accuracy any "
            "better here than it did within the consensus-gated subsample (11's result). Whatever is "
            "driving this model's real edge, it isn't 'bigger edge = more trustworthy pick' -- it's "
            "something else (agreement between models, per 07, or the favorite/underdog split, per "
            "13/14)."
        )
    elif corr > 0:
        verdict = (
            "Positive and significant -- bigger edge genuinely tracks higher accuracy across the "
            "full slate, even without an agreement gate. Worth checking whether this holds up when "
            "split by favorite/underdog too (14's split), since a pooled positive correlation can "
            "still hide an inverted relationship within one subgroup."
        )
    else:
        verdict = (
            "Negative and significant -- bigger edge is associated with WORSE accuracy across the "
            "full slate, the same backwards pattern as 14 found within favorite picks specifically. "
            "Treat any edge-based selection rule with real suspicion until this is understood."
        )
    print("\n" + verdict)

    with open(config.EDGE_ONLY_BREAKDOWN_METRICS_JSON, "w") as f:
        json.dump({
            "n_picks": overall_n,
            "overall_accuracy": overall_accuracy,
            "overall_z_score": overall_z,
            "overall_units": overall_units,
            "overall_roi": overall_roi,
            "spearman_corr": corr,
            "spearman_p_value": p_value,
            "deciles": deciles.to_dict(orient="records"),
        }, f, indent=2, default=str)

    print(f"\n  -> {config.EDGE_ONLY_BREAKDOWN_PICKS_CSV}")
    print(f"  -> {config.EDGE_ONLY_BREAKDOWN_DECILE_CSV}")
    print(f"  -> {config.EDGE_ONLY_BREAKDOWN_METRICS_JSON}")


if __name__ == "__main__":
    main()
