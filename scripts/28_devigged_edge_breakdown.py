"""
Re-runs 22_edge_only_breakdown.py's exact question -- "if you ignore
model agreement entirely and just bet whichever side has the higher edge,
on every game, does edge size predict accuracy?" -- using the CORRECTED,
de-vigged edge/probability definition instead of the raw (vig-included)
one every prior edge diagnostic (08/09/11/14/19/20/22) used.

Why this exists: orb-analytics-web's live "Win Probability" card was found
to blend the model against a RAW (vig-included) market-implied
probability, which breaks the away = 1 - home relationship the model's own
raw output guarantees by construction (see 26_write_predictions_json.py's
fix). That was a DISPLAY bug. This script asks the separate, real question
it raised: does using the CORRECTED (de-vigged) probability to decide
which side to pick change whether edge size actually predicts accuracy --
which every prior run (11's consensus-gated check, 22's no-gate check)
found no relationship for?

IMPORTANT -- this does NOT change any selection rule already in
production. 25_live_weekly_scoring.py's keep_favorite gate
(THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD) still uses the ORIGINAL raw-vig
edge from feature_utils.compute_edges(), untouched. This script is a
diagnostic, run alongside 22 (not a replacement for it), specifically to
answer: "was the null edge-accuracy finding in 11/22 an artifact of not
de-vigging, or is it real regardless of which market basis is used?"

Method, identical to 22 except for the edge/probability formula:
  1. Same expanding-window walk-forward folds (refit logit + XGBoost per
     fold, same curated feature set, same everything else as 22).
  2. combined_prob = average(logit_prob, xgb_prob) for home cover -- same
     as 22.
  3. For each game, compute home_edge/away_edge using DE-VIGGED
     home_implied/away_implied (feature_utils.compute_edges_devigged),
     pick whichever side has the higher edge -- every game gets a pick.
  4. Pool picks across all folds, bucket into deciles by (de-vigged) edge
     size, report accuracy/units/ROI per decile plus a Spearman
     correlation -- SAME output shape as 22, for direct side-by-side
     comparison. Also reports the picked side's final blended cover
     probability (picked_confidence) per decile, since that's the number
     actually shown on the live site as "Win Probability."

Run:
    python scripts/28_devigged_edge_breakdown.py

Requires data/processed/training_set.csv to exist (same prerequisite as
every other backtest script -- run 01_fetch_historical.py + 02_build_training_set.py
first if it's missing).
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
    compute_edges_devigged,
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


def decile_table(picks_df, n_buckets, label):
    n_buckets = min(n_buckets, picks_df["edge"].nunique())
    picks_df = picks_df.copy()
    picks_df["edge_decile"] = pd.qcut(picks_df["edge"], n_buckets, labels=False, duplicates="drop") + 1

    deciles = (
        picks_df.groupby("edge_decile")
        .agg(
            n=("correct", "size"),
            min_edge=("edge", "min"),
            max_edge=("edge", "max"),
            mean_edge=("edge", "mean"),
            mean_confidence=("confidence", "mean") if "confidence" in picks_df.columns else ("edge", "mean"),
            accuracy=("correct", "mean"),
            units=("profit", "sum"),
            roi=("profit", "mean"),
        )
        .reset_index()
    )
    deciles["accuracy_se"] = (deciles["accuracy"] * (1 - deciles["accuracy"]) / deciles["n"]) ** 0.5

    print(f"\n--- {label}: accuracy by edge decile (1 = lowest edge, {n_buckets} = highest edge) ---")
    fmt = {
        "min_edge": "{:+.1%}".format,
        "max_edge": "{:+.1%}".format,
        "mean_edge": "{:+.1%}".format,
        "mean_confidence": "{:.1%}".format,
        "accuracy": "{:.1%}".format,
        "accuracy_se": "±{:.1%}".format,
        "units": "{:+.2f}".format,
        "roi": "{:+.1%}".format,
    }
    print(deciles.to_string(index=False, formatters=fmt))

    corr, p_value = spearmanr(picks_df["edge"], picks_df["correct"])
    print(f"\nSpearman correlation (edge vs. correctness), {label}: {corr:+.3f} (p={p_value:.3f})")

    overall_accuracy = picks_df["correct"].mean()
    overall_n = len(picks_df)
    overall_se = (overall_accuracy * (1 - overall_accuracy) / overall_n) ** 0.5
    overall_z = (overall_accuracy - 0.5) / overall_se if overall_se > 0 else float("nan")
    overall_units = picks_df["profit"].sum()
    overall_roi = picks_df["profit"].mean()
    print(f"Overall, {label}: n={overall_n}, accuracy={overall_accuracy*100:.1f}% (z={overall_z:.2f} vs 50%), "
          f"units={overall_units:+.2f}, ROI={overall_roi*100:+.1f}%")

    return deciles, {
        "n_picks": overall_n,
        "overall_accuracy": overall_accuracy,
        "overall_z_score": overall_z,
        "overall_units": overall_units,
        "overall_roi": overall_roi,
        "spearman_corr": corr,
        "spearman_p_value": p_value,
        "deciles": deciles.to_dict(orient="records"),
    }


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

    raw_picks, devig_picks = [], []
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

        def build_pick_rows(edges_df, has_confidence):
            is_home_pick = edges_df["picked_side"].to_numpy() == "home"
            picked_odds = np.where(
                is_home_pick,
                test_df["home_spread_odds"].to_numpy(),
                test_df["away_spread_odds"].to_numpy(),
            )
            correct = np.where(
                is_home_pick,
                actual_home_cover.to_numpy() == 1,
                actual_home_cover.to_numpy() == 0,
            )
            # Favorite/underdog of the PICKED side, same convention as
            # 25_live_weekly_scoring.py's picked_favorite: spread_line is
            # nflverse's home-perspective spread (positive = home favored).
            spread_line = test_df["spread_line"].to_numpy()
            picked_favorite = np.where(is_home_pick, spread_line > 0, spread_line < 0)
            row = {
                "season": test_season,
                "week": test_df["week"].to_numpy(),
                "game_id": test_df["game_id"].to_numpy() if "game_id" in test_df.columns else np.arange(len(test_df)),
                "picked_side": edges_df["picked_side"].to_numpy(),
                "picked_favorite": picked_favorite,
                "edge": edges_df["picked_edge"].to_numpy(),
                "odds": picked_odds,
                "correct": correct.astype(int),
            }
            if has_confidence:
                row["confidence"] = edges_df["picked_confidence"].to_numpy()
            return pd.DataFrame(row)

        raw_edges = compute_edges(combined_pred, home_implied, away_implied)
        raw_picks.append(build_pick_rows(raw_edges, has_confidence=False))

        devig_edges = compute_edges_devigged(combined_pred, home_implied, away_implied)
        devig_picks.append(build_pick_rows(devig_edges, has_confidence=True))

        print(f"Fold {test_season}: {len(test_df)} games, every game picked (no consensus gate)")

    raw_df = pd.concat(raw_picks, ignore_index=True)
    raw_df["profit_if_win"] = american_odds_to_profit_if_win(raw_df["odds"])
    raw_df["profit"] = np.where(raw_df["correct"] == 1, raw_df["profit_if_win"], -1.0)

    devig_df = pd.concat(devig_picks, ignore_index=True)
    devig_df["profit_if_win"] = american_odds_to_profit_if_win(devig_df["odds"])
    devig_df["profit"] = np.where(devig_df["correct"] == 1, devig_df["profit_if_win"], -1.0)

    agree_pct = (raw_df["picked_side"].to_numpy() == devig_df["picked_side"].to_numpy()).mean()
    print(f"\n{len(raw_df)} total picks pooled across {len(test_seasons)} walk-forward folds.")
    print(f"Raw-vig and de-vigged edge picked the SAME side on {agree_pct*100:.1f}% of games "
          f"({int(round((1-agree_pct)*len(raw_df)))} games flipped sides).\n")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(config.PROCESSED_DIR / "devig_check_raw_picks.csv", index=False)
    devig_df.to_csv(config.PROCESSED_DIR / "devig_check_devigged_picks.csv", index=False)

    raw_deciles, raw_metrics = decile_table(raw_df, config.EDGE_ONLY_N_BUCKETS, "RAW (vig-included) edge -- matches 22's original method")
    devig_deciles, devig_metrics = decile_table(devig_df, config.EDGE_ONLY_N_BUCKETS, "DE-VIGGED edge (corrected)")

    raw_deciles.to_csv(config.PROCESSED_DIR / "devig_check_raw_deciles.csv", index=False)
    devig_deciles.to_csv(config.PROCESSED_DIR / "devig_check_devigged_deciles.csv", index=False)

    with open(config.PROCESSED_DIR / "devig_check_metrics.json", "w") as f:
        json.dump({
            "side_agreement_pct": agree_pct,
            "raw": raw_metrics,
            "devigged": devig_metrics,
        }, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    print(f"Raw edge:      n={raw_metrics['n_picks']}, accuracy={raw_metrics['overall_accuracy']*100:.1f}%, "
          f"z={raw_metrics['overall_z_score']:.2f}, ROI={raw_metrics['overall_roi']*100:+.1f}%, "
          f"Spearman={raw_metrics['spearman_corr']:+.3f} (p={raw_metrics['spearman_p_value']:.3f})")
    print(f"De-vigged edge: n={devig_metrics['n_picks']}, accuracy={devig_metrics['overall_accuracy']*100:.1f}%, "
          f"z={devig_metrics['overall_z_score']:.2f}, ROI={devig_metrics['overall_roi']*100:+.1f}%, "
          f"Spearman={devig_metrics['spearman_corr']:+.3f} (p={devig_metrics['spearman_p_value']:.3f})")

    if devig_metrics["spearman_p_value"] < 0.05 and raw_metrics["spearman_p_value"] >= 0.05:
        print(
            "\nDe-vigging changed the finding: edge now shows a significant relationship with "
            "accuracy that the raw (vig-included) version did NOT show. The earlier null result "
            "(11, 22) may have been partly an artifact of not de-vigging."
        )
    elif devig_metrics["spearman_p_value"] >= 0.05 and raw_metrics["spearman_p_value"] >= 0.05:
        print(
            "\nDe-vigging did NOT change the finding: edge still shows no significant relationship "
            "with accuracy either way. The null result in 11/22 was not an artifact of vig -- "
            "whatever drives real accuracy here, it isn't raw edge size, devigged or not."
        )
    else:
        print(
            "\nBoth raw and de-vigged edge show a significant relationship with accuracy -- compare "
            "the sign and magnitude above to see whether de-vigging strengthened, weakened, or "
            "flipped it."
        )

    print(f"\n  -> {config.PROCESSED_DIR / 'devig_check_raw_deciles.csv'}")
    print(f"  -> {config.PROCESSED_DIR / 'devig_check_devigged_deciles.csv'}")
    print(f"  -> {config.PROCESSED_DIR / 'devig_check_metrics.json'}")


if __name__ == "__main__":
    main()
