"""
Combined favorite/underdog + edge rule: bet every underdog consensus pick,
and only the LOW-edge favorite consensus picks -- built directly from two
real findings earlier in this build (13 and 14), not a new fishing
expedition.

  13 (pick-type breakdown): favorite picks are break-even (51.3% accuracy,
     -2.29 units); underdog picks carry virtually all the profit (56.0%,
     +71.56 units).
  14 (pick-type x edge breakdown): WITHIN favorite picks specifically,
     bigger edge is significantly associated with WORSE accuracy
     (Spearman r=-0.109, p=0.027) -- low-edge favorites hit ~57-59%,
     high-edge favorites collapse to ~40-48%. Underdog picks showed no
     edge relationship at all (p=0.91), so underdogs are left unfiltered.

Rule: keep every underdog consensus pick, and only favorite consensus
picks whose edge is <= config.FAVORITE_EDGE_MAX_THRESHOLD (default 0.7%,
chosen from where 14's quintile results were still strong before the
cliff). Pick'em games (spread_line == 0) are excluded, same as 13/14.

IMPORTANT CAVEAT this script prints explicitly: the 0.7% cutoff was chosen
by inspecting quintile boundaries computed on this same backtest. Running
that exact threshold against the same data isn't a clean out-of-sample
test of the cutoff itself -- it shows whether combining what 13 and 14
found produces a plausible improved backtest, not proof this specific
number holds going forward. A rigorous version would re-derive the
threshold inside each walk-forward fold using only prior seasons -- worth
doing before trusting this for real money.

Output is written in the same schema as 07's picks file (season, week,
correct, profit, etc.), so it plugs directly into
10_season_backtest_report.py for the week-by-week view:
    python scripts/10_season_backtest_report.py --input data/processed/combined_rule_picks.csv --all-seasons

Run:
    python scripts/15_combined_rule.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from feature_utils import american_odds_to_implied_prob


def main():
    input_path = config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run 07_walk_forward_validation.py first.")

    picks = pd.read_csv(input_path)
    required = {"logit_prob", "xgb_prob", "predicted_home_cover", "odds", "spread_line",
                "correct", "profit", "season", "week"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{input_path} is missing column(s) {missing}.")

    before = len(picks)
    picks = picks[picks["odds"].notna()].copy()
    dropped = before - len(picks)
    if dropped:
        print(f"Dropped {dropped} picks with no market price available.")

    print(
        f"CAVEAT: FAVORITE_EDGE_MAX_THRESHOLD ({config.FAVORITE_EDGE_MAX_THRESHOLD*100:.1f}%) was "
        "chosen by inspecting quintile boundaries on THIS SAME backtest (script 14's output). "
        "This run shows whether combining 13+14's findings produces a plausible improved "
        "backtest -- it is NOT a clean out-of-sample validation of this exact cutoff. Treat "
        "accordingly.\n"
    )

    combined_prob_home = (picks["logit_prob"] + picks["xgb_prob"]) / 2
    is_home = picks["predicted_home_cover"] == 1
    model_prob_for_pick = np.where(is_home, combined_prob_home, 1 - combined_prob_home)
    implied_prob_for_pick = american_odds_to_implied_prob(picks["odds"]).to_numpy()
    picks["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)

    picked_em = picks["spread_line"] == 0
    picked_favorite = np.where(is_home, picks["spread_line"] > 0, picks["spread_line"] < 0)
    picks["picked_favorite"] = picked_favorite
    n_pickem = int(picked_em.sum())
    picks = picks[~picked_em].copy()
    if n_pickem:
        print(f"Excluded {n_pickem} pick'em games (spread_line == 0).\n")

    is_underdog = ~picks["picked_favorite"]
    is_low_edge_favorite = picks["picked_favorite"] & (picks["edge"] <= config.FAVORITE_EDGE_MAX_THRESHOLD)
    selected = is_underdog | is_low_edge_favorite

    n_favorites_total = int(picks["picked_favorite"].sum())
    n_favorites_kept = int(is_low_edge_favorite.sum())
    n_underdogs = int(is_underdog.sum())
    print(f"Underdog picks (all kept): {n_underdogs}")
    print(f"Favorite picks kept (edge <= {config.FAVORITE_EDGE_MAX_THRESHOLD*100:.1f}%): "
          f"{n_favorites_kept} of {n_favorites_total} "
          f"({n_favorites_kept / n_favorites_total * 100:.1f}% of favorites survive the filter)\n")

    selected_picks = picks[selected].copy()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    selected_picks.to_csv(config.COMBINED_RULE_PICKS_CSV, index=False)

    n_total = len(selected_picks)
    n_correct = int(selected_picks["correct"].sum())
    accuracy = n_correct / n_total
    se = (accuracy * (1 - accuracy) / n_total) ** 0.5
    z_score = (accuracy - 0.5) / se if se > 0 else float("nan")
    units = selected_picks["profit"].sum()
    roi = selected_picks["profit"].mean()
    roi_se = selected_picks["profit"].std(ddof=1) / (n_total ** 0.5) if n_total > 1 else float("nan")

    print(f"--- Combined rule: all picks vs. 07's unfiltered baseline ---")
    baseline_n = len(picks)
    baseline_acc = picks["correct"].mean()
    baseline_units = picks["profit"].sum()
    baseline_roi = picks["profit"].mean()
    print(f"07 baseline (no filter, ex-pick'em): n={baseline_n}, accuracy={baseline_acc*100:.1f}%, "
          f"units={baseline_units:+.2f}, ROI={baseline_roi*100:+.1f}%")
    print(f"Combined rule:                       n={n_total}, accuracy={accuracy*100:.1f}%, "
          f"units={units:+.2f}, ROI={roi*100:+.1f}%")
    print(f"\nAccuracy: {accuracy*100:.1f}% (SE +/-{se*100:.1f} pts), z-score vs 50% = {z_score:.2f}")
    print(f"ROI: {roi*100:+.1f}% (SE +/-{roi_se*100:.1f} pts)")
    print(f"Volume: {n_total} picks ({n_total / baseline_n * 100:.1f}% of the unfiltered {baseline_n})")

    with open(config.COMBINED_RULE_METRICS_JSON, "w") as f:
        json.dump({
            "favorite_edge_max_threshold": config.FAVORITE_EDGE_MAX_THRESHOLD,
            "n_underdogs": n_underdogs,
            "n_favorites_kept": n_favorites_kept,
            "n_favorites_total": n_favorites_total,
            "n_total_picks": n_total,
            "accuracy": accuracy,
            "accuracy_se": se,
            "z_score": z_score,
            "units": units,
            "roi": roi,
            "roi_se": roi_se,
            "baseline_n": baseline_n,
            "baseline_accuracy": baseline_acc,
            "baseline_units": baseline_units,
            "baseline_roi": baseline_roi,
        }, f, indent=2)

    print(f"\n  -> {config.COMBINED_RULE_PICKS_CSV}")
    print(f"  -> {config.COMBINED_RULE_METRICS_JSON}")
    print(f"\nFor the week-by-week / season view:")
    print(f"  python scripts/10_season_backtest_report.py --input {config.COMBINED_RULE_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
