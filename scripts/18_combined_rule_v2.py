"""
Combined rule v2: same underdogs-unfiltered + low-edge-favorites-only idea
as 15_combined_rule.py, but with SEPARATE edge thresholds for home
favorites and away favorites, following 17's finding that they behave
differently.

  15's rule: underdogs unfiltered, favorites kept if edge <= 0.7% (one
             shared threshold for home and away favorites).
  17's finding: home favorites only look reliable in the lowest edge
             bucket (below ~0.4%; the very next bucket is already close to
             break-even), while away favorites hold up through a wider
             range (up to ~1.3-1.4% edge) before collapsing.

This rule: underdogs unfiltered (same as 15), home favorites kept only if
edge <= config.HOME_FAVORITE_EDGE_MAX_THRESHOLD (default 0.4%), away
favorites kept if edge <= config.AWAY_FAVORITE_EDGE_MAX_THRESHOLD (default
1.3%).

STRONGER caveat than 15's already-flagged one: these two thresholds came
from bucket boundaries computed on an even smaller, further-split sample
(89 home favorite picks across only 3 buckets) than 15's single threshold
was. More granularity here means more opportunity to fit noise, not
necessarily a more trustworthy number -- this script exists to see if the
refinement helps on this backtest, not to declare it correct. Compare
directly against 15's output before preferring either one, and treat both
as needing the same eventual nested walk-forward validation before real
money is involved.

Output uses the same schema as 07's/15's picks files, so it plugs directly
into 10_season_backtest_report.py:
    python scripts/10_season_backtest_report.py --input data/processed/combined_rule_v2_picks.csv --all-seasons

Run:
    python scripts/18_combined_rule_v2.py
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
        f"CAVEAT: HOME_FAVORITE_EDGE_MAX_THRESHOLD "
        f"({config.HOME_FAVORITE_EDGE_MAX_THRESHOLD*100:.1f}%) and "
        f"AWAY_FAVORITE_EDGE_MAX_THRESHOLD ({config.AWAY_FAVORITE_EDGE_MAX_THRESHOLD*100:.1f}%) were "
        "chosen by inspecting bucket boundaries on THIS SAME backtest (script 17's output), on an "
        "even smaller/further-split sample than 15's single threshold. This run shows whether the "
        "refinement helps here -- it is NOT a clean out-of-sample validation of either number.\n"
    )

    combined_prob_home = (picks["logit_prob"] + picks["xgb_prob"]) / 2
    is_home = picks["predicted_home_cover"] == 1
    model_prob_for_pick = np.where(is_home, combined_prob_home, 1 - combined_prob_home)
    implied_prob_for_pick = american_odds_to_implied_prob(picks["odds"]).to_numpy()
    picks["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)

    picked_em = picks["spread_line"] == 0
    picked_favorite = np.where(is_home, picks["spread_line"] > 0, picks["spread_line"] < 0)
    picks["picked_favorite"] = picked_favorite
    picks["is_home"] = is_home
    n_pickem = int(picked_em.sum())
    picks = picks[~picked_em].copy()
    if n_pickem:
        print(f"Excluded {n_pickem} pick'em games (spread_line == 0).\n")

    is_underdog = ~picks["picked_favorite"]
    is_home_favorite = picks["picked_favorite"] & picks["is_home"]
    is_away_favorite = picks["picked_favorite"] & ~picks["is_home"]

    keep_home_favorite = is_home_favorite & (picks["edge"] <= config.HOME_FAVORITE_EDGE_MAX_THRESHOLD)
    keep_away_favorite = is_away_favorite & (picks["edge"] <= config.AWAY_FAVORITE_EDGE_MAX_THRESHOLD)
    selected = is_underdog | keep_home_favorite | keep_away_favorite

    n_home_fav_total = int(is_home_favorite.sum())
    n_home_fav_kept = int(keep_home_favorite.sum())
    n_away_fav_total = int(is_away_favorite.sum())
    n_away_fav_kept = int(keep_away_favorite.sum())
    n_underdogs = int(is_underdog.sum())

    print(f"Underdog picks (all kept): {n_underdogs}")
    print(f"Home favorite picks kept: {n_home_fav_kept} of {n_home_fav_total} "
          f"({n_home_fav_kept / n_home_fav_total * 100:.1f}% survive)" if n_home_fav_total else "Home favorite picks: none")
    print(f"Away favorite picks kept: {n_away_fav_kept} of {n_away_fav_total} "
          f"({n_away_fav_kept / n_away_fav_total * 100:.1f}% survive)\n" if n_away_fav_total else "Away favorite picks: none\n")

    selected_picks = picks[selected].copy()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    selected_picks.to_csv(config.COMBINED_RULE_V2_PICKS_CSV, index=False)

    n_total = len(selected_picks)
    n_correct = int(selected_picks["correct"].sum())
    accuracy = n_correct / n_total
    se = (accuracy * (1 - accuracy) / n_total) ** 0.5
    z_score = (accuracy - 0.5) / se if se > 0 else float("nan")
    units = selected_picks["profit"].sum()
    roi = selected_picks["profit"].mean()
    roi_se = selected_picks["profit"].std(ddof=1) / (n_total ** 0.5) if n_total > 1 else float("nan")

    baseline_n = len(picks)
    baseline_acc = picks["correct"].mean()
    baseline_units = picks["profit"].sum()
    baseline_roi = picks["profit"].mean()

    print(f"--- v2 (separate home/away favorite thresholds) vs. 07's unfiltered baseline ---")
    print(f"07 baseline (no filter, ex-pick'em): n={baseline_n}, accuracy={baseline_acc*100:.1f}%, "
          f"units={baseline_units:+.2f}, ROI={baseline_roi*100:+.1f}%")
    print(f"Combined rule v2:                    n={n_total}, accuracy={accuracy*100:.1f}%, "
          f"units={units:+.2f}, ROI={roi*100:+.1f}%")
    print(f"\nAccuracy: {accuracy*100:.1f}% (SE +/-{se*100:.1f} pts), z-score vs 50% = {z_score:.2f}")
    print(f"ROI: {roi*100:+.1f}% (SE +/-{roi_se*100:.1f} pts)")
    print(f"Volume: {n_total} picks ({n_total / baseline_n * 100:.1f}% of the unfiltered {baseline_n})")

    with open(config.COMBINED_RULE_V2_METRICS_JSON, "w") as f:
        json.dump({
            "home_favorite_edge_max_threshold": config.HOME_FAVORITE_EDGE_MAX_THRESHOLD,
            "away_favorite_edge_max_threshold": config.AWAY_FAVORITE_EDGE_MAX_THRESHOLD,
            "n_underdogs": n_underdogs,
            "n_home_favorites_kept": n_home_fav_kept,
            "n_home_favorites_total": n_home_fav_total,
            "n_away_favorites_kept": n_away_fav_kept,
            "n_away_favorites_total": n_away_fav_total,
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

    print(f"\n  -> {config.COMBINED_RULE_V2_PICKS_CSV}")
    print(f"  -> {config.COMBINED_RULE_V2_METRICS_JSON}")
    print(f"\nCompare against 15's v1 result (data/processed/combined_rule_metrics.json) before "
          f"preferring either one -- more granular thresholds are not automatically better.")
    print(f"\nFor the week-by-week / season view:")
    print(f"  python scripts/10_season_backtest_report.py --input {config.COMBINED_RULE_V2_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
