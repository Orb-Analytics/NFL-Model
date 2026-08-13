"""
Final combined rule: 3-way consensus (21) as the base, every underdog kept
unfiltered, every favorite kept only if its edge (computed from the
3-model average probability) is <= config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD.

Consolidates everything this build has found:
  - 21: 3-way agreement (logit, xgb, Naive Bayes) is the strongest,
    most-replicated volume/quality lever tried -- stronger than any edge
    threshold (07/09/11/14/19/20 all point the same way).
  - 23: requiring edge-DIRECTION alignment on top of 3-way agreement added
    complexity without adding value (removed only 1.4% of picks, and
    accuracy/z/ROI came back slightly WORSE, not better). Left out of this
    rule for the same reason 15 was preferred over 18 -- Occam's razor,
    not "more filters = better."
  - 13/14/16/17, now replicated a THIRD time on the 3-way-edge-aligned
    picks (p=0.005, the strongest version of this finding yet): underdogs,
    especially HOME underdogs, carry almost all the profit. Favorites are
    break-even to negative in aggregate, but that average hides a real
    split -- low-edge favorites are strong (68.9% accuracy, +30.2% ROI in
    the lowest decile), high-edge favorites collapse (~42% accuracy,
    -16 to -18% ROI).

Rule, in plain terms: bet every underdog the 3-way-consensus model likes,
skip high-edge favorites, keep low-edge favorites.

IMPORTANT CAVEAT, same as 15/18/19/20: THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD
was chosen by looking at a decile boundary computed on THIS SAME backtest.
This is a reasonable first pass at combining everything found so far, not
proof the exact cutoff holds going forward -- a rigorous version re-derives
the threshold inside each walk-forward fold using only prior seasons' data
(nested walk-forward), which hasn't been built yet. Worth doing before this
is trusted with real money.

Run:
    python scripts/24_final_combined_rule.py
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
    input_path = config.THREE_WAY_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run 21_three_way_consensus.py first.")

    picks = pd.read_csv(input_path)
    required = {"logit_prob", "xgb_prob", "nb_prob", "predicted_home_cover", "odds", "spread_line",
                "correct", "profit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{input_path} is missing column(s) {missing}.")

    before = len(picks)
    picks = picks[picks["odds"].notna()].copy()
    dropped = before - len(picks)
    if dropped:
        print(f"Dropped {dropped} picks with no market price available.")

    print(
        f"Threshold: keep favorites with edge <= {config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD*100:.1f}% "
        "-- chosen from a decile boundary on this same backtest (see the caveat in this script's "
        "docstring and config.py's comment above THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD). All "
        "underdogs are kept regardless of edge (14's null finding for underdog edge, replicated "
        "three times now).\n"
    )

    avg_prob_3 = (picks["logit_prob"] + picks["xgb_prob"] + picks["nb_prob"]) / 3
    is_home = picks["predicted_home_cover"] == 1
    model_prob_for_pick = np.where(is_home, avg_prob_3, 1 - avg_prob_3)
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
    is_favorite = picks["picked_favorite"]
    keep_favorite = is_favorite & (picks["edge"] <= config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD)
    selected = is_underdog | keep_favorite

    n_underdogs = int(is_underdog.sum())
    n_fav_total = int(is_favorite.sum())
    n_fav_kept = int(keep_favorite.sum())

    print(f"Underdog picks (all kept): {n_underdogs}")
    print(f"Favorite picks kept: {n_fav_kept} of {n_fav_total} "
          f"({n_fav_kept / n_fav_total * 100:.1f}% survive)" if n_fav_total else "Favorite picks: none")

    selected_picks = picks[selected].copy()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    selected_picks.to_csv(config.FINAL_COMBINED_RULE_PICKS_CSV, index=False)

    n_total = len(selected_picks)
    n_correct = int(selected_picks["correct"].sum())
    accuracy = n_correct / n_total
    se = (accuracy * (1 - accuracy) / n_total) ** 0.5
    z_score = (accuracy - 0.5) / se if se > 0 else float("nan")
    units = selected_picks["profit"].sum()
    roi = selected_picks["profit"].mean()
    roi_se = selected_picks["profit"].std(ddof=1) / (n_total ** 0.5) if n_total > 1 else float("nan")
    favorite_pct = n_fav_kept / n_total * 100 if n_total else float("nan")

    baseline_n = len(picks)
    baseline_acc = picks["correct"].mean()
    baseline_units = picks["profit"].sum()
    baseline_roi = picks["profit"].mean()
    baseline_favorite_pct = is_favorite.mean() * 100

    print(f"\n--- Final combined rule vs. 21's unfiltered 3-way baseline ---")
    print(f"21 baseline (3-way, no filter, ex-pick'em): n={baseline_n}, accuracy={baseline_acc*100:.1f}%, "
          f"units={baseline_units:+.2f}, ROI={baseline_roi*100:+.1f}%, favorite%={baseline_favorite_pct:.1f}%")
    print(f"Final combined rule:                        n={n_total}, accuracy={accuracy*100:.1f}%, "
          f"units={units:+.2f}, ROI={roi*100:+.1f}%, favorite%={favorite_pct:.1f}%")
    print(f"\nAccuracy: {accuracy*100:.1f}% (SE +/-{se*100:.1f} pts), z-score vs 50% = {z_score:.2f}")
    print(f"ROI: {roi*100:+.1f}% (SE +/-{roi_se*100:.1f} pts)")
    print(f"Volume: {n_total} picks ({n_total / baseline_n * 100:.1f}% of the unfiltered {baseline_n})")

    with open(config.FINAL_COMBINED_RULE_METRICS_JSON, "w") as f:
        json.dump({
            "favorite_edge_max_threshold": config.THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD,
            "n_underdogs": n_underdogs,
            "n_favorites_kept": n_fav_kept,
            "n_favorites_total": n_fav_total,
            "n_total_picks": n_total,
            "accuracy": accuracy,
            "accuracy_se": se,
            "z_score": z_score,
            "units": units,
            "roi": roi,
            "roi_se": roi_se,
            "favorite_pct": favorite_pct,
            "baseline_n": baseline_n,
            "baseline_accuracy": baseline_acc,
            "baseline_units": baseline_units,
            "baseline_roi": baseline_roi,
            "baseline_favorite_pct": baseline_favorite_pct,
        }, f, indent=2)

    print(f"\n  -> {config.FINAL_COMBINED_RULE_PICKS_CSV}")
    print(f"  -> {config.FINAL_COMBINED_RULE_METRICS_JSON}")
    print(f"\nFor the week-by-week / season view:")
    print(f"  python scripts/10_season_backtest_report.py --input {config.FINAL_COMBINED_RULE_PICKS_CSV} --all-seasons")
    print(f"\nBefore trusting this for real money: re-derive THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD inside "
          f"a nested walk-forward loop (using only prior seasons per fold) rather than the fixed, "
          f"in-sample-derived value used here.")


if __name__ == "__main__":
    main()
