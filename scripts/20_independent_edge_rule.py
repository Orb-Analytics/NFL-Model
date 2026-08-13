"""
Independent favorite/underdog edge floors: a 2D grid over a favorite-side
edge threshold and an underdog-side edge threshold, swept INDEPENDENTLY --
the direct fix for what 19_positive_edge_rule.py's single shared floor left
unsolved.

19's result: a single edge floor applied uniformly to every pick fixed two
of three concerns (positive edge everywhere, lower volume) but NOT the
third -- favorite_pct barely moved (31.1% -> 23-29%) as the shared
threshold rose, because a uniform cut removes favorites and underdogs in
roughly the same proportion. It doesn't rebalance the mix; it just shrinks
both sides together.

This script instead applies TWO independent thresholds -- one for favorite
picks, one for underdog picks -- and sweeps every combination of
config.FAVORITE_EDGE_THRESHOLDS_2D x config.UNDERDOG_EDGE_THRESHOLDS_2D
(default 5x5=25 combinations). For each cell it reports n_picks,
favorite/underdog split, accuracy, z-score, units, and ROI, so a
combination can be picked that actually pulls favorite_pct toward center
(not just shrinks volume) while still meeting accuracy/significance/volume
targets. The intuition for why asymmetric thresholds might do this: 13/14
found favorites have real (if smaller-sample) edge-accuracy structure at
low edge, while underdogs showed no edge relationship in that pooled check
-- so a LOWER bar for favorites and a HIGHER bar for underdogs is the most
football-motivated way to thin underdogs preferentially without just
discarding the favorite signal 13/14 already found.

Same caveat as everything downstream of 13/14/15/18: this is still an
in-sample exploration on one backtest, not a nested walk-forward
validation. Use it to pick a promising combination to sanity-check further,
not as a final, provably robust rule.

Pass --favorite-threshold and --underdog-threshold together to also write a
full picks file at one specific grid cell (for 10_season_backtest_report.py):
    python scripts/20_independent_edge_rule.py --favorite-threshold 0.0 --underdog-threshold 0.02

Run:
    python scripts/20_independent_edge_rule.py                 # grid only
    python scripts/20_independent_edge_rule.py --favorite-threshold 0.0 --underdog-threshold 0.02
"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--favorite-threshold", type=float, default=None,
        help="Edge floor for favorite picks (fraction, e.g. 0.0). Requires --underdog-threshold too.",
    )
    parser.add_argument(
        "--underdog-threshold", type=float, default=None,
        help="Edge floor for underdog picks (fraction, e.g. 0.02). Requires --favorite-threshold too.",
    )
    args = parser.parse_args()
    if (args.favorite_threshold is None) != (args.underdog_threshold is None):
        parser.error("--favorite-threshold and --underdog-threshold must be given together.")

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

    baseline_n = len(picks)
    baseline_favorite_pct = picks["picked_favorite"].mean() * 100
    print(f"NOTE: edge is computed against this dataset's historical odds (~standard -110), not "
          f"Novig's actual prices -- if Novig's odds are more favorable, real edge at each "
          f"combination below is likely HIGHER than shown here, not lower.\n")
    print(f"Baseline (unfiltered consensus, ex-pick'em): {baseline_n} picks, "
          f"{baseline_favorite_pct:.1f}% favorite / {100 - baseline_favorite_pct:.1f}% underdog.\n")

    is_favorite = picks["picked_favorite"]
    is_underdog = ~picks["picked_favorite"]

    grid_rows = []
    for fav_thresh in config.FAVORITE_EDGE_THRESHOLDS_2D:
        for dog_thresh in config.UNDERDOG_EDGE_THRESHOLDS_2D:
            keep = (is_favorite & (picks["edge"] >= fav_thresh)) | (is_underdog & (picks["edge"] >= dog_thresh))
            eligible = picks[keep]
            n = len(eligible)
            if n == 0:
                grid_rows.append({
                    "favorite_threshold": fav_thresh, "underdog_threshold": dog_thresh,
                    "n_picks": 0, "n_favorite": 0, "n_underdog": 0,
                    "favorite_pct": float("nan"), "accuracy": float("nan"),
                    "accuracy_se": float("nan"), "z_score": float("nan"),
                    "units": 0.0, "roi": float("nan"), "pct_of_unfiltered": 0.0,
                })
                continue
            n_favorite = int(eligible["picked_favorite"].sum())
            n_underdog = n - n_favorite
            accuracy = eligible["correct"].mean()
            se = (accuracy * (1 - accuracy) / n) ** 0.5
            z = (accuracy - 0.5) / se if se > 0 else float("nan")
            units = eligible["profit"].sum()
            roi = eligible["profit"].mean()
            grid_rows.append({
                "favorite_threshold": fav_thresh,
                "underdog_threshold": dog_thresh,
                "n_picks": n,
                "n_favorite": n_favorite,
                "n_underdog": n_underdog,
                "favorite_pct": n_favorite / n * 100,
                "accuracy": accuracy,
                "accuracy_se": se,
                "z_score": z,
                "units": units,
                "roi": roi,
                "pct_of_unfiltered": n / baseline_n * 100,
            })

    grid_df = pd.DataFrame(grid_rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(config.INDEPENDENT_EDGE_RULE_GRID_CSV, index=False)

    print(f"--- Independent favorite/underdog edge threshold grid "
          f"({len(config.FAVORITE_EDGE_THRESHOLDS_2D)}x{len(config.UNDERDOG_EDGE_THRESHOLDS_2D)}) ---")
    print(
        grid_df.to_string(
            index=False,
            formatters={
                "favorite_threshold": "{:.1%}".format,
                "underdog_threshold": "{:.1%}".format,
                "favorite_pct": lambda x: f"{x:.1f}%" if pd.notna(x) else "n/a",
                "accuracy": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "n/a",
                "accuracy_se": lambda x: f"±{x*100:.1f}pt" if pd.notna(x) else "n/a",
                "z_score": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
                "units": "{:+.2f}".format,
                "roi": lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "n/a",
                "pct_of_unfiltered": "{:.1f}%".format,
            },
        )
    )
    print(
        f"\nReading this grid: favorite_threshold=0%/underdog_threshold=0% (top-left) reproduces "
        f"19's 0%-floor row ({baseline_n} picks, {baseline_favorite_pct:.1f}% favorite). Move down "
        f"a favorite_threshold's rows and favorite volume thins; move across an underdog_threshold's "
        f"columns and underdog volume thins. Look for a cell where favorite_pct has moved meaningfully "
        f"toward 50% (not just where n_picks is smaller) while z_score and roi are still acceptable -- "
        f"that's evidence the asymmetric filter is doing real rebalancing, not just volume cutting."
    )

    with open(config.INDEPENDENT_EDGE_RULE_METRICS_JSON, "w") as f:
        json.dump({
            "baseline_n": baseline_n,
            "baseline_favorite_pct": baseline_favorite_pct,
            "grid": grid_rows,
        }, f, indent=2, default=str)

    print(f"\n  -> {config.INDEPENDENT_EDGE_RULE_GRID_CSV}")
    print(f"  -> {config.INDEPENDENT_EDGE_RULE_METRICS_JSON}")

    if args.favorite_threshold is not None:
        keep = (
            (is_favorite & (picks["edge"] >= args.favorite_threshold))
            | (is_underdog & (picks["edge"] >= args.underdog_threshold))
        )
        selected = picks[keep].copy()
        selected.to_csv(config.INDEPENDENT_EDGE_RULE_PICKS_CSV, index=False)
        print(f"\nWrote picks file at favorite_threshold={args.favorite_threshold*100:.1f}%, "
              f"underdog_threshold={args.underdog_threshold*100:.1f}%: {len(selected)} picks")
        print(f"  -> {config.INDEPENDENT_EDGE_RULE_PICKS_CSV}")
        print(f"\nFor the week-by-week / season view:")
        print(f"  python scripts/10_season_backtest_report.py --input {config.INDEPENDENT_EDGE_RULE_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
