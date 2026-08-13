"""
Positive-edge rule: require every pick -- favorite OR underdog -- to clear
a positive edge against the market, instead of 15/18's approach of only
filtering favorites while taking every underdog unconditionally.

Three concerns prompted this, all addressed by the same fix:

  1. A real betting operation should only give out +EV picks by the
     model's own estimate, on both sides of the ledger -- not "all
     underdogs regardless of price, plus selective favorites." Note this
     backtest's edge is computed against nflverse's historical odds
     (~standard -110), not Novig's actual live prices, which are expected
     to be more favorable in general (more +money on spreads) -- so this
     is likely a CONSERVATIVE estimate of live edge, not an inflated one.
  2. Volume under 15/18 (~7-10 picks/week) is too high for a real
     operation. A universal edge floor cuts volume directly.
  3. The favorite/underdog split under 15/18 (~80/20) is more lopsided
     than desired. Filtering underdogs by edge too (not just favorites)
     should pull the split back toward center, since the cut is no longer
     one-sided.

Rather than pick one threshold, this sweeps config.POSITIVE_EDGE_THRESHOLDS_SWEEP
(0% to 3%) and reports, at each level: total picks, favorite/underdog
split, accuracy, z-score, units, and ROI -- the actual tradeoff curve to
choose a threshold from, rather than guessing one number. Pick'em games are
excluded (no favorite to classify against, consistent with 13/14/15/17/18).

Pass --threshold to also write the full picks file at ONE specific
threshold (for follow-up with 10_season_backtest_report.py):
    python scripts/19_positive_edge_rule.py --threshold 0.01

Run:
    python scripts/19_positive_edge_rule.py                  # sweep only
    python scripts/19_positive_edge_rule.py --threshold 0.01  # sweep + picks file at 1%
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
        "--threshold", type=float, default=None,
        help="Also write the full picks file at this one threshold (fraction, e.g. 0.01 for 1%%).",
    )
    args = parser.parse_args()

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
    print(f"NOTE: edge is computed against this dataset's historical odds (~standard -110), not "
          f"Novig's actual prices -- if Novig's odds are more favorable, real edge at each "
          f"threshold below is likely HIGHER than shown here, not lower.\n")

    sweep_rows = []
    for threshold in config.POSITIVE_EDGE_THRESHOLDS_SWEEP:
        eligible = picks[picks["edge"] >= threshold]
        n = len(eligible)
        if n == 0:
            sweep_rows.append({
                "threshold": threshold, "n_picks": 0, "n_favorite": 0, "n_underdog": 0,
                "favorite_pct": float("nan"), "accuracy": float("nan"), "accuracy_se": float("nan"),
                "z_score": float("nan"), "units": 0.0, "roi": float("nan"),
                "pct_of_unfiltered": 0.0,
            })
            continue
        n_favorite = int(eligible["picked_favorite"].sum())
        n_underdog = n - n_favorite
        accuracy = eligible["correct"].mean()
        se = (accuracy * (1 - accuracy) / n) ** 0.5
        z = (accuracy - 0.5) / se if se > 0 else float("nan")
        units = eligible["profit"].sum()
        roi = eligible["profit"].mean()
        sweep_rows.append({
            "threshold": threshold,
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

    sweep_df = pd.DataFrame(sweep_rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(config.POSITIVE_EDGE_RULE_SWEEP_CSV, index=False)

    print(f"--- Positive-edge threshold sweep (baseline: {baseline_n} unfiltered consensus picks, "
          f"ex-pick'em) ---")
    print(
        sweep_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.1%}".format,
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
        "\nReading this table: as threshold rises, n_picks and pct_of_unfiltered fall (volume "
        "reduction), favorite_pct should move away from the ~32% baseline share toward something "
        "less lopsided if edge filtering is cutting underdogs proportionally more (class-imbalance "
        "check), and z_score/roi show what's actually being given up (or gained) in exchange for "
        "the lower volume. Pick the threshold that best matches your volume target without giving "
        "up more accuracy/significance than you're willing to."
    )

    with open(config.POSITIVE_EDGE_RULE_METRICS_JSON, "w") as f:
        json.dump({"baseline_n": baseline_n, "sweep": sweep_rows}, f, indent=2, default=str)

    print(f"\n  -> {config.POSITIVE_EDGE_RULE_SWEEP_CSV}")
    print(f"  -> {config.POSITIVE_EDGE_RULE_METRICS_JSON}")

    if args.threshold is not None:
        selected = picks[picks["edge"] >= args.threshold].copy()
        selected.to_csv(config.POSITIVE_EDGE_RULE_PICKS_CSV, index=False)
        print(f"\nWrote picks file at threshold={args.threshold*100:.1f}%: {len(selected)} picks")
        print(f"  -> {config.POSITIVE_EDGE_RULE_PICKS_CSV}")
        print(f"\nFor the week-by-week / season view:")
        print(f"  python scripts/10_season_backtest_report.py --input {config.POSITIVE_EDGE_RULE_PICKS_CSV} --all-seasons")


if __name__ == "__main__":
    main()
