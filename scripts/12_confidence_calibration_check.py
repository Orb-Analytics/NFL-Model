"""
Confidence calibration check: does the MODEL's own raw confidence (how far
its probability sits from a coin flip, with no market comparison at all)
predict which consensus picks are more likely to be correct?

11_edge_calibration_check.py already showed market-based edge (model
probability blended against and compared to the market's implied
probability) has no relationship to accuracy for this feature set --
filtering by it just cuts volume without concentrating good picks. This is
a different, more basic question: forget the market entirely -- when the
model itself is more confident (its probability further from 50%), is it
actually more often right?

Confidence is defined as |combined_prob - 0.5|, where combined_prob is the
average of the logistic regression's and XGBoost's predicted probability
that the home team covers. This is symmetric across home and away picks:
a combined_prob of 0.30 (a confident AWAY pick) and 0.70 (a confident HOME
pick) both give a confidence of 0.20 -- taking the absolute value collapses
both directions into one comparable number, correctly, regardless of which
side the model actually picked.

If this comes back positive and significant, it's a legitimate lever for
cutting volume and raising ROI (bet only when the model is more sure of
itself) -- something market-based edge failed to deliver. It's also the
first step toward a properly calibrated probability, which is a
prerequisite for a market-edge calculation to mean anything (if "the model
says 65%" doesn't actually correspond to being right 65% of the time,
there's no reason a 65%-vs-market comparison should be informative either).

Run:
    python scripts/12_confidence_calibration_check.py
    python scripts/12_confidence_calibration_check.py --input data/processed/walk_forward_consensus_picks.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None,
        help="Path to a consensus-picks CSV with logit_prob/xgb_prob/correct columns (default: "
             "config.WALK_FORWARD_CONSENSUS_PICKS_CSV, i.e. 07's output -- the UNFILTERED "
             "consensus picks, same reasoning as 11: filtering first would bias this check).",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run 07_walk_forward_validation.py first."
        )

    picks = pd.read_csv(input_path)
    required = {"logit_prob", "xgb_prob", "correct"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{input_path} is missing column(s) {missing}.")

    has_profit = "profit" in picks.columns and picks["profit"].notna().any()

    combined_prob = (picks["logit_prob"] + picks["xgb_prob"]) / 2
    picks["confidence"] = (combined_prob - 0.5).abs()

    print(f"Confidence computed for {len(picks)} consensus picks (from {input_path.name}).")
    print(f"Confidence range: {picks['confidence'].min()*100:.1f}% to {picks['confidence'].max()*100:.1f}%, "
          f"mean {picks['confidence'].mean()*100:.1f}%\n")

    n_buckets = min(config.CONFIDENCE_CALIBRATION_N_BUCKETS, picks["confidence"].nunique())
    picks["confidence_decile"] = pd.qcut(picks["confidence"], n_buckets, labels=False, duplicates="drop") + 1

    agg_dict = {
        "n": ("correct", "size"),
        "min_confidence": ("confidence", "min"),
        "max_confidence": ("confidence", "max"),
        "mean_confidence": ("confidence", "mean"),
        "accuracy": ("correct", "mean"),
    }
    if has_profit:
        agg_dict["roi"] = ("profit", "mean")

    deciles = picks.groupby("confidence_decile").agg(**agg_dict).reset_index()
    deciles["accuracy_se"] = (deciles["accuracy"] * (1 - deciles["accuracy"]) / deciles["n"]) ** 0.5

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(config.CONFIDENCE_CALIBRATION_DECILE_CSV, index=False)

    formatters = {
        "min_confidence": "{:.1%}".format,
        "max_confidence": "{:.1%}".format,
        "mean_confidence": "{:.1%}".format,
        "accuracy": "{:.1%}".format,
        "accuracy_se": "±{:.1%}".format,
    }
    if has_profit:
        formatters["roi"] = "{:+.1%}".format

    print("--- Accuracy by model-confidence decile (1 = least confident, 10 = most confident) ---")
    print(deciles.to_string(index=False, formatters=formatters))

    corr, p_value = spearmanr(picks["confidence"], picks["correct"])
    print(f"\nSpearman correlation between confidence and correctness: {corr:+.3f} (p={p_value:.3f})")

    # Also check: does accuracy for just the TOP decile (most confident 10%)
    # clear the naive baseline by enough to matter for a volume-reduction
    # strategy? This is the practical question, separate from statistical
    # significance across the whole range.
    top_decile = deciles.loc[deciles["confidence_decile"] == deciles["confidence_decile"].max()]
    if len(top_decile) > 0:
        top_n = int(top_decile["n"].iloc[0])
        top_acc = top_decile["accuracy"].iloc[0]
        print(f"\nMost-confident decile alone: n={top_n}, accuracy={top_acc*100:.1f}%"
              + (f", ROI={top_decile['roi'].iloc[0]*100:+.1f}%" if has_profit else ""))

    if p_value >= 0.05:
        verdict = (
            "Not statistically distinguishable from zero -- the model's own confidence is NOT "
            "reliably predicting which consensus picks are more likely to be correct, at least not "
            "with this feature set. Combined with 11's null result for market-based edge, this "
            "means neither 'the model disagrees a lot with the market' nor 'the model is very sure "
            "of itself' currently identifies the better picks within the consensus set -- the raw "
            "agreement signal (07's rule) appears to be carrying essentially all the real signal "
            "found so far, and it isn't (yet) separable into a higher-confidence subset. Cutting "
            "volume by confidence would, like edge, just be trimming the sample close to randomly."
        )
    elif corr > 0:
        verdict = (
            "Positive and significant -- the model IS more often right when it's more confident in "
            "itself. This is a real, usable lever: a volume-reducing, ROI-raising rule built on "
            "confidence (not market edge) is worth designing next. Also worth doing from here: a "
            "full calibration curve (does 70% confidence actually mean ~70% accuracy, or just "
            "'more accurate than 55% confidence') to get the mapping right before setting a "
            "specific cutoff."
        )
    else:
        verdict = (
            "Negative and significant -- the model is LESS accurate exactly when it's most "
            "confident in itself, a real overconfidence problem. This would mean the most-sure-of-"
            "itself picks are actively the ones to distrust, not to bet more on -- worth "
            "investigating before using confidence as a filter in either direction."
        )
    print("\n" + verdict)

    with open(config.CONFIDENCE_CALIBRATION_METRICS_JSON, "w") as f:
        json.dump({
            "n_picks": len(picks),
            "spearman_corr": corr,
            "spearman_p_value": p_value,
            "deciles": deciles.to_dict(orient="records"),
        }, f, indent=2, default=str)

    print(f"\n  -> {config.CONFIDENCE_CALIBRATION_DECILE_CSV}")
    print(f"  -> {config.CONFIDENCE_CALIBRATION_METRICS_JSON}")


if __name__ == "__main__":
    main()
