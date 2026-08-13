"""
Edge calibration check: does a bigger model/market disagreement actually
predict a more accurate pick, or is "edge" not adding real information on
top of raw model agreement right now?

Prompted directly by a real result: 07_walk_forward_validation.py's raw
consensus rule (agree on side, no price filter) scored z=3.25 pooled across
8 seasons -- the strongest result in this build. 09_consensus_edge_walk_forward.py,
using the exact same two models and the exact same feature set, but ALSO
requiring the edge to clear 2%, scored z=0.60 -- meaningfully WORSE, on a
stricter subset of the same picks. If "edge" (model probability vs. market
implied probability) were doing what it's supposed to -- flagging the picks
worth the most confidence -- filtering to bigger edges should improve
accuracy, not tank it. This script checks that relationship directly rather
than accepting whichever headline number happens to look better.

Method: take every consensus pick from 07's walk-forward run (every game
where logit and XGBoost agreed on a side -- NOT pre-filtered by edge size),
compute each pick's edge using the exact same formula as 08/09
(config.EDGE_MODEL_WEIGHT-weighted blend of the combined model's
probability against the market's implied probability for the side that was
actually picked), and split those picks into deciles by edge size. If
"bigger edge = more trustworthy" is true, accuracy should trend up across
the deciles. If it's flat or trends down, that's a real finding about this
specific model/feature combination -- not a reason to distrust the
underlying consensus signal, just a reason not to use edge size as a
selection filter on top of it (yet).

If the input file already has its own "edge" column (e.g.
23_three_way_edge_aligned.py's output, which computes edge from the
3-model average probability, not the 2-model logit+xgb average this script
would otherwise compute), that existing column is used directly instead of
being recomputed -- so this always reflects whatever edge definition
actually produced the picks being analyzed.

Output filenames get a suffix matching the input file's stem when --input
is used (matching 14's convention), so different runs don't overwrite each
other.

Run:
    python scripts/11_edge_calibration_check.py
    python scripts/11_edge_calibration_check.py --input data/processed/three_way_edge_aligned_picks.csv
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
from feature_utils import american_odds_to_implied_prob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None,
        help="Path to a consensus-picks CSV with logit_prob/xgb_prob/predicted_home_cover/odds/"
             "correct/profit columns (default: config.WALK_FORWARD_CONSENSUS_PICKS_CSV, i.e. "
             "07's output -- the UNFILTERED consensus picks, which is what this diagnostic needs).",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run 07_walk_forward_validation.py first (it now writes "
            "odds/profit columns onto its consensus picks, which this script needs)."
        )

    suffix = "" if not args.input else f"_{input_path.stem}"
    decile_csv = config.PROCESSED_DIR / f"edge_calibration_deciles{suffix}.csv"
    metrics_json = config.PROCESSED_DIR / f"edge_calibration_metrics{suffix}.json"

    picks = pd.read_csv(input_path)
    required = {"predicted_home_cover", "odds", "correct", "profit"}
    if "edge" not in picks.columns:
        required |= {"logit_prob", "xgb_prob"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing column(s) {missing}. This script needs 07's raw consensus "
            "output (with odds/profit already attached), or any picks file with its own 'edge' "
            "column (e.g. 23's output) -- 09's picks file won't work here since it's already been "
            "filtered by edge, which is exactly the thing being checked."
        )

    before = len(picks)
    picks = picks[picks["odds"].notna()].copy()
    dropped = before - len(picks)
    if dropped:
        print(f"Dropped {dropped} picks with no market price available (can't compute edge for these).\n")

    if "edge" in picks.columns:
        print(f"Using the 'edge' column already present in {input_path.name} (not recomputed here -- "
              f"whatever probability definition produced this file's picks is what's being analyzed).")
    else:
        # Same edge definition as 08/09: blend the model's probability for the
        # SIDE THAT WAS ACTUALLY PICKED against that side's market-implied
        # probability, weighted by config.EDGE_MODEL_WEIGHT. Since these are
        # already-agreed consensus picks, "the model's probability" here is the
        # combined (averaged) probability from logit + xgb, same as the
        # "combined" model in 08/09.
        combined_prob_home = (picks["logit_prob"] + picks["xgb_prob"]) / 2
        picked_home = picks["predicted_home_cover"] == 1
        model_prob_for_pick = np.where(picked_home, combined_prob_home, 1 - combined_prob_home)
        implied_prob_for_pick = american_odds_to_implied_prob(picks["odds"]).to_numpy()
        picks["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)

    print(f"Edge available for {len(picks)} picks (from {input_path.name}).")
    print(f"Edge range: {picks['edge'].min()*100:.1f}% to {picks['edge'].max()*100:.1f}%, "
          f"mean {picks['edge'].mean()*100:.1f}%\n")

    # --- Deciles: if edge is informative, accuracy should trend up from
    # decile 1 (lowest/most negative edge) to decile 10 (highest edge). ---
    n_buckets = min(config.EDGE_CALIBRATION_N_BUCKETS, picks["edge"].nunique())
    picks["edge_decile"] = pd.qcut(picks["edge"], n_buckets, labels=False, duplicates="drop") + 1

    deciles = (
        picks.groupby("edge_decile")
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

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(decile_csv, index=False)

    print("--- Accuracy by edge decile (1 = lowest edge, 10 = highest edge) ---")
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

    # --- Formal monotonic-trend check: Spearman correlation between edge
    # and correctness. Positive + significant = bigger edge really does mean
    # more trustworthy. Near zero or negative = edge isn't adding
    # information beyond raw agreement (or is actively misleading) for this
    # model/feature combination. ---
    corr, p_value = spearmanr(picks["edge"], picks["correct"])
    print(f"\nSpearman correlation between edge and correctness: {corr:+.3f} (p={p_value:.3f})")

    if p_value >= 0.05:
        verdict = (
            "Not statistically distinguishable from zero -- edge size is NOT reliably predicting "
            "which consensus picks are more likely to be correct, at least not with this feature "
            "set and this many picks. This is consistent with 09 scoring worse than 07: filtering "
            "by edge isn't concentrating the good picks, it's just cutting the sample size (and "
            "possibly cutting some of the picks that were actually right). Raw agreement (07's rule) "
            "currently looks like the more trustworthy signal than agreement-plus-edge (09's rule)."
        )
    elif corr > 0:
        verdict = (
            "Positive and significant -- bigger edge genuinely does track higher accuracy here. "
            "09 scoring worse than 07 would then need a different explanation (e.g. the specific 2% "
            "cutoff happening to land in an unlucky spot, or interaction with the agreement "
            "requirement) -- worth re-checking 09 at a few different thresholds before concluding "
            "edge filtering doesn't help."
        )
    else:
        verdict = (
            "Negative and significant -- bigger edge is associated with WORSE accuracy, the opposite "
            "of what 'edge' is supposed to mean. That would suggest the model is most confidently "
            "wrong exactly when it disagrees most with the market (a classic overconfidence pattern) "
            "-- worth treating any edge-based selection rule with real suspicion until this is "
            "understood, and leaning on raw consensus (07) instead."
        )
    print("\n" + verdict)

    with open(metrics_json, "w") as f:
        json.dump({
            "n_picks": len(picks),
            "spearman_corr": corr,
            "spearman_p_value": p_value,
            "deciles": deciles.to_dict(orient="records"),
        }, f, indent=2, default=str)

    print(f"\n  -> {decile_csv}")
    print(f"  -> {metrics_json}")


if __name__ == "__main__":
    main()
