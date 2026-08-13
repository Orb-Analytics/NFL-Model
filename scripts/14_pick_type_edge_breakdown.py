"""
Pick-type x edge breakdown: is market edge actually informative within
favorite picks or within underdog picks specifically, even though 11 found
no relationship pooling both groups together?

13_pick_type_breakdown.py found a real split: favorite picks are
break-even (51.3% accuracy, -2.29 units), underdog picks carry virtually
all the profit (56.0%, +71.56 units). It's possible edge (checked in 11,
pooling favorites and underdogs together) actually IS informative within
one of these groups, and averaging the two groups together washed the
effect out -- e.g. if edge helps identify the best underdog picks but is
pure noise for favorites (or vice versa), the pooled correlation could look
like nothing even though a real, useful pattern exists inside one slice.

Splits consensus picks into favorite / underdog (same logic as 13), then
within EACH group independently: buckets by edge into quintiles (fewer
buckets than 11's deciles, since each group here is smaller than the full
pooled set) and reports accuracy/units/ROI per bucket, plus a Spearman
correlation between edge and correctness computed separately for each
group.

Edge, same definition used everywhere else in this build (08/09/11/19/20):
    edge = EDGE_MODEL_WEIGHT * model_prob + (1 - EDGE_MODEL_WEIGHT) * market_prob - market_prob
         = EDGE_MODEL_WEIGHT * (model_prob - market_prob)
where model_prob is the averaged logit+xgb probability for the side
actually picked, and market_prob is that side's market-implied probability
from its American odds.

Defaults to 07's 2-way consensus picks, but accepts --input to run the same
breakdown against any other picks file with the same schema (e.g.
21_three_way_consensus.py's 3-way output) -- output files get a suffix
matching the input filename's stem so different runs don't overwrite each
other.

Run:
    python scripts/14_pick_type_edge_breakdown.py
    python scripts/14_pick_type_edge_breakdown.py --input data/processed/three_way_consensus_picks.csv
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


def bucket_and_summarize(df: pd.DataFrame, label: str, n_buckets: int) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    n_actual_buckets = min(n_buckets, df["edge"].nunique())
    if n_actual_buckets < 2:
        print(f"{label}: not enough distinct edge values to bucket (n={len(df)}) -- skipping.\n")
        return pd.DataFrame(), {"group": label, "n": len(df), "spearman_corr": float("nan"), "spearman_p_value": float("nan")}

    df["edge_bucket"] = pd.qcut(df["edge"], n_actual_buckets, labels=False, duplicates="drop") + 1
    buckets = (
        df.groupby("edge_bucket")
        .agg(
            n=("correct", "size"),
            min_edge=("edge", "min"),
            max_edge=("edge", "max"),
            accuracy=("correct", "mean"),
            units=("profit", "sum"),
            roi=("profit", "mean"),
        )
        .reset_index()
    )
    buckets.insert(0, "group", label)

    corr, p_value = spearmanr(df["edge"], df["correct"])
    print(f"--- {label} (n={len(df)}) ---")
    print(
        buckets.drop(columns="group").to_string(
            index=False,
            formatters={
                "min_edge": "{:+.1%}".format,
                "max_edge": "{:+.1%}".format,
                "accuracy": "{:.1%}".format,
                "units": "{:+.2f}".format,
                "roi": "{:+.1%}".format,
            },
        )
    )
    print(f"Spearman correlation (edge vs. correctness) within {label}: {corr:+.3f} (p={p_value:.3f})\n")

    return buckets, {"group": label, "n": len(df), "spearman_corr": corr, "spearman_p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a picks CSV to use instead of 07's default 2-way consensus output "
             "(e.g. data/processed/three_way_consensus_picks.csv).",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run 07_walk_forward_validation.py first.")

    suffix = "" if not args.input else f"_{input_path.stem}"
    breakdown_csv = config.PROCESSED_DIR / f"pick_type_edge_breakdown{suffix}.csv"
    metrics_json = config.PROCESSED_DIR / f"pick_type_edge_metrics{suffix}.json"

    picks = pd.read_csv(input_path)
    required = {"predicted_home_cover", "odds", "spread_line", "correct", "profit"}
    if "edge" not in picks.columns:
        required |= {"logit_prob", "xgb_prob"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{input_path} is missing column(s) {missing}.")

    before = len(picks)
    picks = picks[picks["odds"].notna()].copy()
    dropped = before - len(picks)
    if dropped:
        print(f"Dropped {dropped} picks with no market price available.\n")

    is_home = picks["predicted_home_cover"] == 1
    if "edge" in picks.columns:
        print(f"Using the 'edge' column already present in {input_path.name} (not recomputed here).\n")
    else:
        # Same edge definition as 08/09/11: blend the model's probability for
        # the picked side against that side's market-implied probability.
        combined_prob_home = (picks["logit_prob"] + picks["xgb_prob"]) / 2
        model_prob_for_pick = np.where(is_home, combined_prob_home, 1 - combined_prob_home)
        implied_prob_for_pick = american_odds_to_implied_prob(picks["odds"]).to_numpy()
        picks["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)

    # Same favorite/underdog logic as 13. Pick'em games (spread_line == 0)
    # are excluded -- no favorite to classify against.
    picked_em = picks["spread_line"] == 0
    picked_is_favorite = np.where(is_home, picks["spread_line"] > 0, picks["spread_line"] < 0)
    picks["picked_favorite"] = picked_is_favorite
    picks = picks[~picked_em].copy()

    favorites = picks[picks["picked_favorite"]]
    underdogs = picks[~picks["picked_favorite"]]

    all_buckets = []
    all_corrs = []

    b, c = bucket_and_summarize(favorites, "favorite picks", config.PICK_TYPE_EDGE_N_BUCKETS)
    all_buckets.append(b)
    all_corrs.append(c)

    b, c = bucket_and_summarize(underdogs, "underdog picks", config.PICK_TYPE_EDGE_N_BUCKETS)
    all_buckets.append(b)
    all_corrs.append(c)

    combined_df = pd.concat([b for b in all_buckets if not b.empty], ignore_index=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(breakdown_csv, index=False)

    for c in all_corrs:
        if pd.isna(c["spearman_p_value"]):
            continue
        verdict = "informative" if c["spearman_p_value"] < 0.05 else "NOT distinguishable from zero"
        print(f"{c['group']}: edge-vs-correctness correlation is {verdict} within this group "
              f"(r={c['spearman_corr']:+.3f}, p={c['spearman_p_value']:.3f}, n={c['n']}).")

    with open(metrics_json, "w") as f:
        json.dump(all_corrs, f, indent=2, default=str)

    print(f"\n  -> {breakdown_csv}")


if __name__ == "__main__":
    main()
