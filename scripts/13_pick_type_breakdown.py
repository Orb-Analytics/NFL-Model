"""
Pick-type breakdown: does the consensus rule's edge concentrate in specific
KINDS of games, rather than being spread evenly across all of them?

11 and 12 both checked probability-derived quantities (market edge, model
self-confidence) as ways to cut volume and raise ROI, and both came back
null -- neither separates good picks from bad ones within the consensus
set. This checks a different, football-motivated axis instead: which side
was picked (home or away) and whether that side was the market favorite or
underdog (from spread_line's sign). These are pre-registered categories a
football person would think to check, not another round of fishing through
arbitrary thresholds.

Four combinations, plus the two marginal splits:
  - home vs. away (regardless of favorite/underdog)
  - favorite vs. underdog (regardless of home/away)
  - home favorite, home underdog, away favorite, away underdog

spread_line convention (nflverse, matching the rest of this pipeline):
positive = home favored by that many points, negative = away favored.
Games with spread_line == 0 (a pick'em -- no favorite) are reported in
their own category rather than forced into either bucket.

Run:
    python scripts/13_pick_type_breakdown.py
    python scripts/13_pick_type_breakdown.py --input data/processed/consensus_edge_picks.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"category": label, "n_picks": 0, "wins": 0, "losses": 0,
                "accuracy": float("nan"), "units": 0.0, "roi": float("nan")}
    wins = int(df["correct"].sum())
    accuracy = wins / n
    has_profit = "profit" in df.columns and df["profit"].notna().any()
    units = df["profit"].sum() if has_profit else float("nan")
    roi = df["profit"].mean() if has_profit else float("nan")
    return {
        "category": label,
        "n_picks": n,
        "wins": wins,
        "losses": n - wins,
        "accuracy": accuracy,
        "units": units,
        "roi": roi,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None,
        help="Path to a picks CSV with predicted_home_cover/spread_line/correct columns "
             "(default: config.WALK_FORWARD_CONSENSUS_PICKS_CSV, i.e. 07's unfiltered output).",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    suffix = "" if not args.input else f"_{input_path.stem}"
    breakdown_csv = config.PROCESSED_DIR / f"pick_type_breakdown{suffix}.csv"

    picks = pd.read_csv(input_path)

    # Different picks files spell out the picked side differently -- 07's
    # output has predicted_home_cover (1/0); 09's has picked_side (a string).
    # Normalize to a single is_home boolean so this works against either.
    if "predicted_home_cover" in picks.columns:
        is_home = picks["predicted_home_cover"] == 1
    elif "picked_side" in picks.columns:
        is_home = picks["picked_side"] == "home"
    else:
        raise ValueError(
            f"{input_path} has neither 'predicted_home_cover' nor 'picked_side' -- "
            "can't determine which side was picked."
        )

    if "spread_line" not in picks.columns:
        raise ValueError(f"{input_path} is missing 'spread_line' -- needed to determine favorite/underdog.")
    if "correct" not in picks.columns:
        raise ValueError(f"{input_path} is missing 'correct'.")

    picked_em = picks["spread_line"] == 0
    # positive spread_line = home favored; negative = away favored.
    picked_is_favorite = np.where(
        is_home, picks["spread_line"] > 0, picks["spread_line"] < 0
    )

    picks = picks.copy()
    picks["picked_home"] = is_home
    picks["picked_favorite"] = picked_is_favorite
    picks["picked_em"] = picked_em

    non_pickem = picks[~picked_em]
    rows = []
    rows.append(summarize(picks, "all picks"))
    rows.append(summarize(picks[picked_em], "pick'em (spread_line == 0, excluded from favorite/dog splits below)"))
    rows.append(summarize(picks[is_home], "home picks"))
    rows.append(summarize(picks[~is_home], "away picks"))
    rows.append(summarize(non_pickem[non_pickem["picked_favorite"]], "favorite picks"))
    rows.append(summarize(non_pickem[~non_pickem["picked_favorite"]], "underdog picks"))
    rows.append(summarize(non_pickem[non_pickem["picked_home"] & non_pickem["picked_favorite"]], "home favorite"))
    rows.append(summarize(non_pickem[non_pickem["picked_home"] & ~non_pickem["picked_favorite"]], "home underdog"))
    rows.append(summarize(non_pickem[~non_pickem["picked_home"] & non_pickem["picked_favorite"]], "away favorite"))
    rows.append(summarize(non_pickem[~non_pickem["picked_home"] & ~non_pickem["picked_favorite"]], "away underdog"))

    result_df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(breakdown_csv, index=False)

    print(f"Pick-type breakdown -- {input_path.name} ({len(picks)} total picks)\n")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "n/a",
                "units": lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a",
                "roi": lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "n/a",
            },
        )
    )
    print(f"\n  -> {breakdown_csv}")


if __name__ == "__main__":
    main()
