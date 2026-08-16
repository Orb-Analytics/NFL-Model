"""
Home vs Away and Favorite vs Underdog breakdown for the custom asymmetric
edge rule in 32_custom_rule_season_report.py (default: favorite edge >= 1%,
underdog edge >= 2%, de-vigged), pooled across ALL seasons by default (the
same picks that produced the "Grand total: 456-401 (53.2%), +36.17 units"
8-season result) -- same class-table format as
30_devig_class_breakdown.py, but applied to the RULE-FILTERED survivors
instead of every edge-only pick.

Note the Favorite vs Underdog split here is partly definitional (favorites
only survive at >=1% edge, underdogs only survive at >=2%), so its main
value is comparing accuracy/ROI ACROSS the two surviving groups, not
re-discovering that favorites and underdogs are different populations.
Home vs Away is the more informative half of this breakdown, since the
rule doesn't gate on it at all.

Prerequisite: run 28_devigged_edge_breakdown.py first (it writes the two
picks CSVs this reads).

Run:
    python scripts/33_custom_rule_class_breakdown.py                      # all seasons, devigged, 1%/2%
    python scripts/33_custom_rule_class_breakdown.py --season 2025
    python scripts/33_custom_rule_class_breakdown.py --favorite-min 0.0 --underdog-min 0.01
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

RAW_PICKS_CSV = config.PROCESSED_DIR / "devig_check_raw_picks.csv"
DEVIG_PICKS_CSV = config.PROCESSED_DIR / "devig_check_devigged_picks.csv"


def class_table(picks_df: pd.DataFrame, label: str) -> pd.DataFrame:
    picks_df = picks_df.copy()
    picks_df["home_away"] = picks_df["picked_side"].map({"home": "Home", "away": "Away"})
    picks_df["fav_dog"] = picks_df["picked_favorite"].map({True: "Favorite", False: "Underdog"})

    rows = []
    for dim_name, dim_col in [("Home vs Away", "home_away"), ("Favorite vs Underdog", "fav_dog")]:
        for cls, g in picks_df.groupby(dim_col):
            n = len(g)
            accuracy = g["correct"].mean()
            se = (accuracy * (1 - accuracy) / n) ** 0.5 if n > 0 else float("nan")
            z = (accuracy - 0.5) / se if se > 0 else float("nan")
            units = g["profit"].sum()
            roi = g["profit"].mean()
            rows.append({
                "dimension": dim_name, "class": cls, "n": n,
                "accuracy": accuracy, "z_vs_50pct": z, "units": units, "roi": roi,
            })
    result = pd.DataFrame(rows)

    print(f"\n--- {label} ---")
    print(result.to_string(
        index=False,
        formatters={
            "accuracy": "{:.1%}".format,
            "z_vs_50pct": "{:+.2f}".format,
            "units": "{:+.2f}".format,
            "roi": "{:+.1%}".format,
        },
    ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                         help="Restrict to one season (default: pool all seasons, matching --all-seasons).")
    parser.add_argument("--edge-basis", choices=["devigged", "raw"], default="devigged")
    parser.add_argument("--underdog-min", type=float, default=0.02)
    parser.add_argument("--favorite-min", type=float, default=0.01)
    args = parser.parse_args()

    picks_csv = DEVIG_PICKS_CSV if args.edge_basis == "devigged" else RAW_PICKS_CSV
    if not picks_csv.exists():
        raise FileNotFoundError(f"{picks_csv} not found. Run 28_devigged_edge_breakdown.py first.")

    picks = pd.read_csv(picks_csv)
    required = {"season", "picked_side", "picked_favorite", "edge", "correct", "profit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{picks_csv} is missing column(s) {missing} -- re-run 28 to regenerate it.")

    if args.season is not None:
        picks = picks[picks["season"] == args.season]
        if picks.empty:
            raise ValueError(f"No picks found for season {args.season}.")

    is_fav = picks["picked_favorite"].astype(bool)
    keep_dog = ~is_fav & (picks["edge"] >= args.underdog_min)
    keep_fav = is_fav & (picks["edge"] >= args.favorite_min)
    selected = picks[keep_dog | keep_fav]

    scope = f"season {args.season}" if args.season is not None else "all seasons pooled"
    label = (f"{args.edge_basis} edge, custom rule (favorite >= {args.favorite_min:+.1%}, "
             f"underdog >= {args.underdog_min:+.1%}), {scope} -- {len(selected)} of {len(picks)} picks survive")
    result = class_table(selected, label)

    total_n = len(selected)
    total_wins = int(selected["correct"].sum())
    total_units = selected["profit"].sum()
    print(f"\nOverall: {total_wins}-{total_n-total_wins} ({total_wins/total_n*100:.1f}%), "
          f"{total_units:+.2f} units, n={total_n}")

    out_csv = config.PROCESSED_DIR / f"custom_rule_class_breakdown_{'all' if args.season is None else args.season}_{args.edge_basis}.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")


if __name__ == "__main__":
    main()
