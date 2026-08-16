"""
Week-by-week season report for a custom, asymmetric edge rule applied to
28_devigged_edge_breakdown.py's picks (2-model average, no consensus gate,
de-vigged edge by default):

    keep underdog pick if edge >= UNDERDOG_EDGE_MIN (default 2%)
    keep favorite pick if edge >= FAVORITE_EDGE_MIN (default 1%)

Motivated directly by 31_devig_class_edge_breakdown.py's finding: edge was
the ONE statistically significant predictor of accuracy within underdog
picks (informative in both raw and de-vigged versions, p=0.011/0.035) --
the bottom edge quintile of underdogs was a clear loser (-22 to -24 units)
while the top quintiles were solidly profitable. That's a real reason to
add an underdog edge FLOOR that doesn't exist in the live rule today
(25_live_weekly_scoring.py's keep_underdog takes every underdog pick
unconditionally). An initial 1%/0% pass (underdog/favorite) produced
8-15 picks per week out of a ~13-16 game slate -- far too much volume to
be a selective strategy, essentially betting most of the board. Both
floors were raised (favorite 0%->1%, underdog 1%->2%) specifically to cut
volume down to something that actually looks like a selective rule. This
is intentionally a DIFFERENT, simpler rule than production, being tested
on its own merits, not a re-derivation of the production threshold.

IMPORTANT CAVEATS, stated plainly (same spirit as every other threshold in
this build -- see config.py's FAVORITE_EDGE_MAX_THRESHOLD /
THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD comments):
  1. This is the 2-model (logit+xgb), no-3-way-consensus-gate population
     from 28 -- NOT the actual production 3-way-consensus picks. Applying
     this rule to production would first require re-deriving it against
     3-way-consensus picks specifically.
  2. The 1%/0% thresholds came from eyeballing 31's quintile boundaries on
     this same backtest data -- not an out-of-sample-derived cutoff. A
     single season's week-by-week shape (what this script shows) is
     informative but is not itself a validation of the rule.

Run:
    python scripts/32_custom_rule_season_report.py                       # season 2025, devigged edge, 1%/0%
    python scripts/32_custom_rule_season_report.py --season 2024
    python scripts/32_custom_rule_season_report.py --edge-basis raw
    python scripts/32_custom_rule_season_report.py --underdog-min 0.015 --favorite-min 0.0

Prerequisite: run 28_devigged_edge_breakdown.py first (it writes the two
picks CSVs this reads).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--edge-basis", choices=["devigged", "raw"], default="devigged")
    parser.add_argument("--underdog-min", type=float, default=0.02,
                         help="Minimum edge (decimal, e.g. 0.02 = 2%%), inclusive, required to keep an underdog pick.")
    parser.add_argument("--favorite-min", type=float, default=0.01,
                         help="Minimum edge (decimal, e.g. 0.01 = 1%%), inclusive, required to keep a favorite pick.")
    parser.add_argument("--all-seasons", action="store_true",
                         help="Instead of one season's weekly breakdown, apply the same fixed rule to "
                              "EVERY season independently and print one row per season (record, "
                              "win%%, units) -- this is the actual out-of-a-single-season robustness "
                              "check: does the rule hold up across years, or was one good season "
                              "carrying the story?")
    args = parser.parse_args()

    picks_csv = DEVIG_PICKS_CSV if args.edge_basis == "devigged" else RAW_PICKS_CSV
    if not picks_csv.exists():
        raise FileNotFoundError(
            f"{picks_csv} not found. Run 28_devigged_edge_breakdown.py first -- it writes both "
            f"devig_check_raw_picks.csv and devig_check_devigged_picks.csv."
        )

    picks = pd.read_csv(picks_csv)
    required = {"season", "week", "picked_favorite", "edge", "correct", "profit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(
            f"{picks_csv} is missing column(s) {missing} -- re-run 28_devigged_edge_breakdown.py "
            f"to regenerate it with the current schema."
        )

    def apply_rule(df: pd.DataFrame) -> pd.DataFrame:
        is_fav = df["picked_favorite"].astype(bool)
        keep_dog = ~is_fav & (df["edge"] >= args.underdog_min)
        keep_fav = is_fav & (df["edge"] >= args.favorite_min)
        return df[keep_dog | keep_fav]

    if args.all_seasons:
        print(f"{args.edge_basis} edge -- custom rule: underdog edge >= {args.underdog_min:+.1%}, "
              f"favorite edge >= {args.favorite_min:+.1%}\n")
        rows = []
        for season, season_df in picks.groupby("season"):
            baseline_n = len(season_df)
            selected = apply_rule(season_df)
            n = len(selected)
            if n == 0:
                rows.append({"season": season, "baseline_n": baseline_n, "n_picks": 0,
                             "wins": 0, "losses": 0, "win_pct": float("nan"), "units": 0.0})
                continue
            wins = int(selected["correct"].sum())
            losses = n - wins
            rows.append({
                "season": season, "baseline_n": baseline_n, "n_picks": n,
                "wins": wins, "losses": losses,
                "win_pct": round(wins / n * 100, 1), "units": selected["profit"].sum(),
            })
        by_season = pd.DataFrame(rows).sort_values("season")
        by_season["cume_units"] = by_season["units"].cumsum()

        print(by_season.to_string(
            index=False,
            formatters={"units": "{:+.2f}".format, "cume_units": "{:+.2f}".format,
                        "win_pct": lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a"},
        ))

        n_seasons = len(by_season)
        n_winning = int((by_season["units"] > 0).sum())
        total_n = int(by_season["n_picks"].sum())
        total_wins = int(by_season["wins"].sum())
        total_units = by_season["units"].sum()
        print(f"\n{n_winning}/{n_seasons} seasons finished with positive units under this fixed rule.")
        print(f"Grand total: {total_wins}-{total_n - total_wins} ({total_wins/total_n*100:.1f}%), "
              f"{total_units:+.2f} units across all seasons.")

        out_path = config.PROCESSED_DIR / f"custom_rule_all_seasons_{args.edge_basis}.csv"
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        by_season.to_csv(out_path, index=False)
        print(f"\n  -> {out_path}")
        return

    season_picks = picks[picks["season"] == args.season].copy()
    if season_picks.empty:
        available = sorted(picks["season"].unique())
        raise ValueError(f"No picks found for season {args.season}. Seasons present: {available}")

    is_favorite = season_picks["picked_favorite"].astype(bool)
    selected = apply_rule(season_picks).sort_values("week")

    print(f"Season {args.season}, {args.edge_basis} edge -- custom rule: "
          f"underdog edge >= {args.underdog_min:+.1%}, favorite edge >= {args.favorite_min:+.1%}")
    print(f"{len(season_picks)} total edge-only picks this season -> {len(selected)} survive the rule "
          f"({len(season_picks) - len(selected)} filtered out).\n")

    if selected.empty:
        print("No picks survive this rule for this season -- nothing to report.")
        return

    weekly = (
        selected.groupby("week")
        .agg(n_picks=("correct", "size"), wins=("correct", "sum"), units=("profit", "sum"))
        .reset_index()
    )
    weekly["losses"] = weekly["n_picks"] - weekly["wins"]
    weekly["win_pct"] = (weekly["wins"] / weekly["n_picks"] * 100).round(1)
    weekly["cume_wins"] = weekly["wins"].cumsum()
    weekly["cume_losses"] = weekly["losses"].cumsum()
    weekly["cume_units"] = weekly["units"].cumsum()
    weekly = weekly[
        ["week", "n_picks", "wins", "losses", "win_pct", "units", "cume_wins", "cume_losses", "cume_units"]
    ]

    total_picks = len(selected)
    total_wins = int(selected["correct"].sum())
    total_losses = total_picks - total_wins
    total_win_pct = total_wins / total_picks * 100
    total_units = selected["profit"].sum()
    total_roi = selected["profit"].mean()

    n_underdog = int((~is_favorite[selected.index]).sum())
    n_favorite = int((is_favorite[selected.index]).sum())

    print(f"Overall record: {total_wins}-{total_losses} ({total_win_pct:.1f}%), {total_picks} picks "
          f"({n_underdog} underdog, {n_favorite} favorite)")
    print(f"Overall units:  {total_units:+.2f}  |  ROI: {total_roi*100:+.1f}%\n")

    print("Week-by-week (record and units for that week, plus running season total):")
    print(
        weekly.to_string(
            index=False,
            formatters={
                "units": "{:+.2f}".format,
                "cume_units": "{:+.2f}".format,
                "win_pct": "{:.1f}%".format,
            },
        )
    )

    n_winning_weeks = int((weekly["units"] > 0).sum())
    n_weeks = len(weekly)
    print(f"\n{n_winning_weeks}/{n_weeks} weeks finished with positive units.")

    out_path = config.PROCESSED_DIR / f"custom_rule_season_{args.season}_{args.edge_basis}.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
