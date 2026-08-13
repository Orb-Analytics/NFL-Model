"""
Season backtest report: turns one season's worth of a picks file into a
week-by-week record/units table with a running season total.

The pooled multi-season z-score/ROI numbers from 07/08/09 answer "is there
a statistically detectable edge at all" -- a fair question, but not the one
that decides whether to roll a strategy out for a real season. That
decision is about the actual shape of a season: does it win consistently
week to week, or is a decent overall number secretly one huge week
propping up a lot of average ones? This script exists to make that shape
visible, one season at a time.

Defaults to reading 09_consensus_edge_walk_forward.py's output
(config.CONSENSUS_EDGE_PICKS_CSV), since that's the strictest/most
realistic rule tested so far, but works against any picks file with the
same season/week/correct/profit columns -- pass --input to point it at
08's edge_eval_picks.csv (filtered to one model) or 07's
walk_forward_consensus_picks.csv instead.

Run:
    python scripts/10_season_backtest_report.py                  # latest season in the file
    python scripts/10_season_backtest_report.py --season 2025
    python scripts/10_season_backtest_report.py --input data/processed/edge_eval_picks.csv --model combined --season 2025
    python scripts/10_season_backtest_report.py --all-seasons     # one row per season, for comparing consistency across years
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None,
        help="Path to a picks CSV (default: config.CONSENSUS_EDGE_PICKS_CSV, i.e. script 09's output).",
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Season to report on (default: the most recent season present in the picks file).",
    )
    parser.add_argument(
        "--model", default=None,
        help="If the picks file has a 'model' column (e.g. 08's edge_eval_picks.csv, which mixes "
             "logit/xgb/combined together), filter to just this one before reporting.",
    )
    parser.add_argument(
        "--all-seasons", action="store_true",
        help="Instead of one season's weekly breakdown, print one row per season (record, win%%, "
             "units) so season-to-season consistency can be compared directly -- this is closer "
             "to what actually determines rollout confidence than any single season's detail.",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.CONSENSUS_EDGE_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run the script that produces it first -- "
            f"09_consensus_edge_walk_forward.py for the default input."
        )

    picks = pd.read_csv(input_path)
    required = {"season", "week", "correct", "profit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing column(s) {missing} -- this script expects the same shape "
            "as 09_consensus_edge_walk_forward.py's or 08_edge_based_evaluation.py's picks output."
        )

    if args.model is not None:
        if "model" not in picks.columns:
            raise ValueError("--model was given but this picks file has no 'model' column.")
        picks = picks[picks["model"] == args.model]
        if picks.empty:
            raise ValueError(f"No rows with model == '{args.model}' in {input_path}.")

    if args.all_seasons:
        by_season = (
            picks.groupby("season")
            .agg(n_picks=("correct", "size"), wins=("correct", "sum"), units=("profit", "sum"))
            .reset_index()
            .sort_values("season")
        )
        by_season["losses"] = by_season["n_picks"] - by_season["wins"]
        by_season["win_pct"] = (by_season["wins"] / by_season["n_picks"] * 100).round(1)
        by_season["cume_units"] = by_season["units"].cumsum()
        by_season = by_season[["season", "n_picks", "wins", "losses", "win_pct", "units", "cume_units"]]

        total_picks = len(picks)
        total_wins = int(picks["correct"].sum())
        total_units = picks["profit"].sum()
        n_seasons = len(by_season)
        n_winning_seasons = int((by_season["units"] > 0).sum())

        print(f"All-seasons summary -- {input_path.name}" + (f" (model: {args.model})" if args.model else ""))
        print(
            by_season.to_string(
                index=False,
                formatters={"units": "{:+.2f}".format, "cume_units": "{:+.2f}".format, "win_pct": "{:.1f}%".format},
            )
        )
        print(f"\n{n_winning_seasons}/{n_seasons} seasons finished with positive units.")
        print(f"Grand total: {total_wins}-{total_picks - total_wins} ({total_wins/total_picks*100:.1f}%), "
              f"{total_units:+.2f} units across all seasons.")

        out_path = Path(config.SEASON_REPORT_CSV_TEMPLATE.format(season="all"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        by_season.to_csv(out_path, index=False)
        print(f"\n  -> {out_path}")
        return

    season = args.season if args.season is not None else int(picks["season"].max())
    season_picks = picks[picks["season"] == season].sort_values("week")
    if season_picks.empty:
        available = sorted(picks["season"].unique())
        raise ValueError(f"No picks found for season {season}. Seasons present in this file: {available}")

    print(f"Season {season} backtest report -- {input_path.name}"
          + (f" (model: {args.model})" if args.model else ""))
    print(f"Selection rule that produced these picks: see the script that generated {input_path.name}.\n")

    weekly = (
        season_picks.groupby("week")
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

    total_picks = len(season_picks)
    total_wins = int(season_picks["correct"].sum())
    total_losses = total_picks - total_wins
    total_win_pct = total_wins / total_picks * 100
    total_units = season_picks["profit"].sum()

    print(f"Overall record: {total_wins}-{total_losses} ({total_win_pct:.1f}%), {total_picks} picks")
    print(f"Overall units:  {total_units:+.2f}\n")

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

    out_path = Path(config.SEASON_REPORT_CSV_TEMPLATE.format(season=season))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
