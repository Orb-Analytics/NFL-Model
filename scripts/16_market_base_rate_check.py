"""
Market favorite/underdog base rate: a completely MODEL-FREE check on
whether favorites actually cover the spread less often than underdogs,
across the full historical dataset -- not just the 1,286 consensus picks
from the walk-forward run, and not involving any model prediction at all.

Prompted by: "since the model predicts home cover probability, why is it so
skewed toward picking underdogs?" (13's finding: 875 of 1,286 consensus
picks were on the underdog side). The likely explanation is that this
isn't a model artifact -- it's the model correctly learning a real pattern
in the market: sportsbooks are generally understood to shade spread lines
a bit further toward favorites than pure team quality would justify
(commonly attributed to the public preferring to bet favorites/name-brand
teams, forcing books to move the number to balance action rather than to
set a perfectly fair line). If that's true, the raw historical cover rate
for favorites should sit measurably under 50%, and for underdogs measurably
over 50%, independent of any model.

This script checks that directly: every game in training_set.csv (full
2010-2025 history, not just the walk-forward window), split by whether the
HOME or AWAY team was favored (from spread_line's sign), with the actual
cover outcome (home_cover/away_cover, already computed in
build_features.add_labels). Pick'em games (spread_line == 0, no favorite)
and pushes (margin == spread_line exactly, no cover either way) are
excluded from the rate calculation and reported separately.

ALSO breaks the same rate out BY SEASON (not just pooled across all 16
years) -- prompted by a real concern: 15/18's combined rules bet the
underdog on 80%+ of picks, which is a large, concentrated structural bet
that the historical underdog tilt keeps holding. If that tilt reverses in
a given season (favorites just cover more, whether from real market
correction or plain variance), a strategy this lopsided has little in it
to cushion that. The season-by-season base rate here is the model-free
half of checking that risk directly -- compare it against
10_season_backtest_report.py's season-by-season units for
combined_rule_picks.csv (or v2's) to see whether the strategy's worst
seasons line up with years the market-wide base rate favored favorites.

Run:
    python scripts/16_market_base_rate_check.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config


def summarize(df: pd.DataFrame, favorite_covered_col: pd.Series, label) -> dict:
    n = len(df)
    if n == 0:
        return {"category": label, "n_games": 0, "favorite_cover_rate": float("nan"),
                "underdog_cover_rate": float("nan"), "z_vs_50pct": float("nan")}
    fav_rate = favorite_covered_col.mean()
    se = (fav_rate * (1 - fav_rate) / n) ** 0.5
    z = (fav_rate - 0.5) / se if se > 0 else float("nan")
    return {
        "category": label,
        "n_games": n,
        "favorite_cover_rate": fav_rate,
        "underdog_cover_rate": 1 - fav_rate,
        "z_vs_50pct": z,
    }


def main():
    df = pd.read_csv(config.TRAINING_SET_CSV, low_memory=False)

    required = {"spread_line", "margin", "home_cover", "away_cover"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{config.TRAINING_SET_CSV} is missing column(s) {missing}.")

    df = df[df["home_cover"].notna()].copy()
    total_games = len(df)

    pickem = df["spread_line"] == 0
    push = df["margin"] == df["spread_line"]
    usable = df[~pickem & ~push].copy()

    print(f"Total games (with a final score): {total_games}")
    print(f"Pick'em games (spread_line == 0, excluded -- no favorite): {int(pickem.sum())}")
    print(f"Pushes (margin == spread_line exactly, excluded -- nobody covered): {int(push.sum())}")
    print(f"Usable games for this check: {len(usable)}\n")

    home_favorite = usable["spread_line"] > 0
    away_favorite = usable["spread_line"] < 0

    favorite_covered = pd.Series(np.where(home_favorite, usable["home_cover"], usable["away_cover"]),
                                  index=usable.index)

    rows = []
    rows.append(summarize(usable, favorite_covered, "all games (home or away favored)"))
    rows.append(summarize(usable[home_favorite], favorite_covered[home_favorite], "home team favored"))
    rows.append(summarize(usable[away_favorite], favorite_covered[away_favorite], "away team favored"))

    result_df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(config.MARKET_BASE_RATE_CSV, index=False)

    print("--- Model-free favorite vs. underdog cover rate (full 2010-2025 history) ---")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "favorite_cover_rate": "{:.1%}".format,
                "underdog_cover_rate": "{:.1%}".format,
                "z_vs_50pct": "{:+.2f}".format,
            },
        )
    )

    overall = rows[0]
    print(f"\nFavorites cover {overall['favorite_cover_rate']*100:.1f}% of the time, underdogs "
          f"{overall['underdog_cover_rate']*100:.1f}%, across {overall['n_games']} games with no "
          f"model involved at all -- z={overall['z_vs_50pct']:.2f} vs. a fair 50/50.")

    if abs(overall["z_vs_50pct"]) >= 1.96:
        print(
            "\nThis is a real, statistically significant, MODEL-FREE market pattern -- confirms the "
            "model's tilt toward picking underdogs (see 13_pick_type_breakdown.py) reflects an actual "
            "historical inefficiency in how spreads get set, not an artifact specific to this model "
            "or feature set. The model didn't invent this pattern; it learned something that was "
            "already true in the raw outcomes it was trained on."
        )
    else:
        print(
            "\nNot statistically significant on its own at the full-history level -- the model's tilt "
            "toward underdogs may be driven more by how it weighs specific features (team-quality "
            "differentials interacting with spread_line) than by a simple, uniform market-wide bias. "
            "Worth checking whether the effect is stronger in specific eras or spread-size ranges "
            "rather than concluding there's no real pattern here."
        )

    print(f"\n  -> {config.MARKET_BASE_RATE_CSV}")

    # --- By-season breakdown: does the favorite/underdog gap swing a lot
    # season to season, or stay fairly stable? This is the direct,
    # model-free evidence for whether 15/18's underdog-heavy rule is
    # exposed to years where the tilt weakens or reverses. ---
    if "season" not in usable.columns:
        print("\n(no 'season' column found -- skipping by-season breakdown)")
        return

    season_rows = []
    for season, season_df in usable.groupby("season"):
        season_home_fav = season_df["spread_line"] > 0
        season_fav_covered = pd.Series(
            np.where(season_home_fav, season_df["home_cover"], season_df["away_cover"]),
            index=season_df.index,
        )
        season_rows.append(summarize(season_df, season_fav_covered, int(season)))

    season_df_out = pd.DataFrame(season_rows).rename(columns={"category": "season"})
    season_csv_path = config.PROCESSED_DIR / "market_base_rate_by_season.csv"
    season_df_out.to_csv(season_csv_path, index=False)

    print("\n--- Model-free favorite/underdog cover rate BY SEASON ---")
    print(
        season_df_out.to_string(
            index=False,
            formatters={
                "favorite_cover_rate": "{:.1%}".format,
                "underdog_cover_rate": "{:.1%}".format,
                "z_vs_50pct": "{:+.2f}".format,
            },
        )
    )
    print(
        "\nCompare this against 10_season_backtest_report.py's season-by-season units for "
        "combined_rule_picks.csv (or v2's) -- if the strategy's worst seasons line up with seasons "
        "where favorite_cover_rate here is well above 50% (favorites covering more than usual), "
        "that confirms the underdog-heavy rule is exposed to years the tilt weakens, exactly as "
        "expected for a strategy this concentrated on one side. If the strategy holds up even in "
        "those years, that's evidence of real selection skill beyond just riding the tilt."
    )
    print(f"\n  -> {season_csv_path}")


if __name__ == "__main__":
    main()
