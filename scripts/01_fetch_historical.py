"""
Pull raw historical data from nflreadpy and cache it locally as parquet.

This script is the ONLY place that talks to nflreadpy for the training
pipeline. Everything downstream reads from data/raw/*.parquet so that
feature-building is reproducible without re-hitting the network every time.

Run:
    python scripts/01_fetch_historical.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import nflreadpy as nfl

import config


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching schedules for seasons: {config.SEASONS[0]}-{config.SEASONS[-1]}")
    schedules = nfl.load_schedules(seasons=True).to_pandas()
    schedules = schedules[schedules["season"].isin(config.SEASONS)]
    schedules.to_parquet(config.RAW_SCHEDULES_PATH, index=False)
    print(f"  -> {schedules.shape[0]} games saved to {config.RAW_SCHEDULES_PATH}")

    print("Fetching team stats (week-level)...")
    team_stats = nfl.load_team_stats(seasons=config.SEASONS, summary_level="week").to_pandas()
    team_stats.to_parquet(config.RAW_TEAM_STATS_PATH, index=False)
    print(f"  -> {team_stats.shape[0]} team-weeks saved to {config.RAW_TEAM_STATS_PATH}")

    if config.INCLUDE_PFR_ADVSTATS:
        pfr_seasons = [s for s in config.SEASONS if s >= config.PFR_ADVSTATS_MIN_SEASON]
        if not pfr_seasons:
            print(f"\nSkipping PFR advstats -- none of {config.SEASONS} are >= {config.PFR_ADVSTATS_MIN_SEASON}.")
        for stat_type in config.PFR_ADVSTATS_TYPES:
            print(f"Fetching PFR advanced stats ({stat_type})...")
            try:
                df = nfl.load_pfr_advstats(
                    seasons=pfr_seasons, stat_type=stat_type, summary_level="week"
                ).to_pandas()
                path = Path(str(config.RAW_PFR_ADVSTATS_PATH).format(stat_type=stat_type))
                df.to_parquet(path, index=False)
                print(f"  -> {df.shape[0]} rows saved to {path}")
                print(f"  -> columns: {list(df.columns)}")
            except Exception as e:
                print(f"  !! Failed to fetch PFR advstats for '{stat_type}': {e}")
                print("     Pipeline will continue without this source -- check nflreadpy's")
                print("     load_pfr_advstats() signature if this persists.")

    print("\nColumn check (verify these match what build_features.py expects):")
    print("schedules columns:", list(schedules.columns))
    print("team_stats columns:", list(team_stats.columns))


if __name__ == "__main__":
    main()
