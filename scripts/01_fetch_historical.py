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

import pandas as pd
import nflreadpy as nfl

import config


def _load_team_stats_resilient(seasons: list[int]) -> pd.DataFrame:
    """Load week-level team stats one season at a time instead of one batch
    call, so a missing file for the CURRENT season (nflverse doesn't publish
    stats_team_week_{season}.parquet until that season's games start
    generating stats -- a 404 is expected/normal before Week 1, including
    during preseason) doesn't take down every other season's data with it.
    Any season that fails to load is skipped with a warning; the caller
    still gets every season that succeeded.
    """
    frames = []
    skipped = []
    for season in seasons:
        try:
            df = nfl.load_team_stats(seasons=[season], summary_level="week").to_pandas()
            frames.append(df)
        except Exception as e:
            skipped.append(season)
            print(f"  !! No team stats available yet for {season} ({e.__class__.__name__}: {e})")

    if skipped:
        print(f"  Skipped {len(skipped)} season(s) with no team-stats data yet: {skipped}")
        print("     (normal for the current season before real games have been played -- "
              "live scoring only needs completed games' stats, and build_features.py's "
              "rolling/EWMA features already handle upcoming games having no own-week stats.)")
    if not frames:
        raise RuntimeError(
            f"Failed to load team stats for ALL of {seasons} -- this is not the expected "
            "current-season-only gap, something else is wrong (network/nflreadpy issue)."
        )
    return pd.concat(frames, ignore_index=True)


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching schedules for seasons: {config.SEASONS[0]}-{config.SEASONS[-1]}")
    schedules = nfl.load_schedules(seasons=True).to_pandas()
    schedules = schedules[schedules["season"].isin(config.SEASONS)]
    schedules.to_parquet(config.RAW_SCHEDULES_PATH, index=False)
    print(f"  -> {schedules.shape[0]} games saved to {config.RAW_SCHEDULES_PATH}")

    print("Fetching team stats (week-level)...")
    team_stats = _load_team_stats_resilient(config.SEASONS)
    team_stats.to_parquet(config.RAW_TEAM_STATS_PATH, index=False)
    print(f"  -> {team_stats.shape[0]} team-weeks saved to {config.RAW_TEAM_STATS_PATH}")

    if config.INCLUDE_PFR_ADVSTATS:
        pfr_seasons = [s for s in config.SEASONS if s >= config.PFR_ADVSTATS_MIN_SEASON]
        if not pfr_seasons:
            print(f"\nSkipping PFR advstats -- none of {config.SEASONS} are >= {config.PFR_ADVSTATS_MIN_SEASON}.")
        for stat_type in config.PFR_ADVSTATS_TYPES:
            print(f"Fetching PFR advanced stats ({stat_type})...")
            # Same per-season resilience as team stats: a single batched call
            # with pfr_seasons would drop EVERY season's data for this
            # stat_type if just the current season's file doesn't exist yet
            # (normal before real games have been played). Load one season
            # at a time so a current-season 404 only costs that one season.
            frames = []
            skipped = []
            for season in pfr_seasons:
                try:
                    df = nfl.load_pfr_advstats(
                        seasons=[season], stat_type=stat_type, summary_level="week"
                    ).to_pandas()
                    frames.append(df)
                except Exception as e:
                    skipped.append(season)
                    print(f"  !! No PFR '{stat_type}' advstats yet for {season} "
                          f"({e.__class__.__name__}: {e})")
            if skipped:
                print(f"  Skipped {len(skipped)} season(s) for '{stat_type}': {skipped}")
            if frames:
                df = pd.concat(frames, ignore_index=True)
                path = Path(str(config.RAW_PFR_ADVSTATS_PATH).format(stat_type=stat_type))
                df.to_parquet(path, index=False)
                print(f"  -> {df.shape[0]} rows saved to {path}")
                print(f"  -> columns: {list(df.columns)}")
            else:
                print(f"  !! No seasons succeeded for '{stat_type}' -- skipping this source entirely.")

    print("\nColumn check (verify these match what build_features.py expects):")
    print("schedules columns:", list(schedules.columns))
    print("team_stats columns:", list(team_stats.columns))


if __name__ == "__main__":
    main()
