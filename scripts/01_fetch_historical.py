"""
Pull raw historical data from nflreadpy and cache it locally as parquet.

This script is the ONLY place that talks to nflreadpy for the training
pipeline. Everything downstream reads from data/raw/*.parquet so that
feature-building is reproducible without re-hitting the network every time.

Run:
    python scripts/01_fetch_historical.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import nflreadpy as nfl

import config

# nflreadpy's downloader wraps EVERY download failure -- a genuine 404
# (file doesn't exist yet, e.g. current season's stats before real games
# are played) and a transient network blip (dropped connection, timeout on
# GitHub's release CDN) -- in the same exception type and code path. There
# is no reliable way to tell them apart by exception class, only by
# message text. So: a message containing "404"/"Not Found" is treated as
# permanent (don't waste time retrying, the file just isn't there yet);
# anything else is treated as possibly transient and retried with backoff,
# since this pipeline runs unattended weekly and shouldn't need a manual
# re-run over what's usually just a flaky connection.
_TRANSIENT_RETRY_ATTEMPTS = 3
_TRANSIENT_RETRY_BACKOFF_SECONDS = 8


def _with_retries(fn, description: str):
    last_exc = None
    for attempt in range(1, _TRANSIENT_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "404" in msg or "Not Found" in msg:
                raise  # permanent -- caller decides how to handle (e.g. skip season)
            last_exc = e
            if attempt < _TRANSIENT_RETRY_ATTEMPTS:
                wait = _TRANSIENT_RETRY_BACKOFF_SECONDS * attempt
                print(f"  !! {description} failed (attempt {attempt}/{_TRANSIENT_RETRY_ATTEMPTS}, "
                      f"looks transient: {e.__class__.__name__}: {e}) -- retrying in {wait}s...")
                time.sleep(wait)
    print(f"  !! {description} failed after {_TRANSIENT_RETRY_ATTEMPTS} attempts, giving up: {last_exc}")
    raise last_exc


def _coerce_object_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix a dtype trap introduced by loading one season at a time instead
    of one batched call: if a genuinely-numeric column comes back with a
    slightly different pandas dtype in different seasons' files, pd.concat
    can silently downgrade that column across the WHOLE combined frame.
    That's silent and only surfaces much later, as a cryptic
    'Pandas data cast to numpy dtype of object' error out of statsmodels
    when fitting a model on it. Handles TWO distinct cases, confirmed from
    two separate real failures:

    1. Classic `object` dtype (e.g. pd.concat([Float64-with-nulls,
       object-with-None]) silently produces `object`) -- re-coerced via
       pd.to_numeric, unless doing so would blow away more than half the
       column's values (a genuine string column, e.g. team abbreviations,
       fails to convert almost entirely and is correctly left alone).
    2. Pandas NULLABLE extension dtypes (Int64/Float64/boolean -- capital
       letters, distinct from numpy's int64/float64), a known output of
       nflreadpy's polars-backed `.to_pandas()` conversion. These are NOT
       reported as `object` dtype by pandas -- confirmed from a second real
       failure where an `== object` check found nothing, yet statsmodels
       still failed converting the DataFrame block to one numpy array.
       Any extension-dtype column is force-cast to float64; if that fails
       (a genuine non-numeric extension dtype, e.g. a string/categorical
       column), it's left alone.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == object:
            already_nan = df[col].isna()
            converted = pd.to_numeric(df[col], errors="coerce")
            newly_nan = converted.isna() & ~already_nan
            if len(df) == 0 or (newly_nan.sum() / len(df)) < 0.5:
                df[col] = converted
        elif pd.api.types.is_extension_array_dtype(dtype):
            try:
                df[col] = df[col].astype("float64")
            except (ValueError, TypeError):
                pass  # genuine non-numeric extension dtype -- leave alone
    return df


def _load_team_stats_resilient(seasons: list[int]) -> pd.DataFrame:
    """Load week-level team stats one season at a time instead of one batch
    call, so a missing file for the CURRENT season (nflverse doesn't publish
    stats_team_week_{season}.parquet until that season's games start
    generating stats -- a 404 is expected/normal before Week 1, including
    during preseason) doesn't take down every other season's data with it.
    Any season that fails to load (permanently or after retries) is skipped
    with a warning; the caller still gets every season that succeeded.
    """
    frames = []
    skipped = []
    for season in seasons:
        try:
            df = _with_retries(
                lambda s=season: nfl.load_team_stats(seasons=[s], summary_level="week").to_pandas(),
                f"load_team_stats({season})",
            )
            frames.append(df)
        except Exception as e:
            skipped.append(season)
            print(f"  !! No team stats available for {season} ({e.__class__.__name__}: {e})")

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
    combined = pd.concat(frames, ignore_index=True)
    return _coerce_object_numeric_columns(combined)


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching schedules for seasons: {config.SEASONS[0]}-{config.SEASONS[-1]}")
    schedules = _with_retries(lambda: nfl.load_schedules(seasons=True).to_pandas(), "load_schedules")
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
                    df = _with_retries(
                        lambda s=season: nfl.load_pfr_advstats(
                            seasons=[s], stat_type=stat_type, summary_level="week"
                        ).to_pandas(),
                        f"load_pfr_advstats({stat_type}, {season})",
                    )
                    frames.append(df)
                except Exception as e:
                    skipped.append(season)
                    print(f"  !! No PFR '{stat_type}' advstats for {season} "
                          f"({e.__class__.__name__}: {e})")
            if skipped:
                print(f"  Skipped {len(skipped)} season(s) for '{stat_type}': {skipped}")
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = _coerce_object_numeric_columns(df)
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
