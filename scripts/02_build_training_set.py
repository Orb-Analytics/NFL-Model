"""
Build the historical training set from cached raw data.

Run 01_fetch_historical.py first. This script never talks to the network --
it only reads data/raw/*.parquet, so it's cheap to re-run while you iterate
on feature logic in build_features.py / feature_engineering.py.

Run:
    python scripts/02_build_training_set.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from build_features import build_full_dataset
from feature_engineering import engineer_all, build_feature_manifest


def load_extra_sources() -> dict[str, pd.DataFrame]:
    """Loads whatever optional raw sources 01_fetch_historical.py cached.
    Missing files are skipped (e.g. PFR advstats disabled, or fetch failed
    for one stat_type) rather than raising -- keeps this script runnable
    even if you haven't pulled every optional source yet.
    """
    sources = {}
    if config.INCLUDE_PFR_ADVSTATS:
        for stat_type in config.PFR_ADVSTATS_TYPES:
            path = Path(str(config.RAW_PFR_ADVSTATS_PATH).format(stat_type=stat_type))
            if path.exists():
                sources[f"pfr_{stat_type}"] = pd.read_parquet(path)
            else:
                print(f"  (no cached file for pfr_{stat_type} at {path}, skipping)")
    return sources


def main():
    schedules = pd.read_parquet(config.RAW_SCHEDULES_PATH)
    team_stats = pd.read_parquet(config.RAW_TEAM_STATS_PATH)
    extra_sources = load_extra_sources()

    full = build_full_dataset(schedules, team_stats, extra_sources=extra_sources)
    full = engineer_all(full)

    # training set = games that have already been played
    training = full[full["home_score"].notna() & full["away_score"].notna()].copy()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training.to_csv(config.TRAINING_SET_CSV, index=False)
    training.to_excel(config.TRAINING_SET_XLSX, index=False)

    manifest = build_feature_manifest(training)
    manifest.to_csv(config.FEATURE_MANIFEST_CSV, index=False)

    print(f"\nTraining set: {training.shape[0]} rows, {training.shape[1]} columns")
    print(f"  -> {config.TRAINING_SET_CSV}")
    print(f"  -> {config.TRAINING_SET_XLSX} (snapshot for eyeballing, not the source of truth)")
    print(f"  -> {config.FEATURE_MANIFEST_CSV} (every candidate predictor, for pruning)")

    constant_cols = manifest[manifest["is_constant"]]["column"].tolist()
    if constant_cols:
        print(f"\n{len(constant_cols)} constant columns found (safe to drop first pass): {constant_cols}")

    sparse_cols = manifest[manifest["pct_non_null"] < 50]["column"].tolist()
    if sparse_cols:
        print(f"{len(sparse_cols)} columns are >50% null (check before including): {sparse_cols}")

    upcoming = full[full["home_score"].isna()]
    if not upcoming.empty:
        print(f"\n{upcoming.shape[0]} upcoming/unplayed games found (not included in training set).")


if __name__ == "__main__":
    main()
