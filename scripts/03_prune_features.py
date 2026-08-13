"""
Prune the wide training set down from "every predictor we could find" to
"predictors worth keeping" -- this is the deliberate second half of the
"bring in as many as possible, prune later" approach.

Two categories get cut here:
  1. Constant columns (zero information -- e.g. fg_missed_0_19, which is
     ~always 0 because teams essentially never miss from inside the 20).
  2. Columns that are null more than SPARSE_NULL_THRESHOLD_PCT of the time
     for reasons OTHER than the known/accepted PFR pre-2018 coverage gap.

That second category needs the era-aware check: a PFR-derived column being
>50% null across the FULL 2010-2025 training set is expected (PFR advstats
only exist 2018+, see README's "Decision: full 2010-2025 history" section)
and should NOT be dropped just because it's null before 2018. So for any
column with "pfr" in its name, this script re-checks null % restricted to
season >= PFR_ADVSTATS_MIN_SEASON before deciding whether it's genuinely
sparse or just outside PFR's coverage window.

This script is intentionally conservative: it only drops columns with a
clear, explainable reason, and logs every drop (and why) to prune_log.csv.
Correlation filtering and model-driven feature importance are deliberately
NOT done here -- those require picking a model first, which is the next
step after this one.

Run:
    python scripts/03_prune_features.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from feature_engineering import build_feature_manifest


def classify_column(df: pd.DataFrame, manifest_row: pd.Series) -> tuple[bool, str]:
    """Returns (should_drop, reason) for a single candidate predictor column."""
    col = manifest_row["column"]

    if manifest_row["is_constant"]:
        return True, "constant"

    if manifest_row["pct_non_null"] >= config.SPARSE_NULL_THRESHOLD_PCT:
        return False, "keep"

    is_pfr_col = "pfr" in col.lower()
    if not is_pfr_col:
        return True, f"sparse ({manifest_row['pct_non_null']}% non-null, no known cause)"

    # PFR column below the flat threshold -- check if that's fully explained
    # by the pre-2018 coverage gap by re-measuring null % within the PFR era only.
    if "season" not in df.columns:
        return True, "sparse (pfr column, but couldn't verify era -- no season column found)"

    era_mask = df["season"] >= config.PFR_ADVSTATS_MIN_SEASON
    if era_mask.sum() == 0:
        return True, "sparse (pfr column, no rows in PFR-coverage era to check)"

    pct_non_null_in_era = df.loc[era_mask, col].notna().mean() * 100
    if pct_non_null_in_era >= config.SPARSE_NULL_THRESHOLD_PCT_WITHIN_PFR_ERA:
        return False, f"keep (pfr column, {round(pct_non_null_in_era, 1)}% non-null within {config.PFR_ADVSTATS_MIN_SEASON}+ era -- pre-{config.PFR_ADVSTATS_MIN_SEASON} nulls are expected)"
    else:
        return True, f"sparse even within PFR era ({round(pct_non_null_in_era, 1)}% non-null since {config.PFR_ADVSTATS_MIN_SEASON})"


def main():
    df = pd.read_csv(config.TRAINING_SET_CSV, low_memory=False)
    manifest = build_feature_manifest(df)

    log_rows = []
    drop_cols = []
    for _, row in manifest.iterrows():
        should_drop, reason = classify_column(df, row)
        log_rows.append({"column": row["column"], "dropped": should_drop, "reason": reason})
        if should_drop:
            drop_cols.append(row["column"])

    pruned = df.drop(columns=drop_cols)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(config.TRAINING_SET_PRUNED_CSV, index=False)

    log_df = pd.DataFrame(log_rows).sort_values(["dropped", "column"], ascending=[False, True])
    log_df.to_csv(config.PRUNE_LOG_CSV, index=False)

    print(f"Started with {df.shape[1]} columns ({len(manifest)} candidate predictors)")
    print(f"Dropped {len(drop_cols)} columns:")

    constant_n = sum(1 for r in log_rows if r["dropped"] and r["reason"] == "constant")
    sparse_n = sum(1 for r in log_rows if r["dropped"] and r["reason"] != "constant")
    print(f"  - {constant_n} constant")
    print(f"  - {sparse_n} sparse (not explained by the PFR pre-2018 gap)")

    kept_pfr_era_note = sum(1 for r in log_rows if not r["dropped"] and "pfr column" in r["reason"])
    if kept_pfr_era_note:
        print(f"Kept {kept_pfr_era_note} PFR columns whose nulls are fully explained by pre-2018 coverage.")

    print(f"\nPruned training set: {pruned.shape[0]} rows, {pruned.shape[1]} columns")
    print(f"  -> {config.TRAINING_SET_PRUNED_CSV}")
    print(f"  -> {config.PRUNE_LOG_CSV} (every column's fate + reason, for a sanity check)")


if __name__ == "__main__":
    main()
