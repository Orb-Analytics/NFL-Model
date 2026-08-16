"""
Threshold-sweep follow-up to 28_devigged_edge_breakdown.py: instead of
bucketing into deciles (fixed COUNT per bucket, so no bucket boundary lines
up with a round number like "1% edge"), this reads the two picks CSVs 28
already wrote (data/processed/devig_check_raw_picks.csv and
devig_check_devigged_picks.csv) and reports accuracy/units/ROI for
"picks with edge >= X%" at each threshold in config.EDGE_THRESHOLDS,
raw vs. de-vigged side by side -- directly answering "does a >=1% edge
cutoff actually perform better than betting everything" for both edge
definitions, since the deciles alone can't be sliced at an exact round
number.

Prerequisite: run 28_devigged_edge_breakdown.py first (it writes the two
picks CSVs this reads).

Run:
    python scripts/29_devig_threshold_sweep.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

RAW_PICKS_CSV = config.PROCESSED_DIR / "devig_check_raw_picks.csv"
DEVIG_PICKS_CSV = config.PROCESSED_DIR / "devig_check_devigged_picks.csv"


def sweep(picks_df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for threshold in config.EDGE_THRESHOLDS:
        subset = picks_df[picks_df["edge"] >= threshold]
        n = len(subset)
        if n == 0:
            rows.append({"threshold": threshold, "n": 0, "accuracy": float("nan"),
                         "z_vs_50pct": float("nan"), "units": 0.0, "roi": float("nan")})
            continue
        accuracy = subset["correct"].mean()
        se = (accuracy * (1 - accuracy) / n) ** 0.5
        z = (accuracy - 0.5) / se if se > 0 else float("nan")
        units = subset["profit"].sum()
        roi = subset["profit"].mean()
        rows.append({"threshold": threshold, "n": n, "accuracy": accuracy,
                     "z_vs_50pct": z, "units": units, "roi": roi})
    result = pd.DataFrame(rows)
    print(f"\n--- {label}: picks with edge >= threshold (cumulative, not decile-bucketed) ---")
    print(result.to_string(
        index=False,
        formatters={
            "threshold": "{:.1%}".format,
            "accuracy": "{:.1%}".format,
            "z_vs_50pct": "{:+.2f}".format,
            "units": "{:+.2f}".format,
            "roi": "{:+.1%}".format,
        },
    ))
    return result


def main():
    if not RAW_PICKS_CSV.exists() or not DEVIG_PICKS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_PICKS_CSV} / {DEVIG_PICKS_CSV} not found. Run "
            f"28_devigged_edge_breakdown.py first -- it writes both files."
        )

    raw_df = pd.read_csv(RAW_PICKS_CSV)
    devig_df = pd.read_csv(DEVIG_PICKS_CSV)

    raw_result = sweep(raw_df, "RAW (vig-included) edge")
    devig_result = sweep(devig_df, "DE-VIGGED edge (corrected)")

    print("\n=== Picks with edge >= 1.0% specifically ===")
    for label, result in [("Raw", raw_result), ("De-vigged", devig_result)]:
        row = result[result["threshold"] == 0.01]
        if row.empty:
            print(f"{label}: 1.0% is not one of config.EDGE_THRESHOLDS -- add it there to get an exact match.")
            continue
        r = row.iloc[0]
        print(f"{label}: n={int(r['n'])}, accuracy={r['accuracy']*100:.1f}%, z={r['z_vs_50pct']:+.2f} vs 50%, "
              f"units={r['units']:+.2f}, ROI={r['roi']*100:+.1f}%")

    out_csv = config.PROCESSED_DIR / "devig_threshold_sweep.csv"
    combined = raw_result.copy()
    combined = combined.rename(columns={c: f"raw_{c}" for c in combined.columns if c != "threshold"})
    combined = combined.merge(
        devig_result.rename(columns={c: f"devig_{c}" for c in devig_result.columns if c != "threshold"}),
        on="threshold",
    )
    combined.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")


if __name__ == "__main__":
    main()
