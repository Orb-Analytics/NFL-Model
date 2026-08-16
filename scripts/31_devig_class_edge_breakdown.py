"""
Within-class edge breakdown for 28_devigged_edge_breakdown.py's picks
(2-model average, no consensus gate, side with the higher edge picked on
every game): for EACH class (Home, Away, Favorite, Underdog) separately,
buckets that class's picks into quintiles by edge size and reports
accuracy/units/ROI per bucket -- same method as 14_pick_type_edge_breakdown.py
(which does this for favorite/underdog only, on 07's consensus-gated picks),
extended to all 4 classes and to both the raw-vig and de-vigged edge
definitions from 28, side by side.

Answers: "within picks the model already favors as a class (e.g. away, or
underdog), does bigger edge actually predict a better pick -- or is the
class membership itself doing all the work, with edge size inside that
class just noise?"

Prerequisite: run 28_devigged_edge_breakdown.py first (it writes the two
picks CSVs this reads; must be a version that includes the picked_favorite
column -- re-run 28 if your existing devig_check_*_picks.csv predates it).

Run:
    python scripts/31_devig_class_edge_breakdown.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

import config

RAW_PICKS_CSV = config.PROCESSED_DIR / "devig_check_raw_picks.csv"
DEVIG_PICKS_CSV = config.PROCESSED_DIR / "devig_check_devigged_picks.csv"
N_BUCKETS = config.PICK_TYPE_EDGE_N_BUCKETS  # quintiles, same as 14


def bucket_and_summarize(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    n_actual = min(N_BUCKETS, df["edge"].nunique())
    if len(df) == 0 or n_actual < 2:
        print(f"--- {label} (n={len(df)}) --- not enough distinct edge values to bucket, skipping.\n")
        return pd.DataFrame(), {"group": label, "n": len(df), "spearman_corr": float("nan"), "spearman_p_value": float("nan")}

    df["edge_bucket"] = pd.qcut(df["edge"], n_actual, labels=False, duplicates="drop") + 1
    buckets = (
        df.groupby("edge_bucket")
        .agg(
            n=("correct", "size"),
            min_edge=("edge", "min"),
            max_edge=("edge", "max"),
            accuracy=("correct", "mean"),
            units=("profit", "sum"),
            roi=("profit", "mean"),
        )
        .reset_index()
    )
    buckets.insert(0, "group", label)

    corr, p_value = spearmanr(df["edge"], df["correct"])
    print(f"--- {label} (n={len(df)}) ---")
    print(
        buckets.drop(columns="group").to_string(
            index=False,
            formatters={
                "min_edge": "{:+.1%}".format,
                "max_edge": "{:+.1%}".format,
                "accuracy": "{:.1%}".format,
                "units": "{:+.2f}".format,
                "roi": "{:+.1%}".format,
            },
        )
    )
    print(f"Spearman correlation (edge vs. correctness) within {label}: {corr:+.3f} (p={p_value:.3f})\n")

    return buckets, {"group": label, "n": len(df), "spearman_corr": corr, "spearman_p_value": p_value}


def run_all_classes(picks_df: pd.DataFrame, edge_label: str):
    if "picked_favorite" not in picks_df.columns:
        raise ValueError(
            f"{edge_label}: picks CSV is missing 'picked_favorite' -- re-run "
            f"28_devigged_edge_breakdown.py to regenerate it with that column."
        )

    home = picks_df[picks_df["picked_side"] == "home"]
    away = picks_df[picks_df["picked_side"] == "away"]
    favorite = picks_df[picks_df["picked_favorite"] == True]  # noqa: E712 (explicit bool compare, CSV round-trip)
    underdog = picks_df[picks_df["picked_favorite"] == False]  # noqa: E712

    print(f"\n============================== {edge_label} ==============================")
    all_buckets, all_corrs = [], []
    for label, group_df in [
        ("Home picks", home), ("Away picks", away),
        ("Favorite picks", favorite), ("Underdog picks", underdog),
    ]:
        b, c = bucket_and_summarize(group_df, f"{edge_label} -- {label}")
        if not b.empty:
            all_buckets.append(b)
        all_corrs.append(c)

    for c in all_corrs:
        if pd.isna(c["spearman_p_value"]):
            continue
        verdict = "informative" if c["spearman_p_value"] < 0.05 else "NOT distinguishable from zero"
        print(f"{c['group']}: edge-vs-correctness correlation is {verdict} "
              f"(r={c['spearman_corr']:+.3f}, p={c['spearman_p_value']:.3f}, n={c['n']}).")

    return (pd.concat(all_buckets, ignore_index=True) if all_buckets else pd.DataFrame()), all_corrs


def main():
    if not RAW_PICKS_CSV.exists() or not DEVIG_PICKS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_PICKS_CSV} / {DEVIG_PICKS_CSV} not found. Run "
            f"28_devigged_edge_breakdown.py first -- it writes both files."
        )

    raw_df = pd.read_csv(RAW_PICKS_CSV)
    devig_df = pd.read_csv(DEVIG_PICKS_CSV)

    raw_buckets, raw_corrs = run_all_classes(raw_df, "RAW (vig-included) edge")
    devig_buckets, devig_corrs = run_all_classes(devig_df, "DE-VIGGED edge (corrected)")

    if not raw_buckets.empty:
        raw_buckets["edge_basis"] = "raw"
    if not devig_buckets.empty:
        devig_buckets["edge_basis"] = "devigged"
    combined = pd.concat([b for b in [raw_buckets, devig_buckets] if not b.empty], ignore_index=True)

    out_csv = config.PROCESSED_DIR / "devig_class_edge_breakdown.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")


if __name__ == "__main__":
    main()
