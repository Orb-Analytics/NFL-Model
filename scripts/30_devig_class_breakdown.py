"""
Breaks down 28_devigged_edge_breakdown.py's picks (2-model average, no
consensus gate, side with the higher edge picked on every game) by
Home vs Away and Favorite vs Underdog -- accuracy/units/ROI for each class,
raw-vig edge vs. de-vigged edge side by side.

Prerequisite: run 28_devigged_edge_breakdown.py first (it writes the two
picks CSVs this reads, now including picked_favorite -- re-run 28 if your
existing devig_check_*_picks.csv predates that column).

Run:
    python scripts/30_devig_class_breakdown.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from feature_utils import american_odds_to_profit_if_win

RAW_PICKS_CSV = config.PROCESSED_DIR / "devig_check_raw_picks.csv"
DEVIG_PICKS_CSV = config.PROCESSED_DIR / "devig_check_devigged_picks.csv"


def class_table(picks_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "picked_favorite" not in picks_df.columns:
        raise ValueError(
            f"{label}: picks CSV is missing 'picked_favorite' -- re-run "
            f"28_devigged_edge_breakdown.py to regenerate it with that column."
        )

    picks_df = picks_df.copy()
    picks_df["home_away"] = picks_df["picked_side"].map({"home": "Home", "away": "Away"})
    picks_df["fav_dog"] = picks_df["picked_favorite"].map({True: "Favorite", False: "Underdog"})

    if "profit" not in picks_df.columns:
        picks_df["profit_if_win"] = american_odds_to_profit_if_win(picks_df["odds"])
        picks_df["profit"] = picks_df["correct"] * picks_df["profit_if_win"] - (1 - picks_df["correct"])

    rows = []
    for dim_name, dim_col in [("Home vs Away", "home_away"), ("Favorite vs Underdog", "fav_dog")]:
        for cls, g in picks_df.groupby(dim_col):
            n = len(g)
            accuracy = g["correct"].mean()
            se = (accuracy * (1 - accuracy) / n) ** 0.5 if n > 0 else float("nan")
            z = (accuracy - 0.5) / se if se > 0 else float("nan")
            units = g["profit"].sum()
            roi = g["profit"].mean()
            rows.append({
                "dimension": dim_name, "class": cls, "n": n,
                "accuracy": accuracy, "z_vs_50pct": z, "units": units, "roi": roi,
            })
    result = pd.DataFrame(rows)

    print(f"\n--- {label} ---")
    print(result.to_string(
        index=False,
        formatters={
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

    raw_result = class_table(raw_df, "RAW (vig-included) edge -- picks by class")
    devig_result = class_table(devig_df, "DE-VIGGED edge (corrected) -- picks by class")

    out_csv = config.PROCESSED_DIR / "devig_class_breakdown.csv"
    raw_result["edge_basis"] = "raw"
    devig_result["edge_basis"] = "devigged"
    pd.concat([raw_result, devig_result], ignore_index=True).to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")


if __name__ == "__main__":
    main()
