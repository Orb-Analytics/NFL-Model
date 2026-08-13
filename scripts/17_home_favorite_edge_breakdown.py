"""
Home/away favorite edge breakdown: is the "high-edge favorites do worse"
pattern from 14 actually a HOME-favorite phenomenon specifically, matching
16's model-free finding that the market's tilt away from favorites is
concentrated in home favorites (48.3% cover, z=-1.68) rather than away
favorites (49.8% cover, z=-0.15)?

14_pick_type_edge_breakdown.py pooled all favorite picks (home and away)
together and found a significant negative edge-accuracy relationship
(p=0.027). This splits that same check into home-favorite-only and
away-favorite-only groups, to see whether one side is doing all the work
(consistent with 16's model-free result) or whether the effect is genuinely
shared across both.

Home favorite picks are a much smaller group than away favorite picks, so
fewer buckets are used for that side (config.HOME_FAVORITE_EDGE_N_BUCKETS,
default 3) to keep each bucket's sample large enough to read; away
favorites use more (config.AWAY_FAVORITE_EDGE_N_BUCKETS, default 5, same
as 14 used for all favorites pooled).

Run:
    python scripts/17_home_favorite_edge_breakdown.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from feature_utils import american_odds_to_implied_prob


def bucket_and_summarize(df: pd.DataFrame, label: str, n_buckets: int) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    n_actual_buckets = min(n_buckets, df["edge"].nunique())
    corr, p_value = spearmanr(df["edge"], df["correct"]) if len(df) > 2 else (float("nan"), float("nan"))

    if n_actual_buckets < 2:
        print(f"--- {label} (n={len(df)}) ---")
        print("Not enough distinct edge values to bucket.")
        print(f"Spearman correlation (edge vs. correctness) within {label}: {corr:+.3f} (p={p_value:.3f})\n")
        return pd.DataFrame(), {"group": label, "n": len(df), "spearman_corr": corr, "spearman_p_value": p_value}

    df["edge_bucket"] = pd.qcut(df["edge"], n_actual_buckets, labels=False, duplicates="drop") + 1
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


def main():
    input_path = config.WALK_FORWARD_CONSENSUS_PICKS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run 07_walk_forward_validation.py first.")

    picks = pd.read_csv(input_path)
    required = {"logit_prob", "xgb_prob", "predicted_home_cover", "odds", "spread_line", "correct", "profit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{input_path} is missing column(s) {missing}.")

    before = len(picks)
    picks = picks[picks["odds"].notna()].copy()
    dropped = before - len(picks)
    if dropped:
        print(f"Dropped {dropped} picks with no market price available.\n")

    combined_prob_home = (picks["logit_prob"] + picks["xgb_prob"]) / 2
    is_home = picks["predicted_home_cover"] == 1
    model_prob_for_pick = np.where(is_home, combined_prob_home, 1 - combined_prob_home)
    implied_prob_for_pick = american_odds_to_implied_prob(picks["odds"]).to_numpy()
    picks["edge"] = config.EDGE_MODEL_WEIGHT * (model_prob_for_pick - implied_prob_for_pick)

    picked_em = picks["spread_line"] == 0
    picked_favorite = np.where(is_home, picks["spread_line"] > 0, picks["spread_line"] < 0)
    picks["picked_favorite"] = picked_favorite
    picks["is_home"] = is_home
    picks = picks[~picked_em].copy()

    home_favorites = picks[picks["picked_favorite"] & picks["is_home"]]
    away_favorites = picks[picks["picked_favorite"] & ~picks["is_home"]]

    print(f"Home favorite picks: n={len(home_favorites)}, accuracy={home_favorites['correct'].mean()*100:.1f}%, "
          f"units={home_favorites['profit'].sum():+.2f}, ROI={home_favorites['profit'].mean()*100:+.1f}%")
    print(f"Away favorite picks: n={len(away_favorites)}, accuracy={away_favorites['correct'].mean()*100:.1f}%, "
          f"units={away_favorites['profit'].sum():+.2f}, ROI={away_favorites['profit'].mean()*100:+.1f}%\n")

    all_buckets = []
    all_corrs = []

    b, c = bucket_and_summarize(home_favorites, "home favorite picks", config.HOME_FAVORITE_EDGE_N_BUCKETS)
    all_buckets.append(b)
    all_corrs.append(c)

    b, c = bucket_and_summarize(away_favorites, "away favorite picks", config.AWAY_FAVORITE_EDGE_N_BUCKETS)
    all_buckets.append(b)
    all_corrs.append(c)

    non_empty = [b for b in all_buckets if not b.empty]
    if non_empty:
        combined_df = pd.concat(non_empty, ignore_index=True)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(config.HOME_AWAY_FAVORITE_EDGE_BREAKDOWN_CSV, index=False)

    for c in all_corrs:
        if pd.isna(c["spearman_p_value"]):
            print(f"{c['group']}: not enough data to test (n={c['n']}).")
            continue
        verdict = "informative" if c["spearman_p_value"] < 0.05 else "NOT distinguishable from zero"
        print(f"{c['group']}: edge-vs-correctness correlation is {verdict} "
              f"(r={c['spearman_corr']:+.3f}, p={c['spearman_p_value']:.3f}, n={c['n']}).")

    print(
        "\nIf home favorites show a much stronger (more negative) edge-accuracy relationship than "
        "away favorites, that's consistent with 16's model-free finding that the market's tilt away "
        "from favorites concentrates in home favorites specifically -- and would suggest "
        "15_combined_rule.py's favorite filter could be sharpened further by applying a stricter "
        "edge threshold to home favorites than away favorites, rather than one shared cutoff."
    )

    with open(config.PROCESSED_DIR / "home_away_favorite_edge_metrics.json", "w") as f:
        json.dump(all_corrs, f, indent=2, default=str)

    print(f"\n  -> {config.HOME_AWAY_FAVORITE_EDGE_BREAKDOWN_CSV}")


if __name__ == "__main__":
    main()
