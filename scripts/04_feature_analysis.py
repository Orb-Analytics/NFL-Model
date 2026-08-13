"""
Univariate predictive-power + multicollinearity analysis against the target
(home_cover: 1 if the home team covered spread_line, else 0).

This is deliberately split into two passes rather than one big VIF run:

  1. UNIVARIATE: for every candidate predictor, one at a time, compute its
     point-biserial correlation with home_cover (the correct correlation
     type for a continuous predictor vs. a binary target -- mathematically
     equivalent to a Pearson correlation) plus a p-value and an AUC. This
     tells you which columns have ANY individual signal at all, independent
     of what else is in the model.

  2. MULTICOLLINEARITY, but only among the columns that passed step 1. A
     full pairwise correlation matrix (let alone VIF, which requires
     inverting the design matrix) across all ~2,600 pruned columns would
     mostly surface structural duplication rather than anything useful:
     diff_X = home_X - away_X and matchup_X = (produced + allowed) / 2 are
     exact deterministic functions of the raw produced/allowed columns, not
     independently-arising correlation. Running true VIF against the full
     set would hit exact linear dependencies and either fail (singular
     matrix) or report infinite VIF for large blocks of columns -- not a
     data problem, just how these features were constructed on purpose.
     Narrowing to the top N by individual significance first, then flagging
     high-correlation pairs among THOSE, is the practical stand-in given
     that structure.

Neither pass is the final word -- this is exploratory, to guide what goes
into the actual model. A regularized model (Lasso/Ridge) or a tree-based
model's feature importance, trained next, is what actually accounts for
multiple predictors interacting rather than one at a time.

Run:
    python scripts/04_feature_analysis.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config
from feature_utils import get_predictor_columns


def compute_univariate_stats(df: pd.DataFrame, target: pd.Series, predictor_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in predictor_cols:
        s = df[col]
        mask = s.notna() & target.notna()
        n = mask.sum()

        if n < 30 or s[mask].nunique(dropna=True) <= 1:
            rows.append({"column": col, "n": int(n), "correlation": np.nan, "p_value": np.nan, "auc": np.nan})
            continue

        x = s[mask].to_numpy()
        y = target[mask].to_numpy()

        corr, p_value = stats.pointbiserialr(y, x)

        # AUC via Mann-Whitney U -- rank-based, so it's robust to each
        # feature's own scale (unlike the correlation, this doesn't assume
        # a linear relationship, just a monotonic one).
        pos = x[y == 1]
        neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            auc = np.nan
        else:
            u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            auc = u_stat / (len(pos) * len(neg))

        rows.append({"column": col, "n": int(n), "correlation": corr, "p_value": p_value, "auc": auc})

    result = pd.DataFrame(rows)
    result["abs_correlation"] = result["correlation"].abs()
    return result.sort_values("p_value", na_position="last").reset_index(drop=True)


def find_high_correlation_pairs(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pairwise correlation among the given columns, flagging pairs above
    config.HIGH_CORRELATION_THRESHOLD. Expect a lot of these among diff_X /
    matchup_X / raw produced-allowed columns -- that's the structural
    redundancy described in the module docstring, not a bug in the data.
    """
    corr_matrix = df[columns].corr()
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if pd.notna(r) and abs(r) >= config.HIGH_CORRELATION_THRESHOLD:
                pairs.append({"column_a": cols[i], "column_b": cols[j], "correlation": r})
    if not pairs:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    return pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def main():
    df = pd.read_csv(config.TRAINING_SET_PRUNED_CSV, low_memory=False)

    # only games with a known outcome can be used to assess predictive power
    df = df[df[config.TARGET_COLUMN].notna()].copy()
    target = df[config.TARGET_COLUMN]
    print(f"Analyzing {len(df)} played games against target '{config.TARGET_COLUMN}'")
    print(f"Base rate: home covered {target.mean() * 100:.1f}% of the time")

    predictor_cols = get_predictor_columns(df)
    print(f"{len(predictor_cols)} candidate predictor columns after excluding outcome/ID columns")

    univariate = compute_univariate_stats(df, target, predictor_cols)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    univariate.to_csv(config.UNIVARIATE_STATS_CSV, index=False)
    print(f"  -> {config.UNIVARIATE_STATS_CSV}")

    sig_at_05 = (univariate["p_value"] < 0.05).sum()
    sig_at_01 = (univariate["p_value"] < 0.01).sum()
    print(f"\n{sig_at_05} columns significant at p<0.05, {sig_at_01} at p<0.01 (univariate, uncorrected)")
    print("NOTE: with ~2,600 columns tested individually, some 'significant' results are expected by")
    print("chance alone (multiple-comparisons problem) -- treat this as a ranking/screening tool, not")
    print("proof any single feature matters. A stricter cutoff (e.g. p<0.001) or a correction like")
    print("Benjamini-Hochberg is worth applying before treating this list as final.")

    print("\nTop 15 by |correlation| with home_cover:")
    print(univariate.sort_values("abs_correlation", ascending=False).head(15)[
        ["column", "n", "correlation", "p_value", "auc"]
    ].to_string(index=False))

    top_n_cols = (
        univariate.dropna(subset=["p_value"])
        .sort_values("p_value")
        .head(config.TOP_N_FOR_CORRELATION_CHECK)["column"]
        .tolist()
    )
    high_corr = find_high_correlation_pairs(df, top_n_cols)
    high_corr.to_csv(config.HIGH_CORRELATION_PAIRS_CSV, index=False)
    print(f"\n{len(high_corr)} high-correlation pairs (|r| >= {config.HIGH_CORRELATION_THRESHOLD}) among")
    print(f"the top {len(top_n_cols)} predictors by significance.")
    print(f"  -> {config.HIGH_CORRELATION_PAIRS_CSV}")
    if not high_corr.empty:
        print("\nMost redundant pairs (pick one from each, don't feed both to the model):")
        print(high_corr.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
