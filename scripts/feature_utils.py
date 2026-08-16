"""
Shared helpers used across the feature-analysis and model-training scripts.

Keeping this logic in one place matters for two reasons that have already
come up: (1) 04_feature_analysis.py and 05_train_baseline_model.py need the
exact same answer to "what counts as a predictor here" -- if that drifted
between scripts, one could call a column predictive while the other
silently excludes it, without either of you noticing; (2) every model
script needs to evaluate against the SAME metrics computed the SAME way, or
comparisons between models (baseline vs. curated) aren't actually
apples-to-apples.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

import config


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column except IDs, outcome/label columns, and anything
    that's an alternate representation of the target (see config.OUTCOME_COLUMNS
    -- these must never be treated as predictors, they'd leak the label).

    Non-numeric columns (roof, surface, location, weekday, gameday, gametime,
    team codes) are excluded here too -- they're either already captured by
    an engineered numeric flag (e.g. is_dome covers what roof/surface would
    tell you) or are pure identifiers with no standalone predictive meaning.
    """
    non_predictor = config.OUTCOME_COLUMNS | {
        "season", "week", "game_id", "home_team", "away_team", "gameday",
        "weekday", "gametime", "location",
    }
    return [
        c
        for c in df.columns
        if c not in non_predictor and pd.api.types.is_numeric_dtype(df[c])
    ]


def american_odds_to_implied_prob(odds: pd.Series | np.ndarray) -> pd.Series:
    """Converts American odds (e.g. -110, +120) to implied win probability.
    Vectorized over a pandas Series; NaN odds produce NaN probability (no
    market price available for that side -- handled by dropping those rows
    before edge calculations, since there's nothing to compare the model to).

    Negative odds (favorite): prob = -odds / (-odds + 100)
    Positive odds (underdog): prob = 100 / (odds + 100)

    Note this is the RAW implied probability, vig included (home + away
    implied probabilities on a standard -110/-110 spread bet sum to ~104.8%,
    not 100%) -- deliberately not de-vigged, since the edge calculation
    compares the model against the actual price you'd bet at, not a
    theoretical fair price.
    """
    odds = pd.Series(odds).astype(float)
    prob = pd.Series(np.nan, index=odds.index)
    negative = odds < 0
    positive = odds >= 0
    prob.loc[negative] = -odds.loc[negative] / (-odds.loc[negative] + 100)
    prob.loc[positive] = 100 / (odds.loc[positive] + 100)
    return prob


def american_odds_to_profit_if_win(odds: pd.Series | np.ndarray) -> pd.Series:
    """Profit per 1 unit staked if the bet wins (e.g. -110 -> ~0.909,
    +120 -> 1.2). Used for ROI, not accuracy -- accuracy doesn't need to
    know the price, but "was this profitable" does."""
    odds = pd.Series(odds).astype(float)
    profit = pd.Series(np.nan, index=odds.index)
    negative = odds < 0
    positive = odds >= 0
    profit.loc[negative] = 100 / (-odds.loc[negative])
    profit.loc[positive] = odds.loc[positive] / 100
    return profit


def compute_edges(model_home_prob: pd.Series, home_implied: pd.Series, away_implied: pd.Series) -> pd.DataFrame:
    """For one model's home-cover probability, computes the home and away
    edge for every game (regressing the model's probability toward the
    market's implied probability per config.EDGE_MODEL_WEIGHT -- see
    config.py's comment above EDGE_MODEL_WEIGHT for the full reasoning) and
    returns the picked side (whichever has the higher edge) + that edge.
    Shared by 08_edge_based_evaluation.py and
    09_consensus_edge_walk_forward.py so both scripts agree on exactly what
    "edge" means.

    NOTE: home_implied/away_implied are used RAW here (vig included, do NOT
    sum to 1) -- this is the ORIGINAL edge definition every backtest
    threshold in this build (15/18/19/20/24's *_EDGE_MAX_THRESHOLD values,
    THREE_WAY_FAVORITE_EDGE_MAX_THRESHOLD) was calibrated against. See
    compute_edges_devigged() below for the corrected version used to
    display the live site's "Win Probability" -- the two are deliberately
    different functions so changing the live DISPLAY math never silently
    changes what these already-calibrated SELECTION thresholds mean.
    """
    model_away_prob = 1 - model_home_prob
    home_edge = config.EDGE_MODEL_WEIGHT * (model_home_prob - home_implied)
    away_edge = config.EDGE_MODEL_WEIGHT * (model_away_prob - away_implied)

    pick_home = home_edge >= away_edge
    picked_side = np.where(pick_home, "home", "away")
    picked_edge = np.where(pick_home, home_edge, away_edge)
    return pd.DataFrame({"picked_side": picked_side, "picked_edge": picked_edge}, index=model_home_prob.index)


def compute_edges_devigged(model_home_prob: pd.Series, home_implied: pd.Series, away_implied: pd.Series) -> pd.DataFrame:
    """Same as compute_edges(), except home_implied/away_implied are
    de-vigged (normalized to sum to 1) BEFORE blending with the model's
    probability. This is what orb-analytics-web's predictions.html now
    displays as "Win Probability"/"Market Implied"/"Edge" for live NFL
    picks (see 26_write_predictions_json.py) -- home_cover_prob +
    away_cover_prob = 1 exactly, matching what the raw model output
    already guarantees on its own, unlike compute_edges() above (raw
    vig-included implied probabilities sum to ~103-107%, not 100%, so a
    home-side and away-side blend built from them independently do NOT sum
    to 1).

    Returns picked_side/picked_edge (same shape as compute_edges) PLUS
    picked_confidence -- the final de-vigged blended cover probability for
    the picked side (== market_implied_devigged + picked_edge, i.e. what
    the live site calls "Win Probability").

    Deliberately a SEPARATE function from compute_edges(), not a flag on
    it -- every backtested edge threshold in this build (15/18/19/20/24)
    was calibrated against the RAW-vig definition. Use this function only
    for evaluating whether the CORRECTED probability/edge tracks accuracy
    -- do not use it to re-gate any of those already-calibrated selection
    rules without re-deriving their thresholds from scratch.
    """
    model_away_prob = 1 - model_home_prob
    vig_sum = home_implied + away_implied
    home_implied_devig = home_implied / vig_sum
    away_implied_devig = away_implied / vig_sum

    home_edge = config.EDGE_MODEL_WEIGHT * (model_home_prob - home_implied_devig)
    away_edge = config.EDGE_MODEL_WEIGHT * (model_away_prob - away_implied_devig)

    pick_home = home_edge >= away_edge
    picked_side = np.where(pick_home, "home", "away")
    picked_edge = np.where(pick_home, home_edge, away_edge)
    picked_market_implied = np.where(pick_home, home_implied_devig, away_implied_devig)
    picked_confidence = picked_market_implied + picked_edge
    return pd.DataFrame(
        {
            "picked_side": picked_side,
            "picked_edge": picked_edge,
            "picked_market_implied": picked_market_implied,
            "picked_confidence": picked_confidence,
        },
        index=model_home_prob.index,
    )


def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray, label: str) -> dict:
    """Standard metric set for every model/baseline comparison in this
    pipeline: accuracy, AUC, log loss, and Brier score (a calibration
    check -- important since the model's actual output is a probability,
    not just a pick, per the classification approach this was built for).
    """
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    return {
        "label": label,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred_class),
        "auc": roc_auc_score(y_true, y_pred_proba) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, y_pred_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }
