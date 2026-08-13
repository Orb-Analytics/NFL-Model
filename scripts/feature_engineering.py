"""
Derived/engineered features layered on top of the raw rolled predictors from
build_features.py.

This is the extension point for feature engineering. build_features.py's job
is to maximize raw predictor intake (every numeric stat nflreadpy/PFR expose,
rolled and EWMA'd); this module's job is to combine those raw predictors into
things that are often more directly useful to a classifier than the raw
levels are on their own. Add new functions here following the same pattern:
take the game-level dataframe, return it with more columns added, and don't
mutate columns you didn't create.

Every function here is opt-in via config.py flags so you can turn any of
them off without editing code -- useful when you get to pruning and want to
A/B whether a whole feature family is pulling its weight.
"""

import re

import pandas as pd

import config


def _rolled_feature_stems(columns: list[str]) -> list[str]:
    """Given a list of home_/away_-prefixed columns, return the stems shared
    between the home and away side that are actually rolled predictors
    (e.g. 'passing_yards_produced_ewma') -- NOT labels, scores, or raw
    market columns, which would otherwise sneak in as "home_X - away_X" and
    leak the outcome (e.g. home_cover - away_cover) or add noise.
    """
    home_cols = {c[len("home_") :] for c in columns if c.startswith("home_")}
    away_cols = {c[len("away_") :] for c in columns if c.startswith("away_")}
    stems = home_cols & away_cols
    # matchup included alongside produced/allowed so diff_X_matchup_ewma gets
    # built automatically wherever add_matchup_features() has already run
    # (see engineer_all() -- matchup runs BEFORE differential specifically so
    # this can pick up matchup columns the same call).
    allowed_suffix = re.compile(r"_(produced|allowed|matchup)_(ewma|roll\d+)$")
    return sorted(s for s in stems if allowed_suffix.search(s))


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """diff_X = home_X - away_X for every rolled home/away feature pair.

    Rationale: for a spread classifier, what usually matters is the gap
    between the two teams, not each team's absolute level. Handing the model
    the differential directly saves it from having to (re)learn that
    subtraction itself, which matters more for linear models but rarely hurts
    tree-based ones either.
    """
    if not config.ADD_DIFFERENTIAL_FEATURES:
        return df

    stems = _rolled_feature_stems(list(df.columns))
    new_cols = {}
    for stem in stems:
        home_col, away_col = f"home_{stem}", f"away_{stem}"
        if pd.api.types.is_numeric_dtype(df[home_col]) and pd.api.types.is_numeric_dtype(df[away_col]):
            new_cols[f"diff_{stem}"] = df[home_col] - df[away_col]

    print(f"add_differential_features: added {len(new_cols)} columns")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """For every stat with both a produced and allowed rolled version, pair a
    team's offensive tendency against the opponent's tendency to allow that
    stat. E.g. home_passing_yards_matchup_ewma blends home's passing-yards-
    produced EWMA with away's passing-yards-allowed EWMA -- a rough estimate
    of what should happen when this specific matchup occurs, rather than
    treating each side's numbers in isolation.
    """
    if not config.ADD_MATCHUP_FEATURES:
        return df

    columns = list(df.columns)
    new_cols = {}

    # find stat base names that have a _produced_<suffix> and _allowed_<suffix> variant
    produced_pattern = re.compile(r"^home_(.+)_produced_(ewma|roll\d+)$")
    for col in columns:
        m = produced_pattern.match(col)
        if not m:
            continue
        stat, suffix = m.group(1), m.group(2)

        home_produced = f"home_{stat}_produced_{suffix}"
        away_allowed = f"away_{stat}_allowed_{suffix}"
        away_produced = f"away_{stat}_produced_{suffix}"
        home_allowed = f"home_{stat}_allowed_{suffix}"

        if all(c in columns for c in [home_produced, away_allowed, away_produced, home_allowed]):
            new_cols[f"home_{stat}_matchup_{suffix}"] = (df[home_produced] + df[away_allowed]) / 2
            new_cols[f"away_{stat}_matchup_{suffix}"] = (df[away_produced] + df[home_allowed]) / 2

    print(f"add_matchup_features: added {len(new_cols)} columns")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_schedule_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Differential/derived versions of the schedule-level context columns
    (rest days, weather) that build_features.py now carries through but
    doesn't otherwise transform."""
    if not config.ADD_SCHEDULE_CONTEXT_FEATURES:
        return df

    new_cols = {}
    if {"home_rest", "away_rest"}.issubset(df.columns):
        new_cols["rest_diff"] = df["home_rest"] - df["away_rest"]
        new_cols["home_short_week"] = (df["home_rest"] < 6).astype("Int64")
        new_cols["away_short_week"] = (df["away_rest"] < 6).astype("Int64")
    if "roof" in df.columns:
        new_cols["is_dome"] = df["roof"].isin(["dome", "closed"]).astype("Int64")

    print(f"add_schedule_context_features: added {len(new_cols)} columns")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Lists every column that's usable as a model predictor (i.e. not an ID,
    label, or raw market/outcome column), with basic stats to speed up
    pruning: % non-null and whether it's constant.

    This is the map you'll work from when deciding what to cut -- run this
    after every pipeline change to see how the predictor count moved.
    """
    non_predictor = {
        "season", "week", "game_id", "home_team", "away_team", "gameday",
        "weekday", "gametime", "location", "home_score", "away_score",
        "margin", "home_win", "away_win", "home_cover", "away_cover",
    }
    predictor_cols = [c for c in df.columns if c not in non_predictor]

    rows = []
    for c in predictor_cols:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "pct_non_null": round(s.notna().mean() * 100, 1),
            "is_constant": s.nunique(dropna=True) <= 1,
        })
    manifest = pd.DataFrame(rows).sort_values("column").reset_index(drop=True)
    print(f"build_feature_manifest: {len(manifest)} candidate predictor columns")
    return manifest


def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    """Runs every enabled feature-engineering step in sequence.

    Matchup runs BEFORE differential, deliberately: add_differential_features
    diffs whatever home_X/away_X columns already exist in df at the time it
    runs, and diff_X_matchup_ewma (home offense-vs-away-defense minus away
    offense-vs-home-defense) is only possible once add_matchup_features has
    already created home_X_matchup_ewma / away_X_matchup_ewma. Reversing this
    order would silently produce zero matchup differentials with no error.
    """
    df = add_matchup_features(df)
    df = add_differential_features(df)
    df = add_schedule_context_features(df)
    return df
