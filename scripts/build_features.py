"""
Shared feature-engineering logic.

IMPORTANT: this module is imported by BOTH the historical training-set
builder (02_build_training_set.py) and, eventually, whatever script scores
the upcoming week live. Keeping the logic in one place is what prevents
train/serve skew -- the same rolling-stat math should run whether you're
building 10 years of history or scoring next Sunday's slate.

Pipeline, conceptually:
  1. Turn team_stats (one row per team per week) into a long "team-game"
     table with two rows per game: the team's own production, and what it
     allowed (its opponent's production against it that week).
  2. For every numeric stat, shift by one game (so "as of kickoff" never
     includes the game being predicted) and compute:
       - a trailing N-game rolling mean
       - an EWMA with a configurable half-life
     Both are computed on each team's continuous game log ACROSS SEASON
     BOUNDARIES. That's deliberate: it's what makes early-season numbers
     blend in last season's performance and taper it out naturally as new
     games accumulate, instead of needing a separate hand-written blending
     formula for weeks 1-4.
  3. Pivot the team-game table back to one row per game with home_/away_
     prefixes, matching the shape of the original 2025 Results.xlsx.
  4. Attach schedule/market columns (spread_line, scores) and labels.
"""

import numpy as np
import pandas as pd

import config


def _pick_team_column(df: pd.DataFrame) -> str:
    for candidate in ("team", "recent_team", "tm"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Could not find a team identifier column in team_stats "
        "(expected 'team', 'recent_team', or 'tm'). Run 01_fetch_historical.py "
        "and check the printed column list."
    )


def _stat_columns(team_stats: pd.DataFrame, team_col: str) -> list[str]:
    """Any numeric column that isn't an ID column is treated as a rollable stat.

    This is deliberately permissive -- the philosophy for this pipeline is
    to bring in every numeric stat nflreadpy/PFR exposes at team-week grain,
    and prune deliberately later (correlation filtering, feature importance,
    hand review) rather than deciding up front what might matter.
    """
    exclude = config.ID_COLUMNS | {team_col}
    return [
        c
        for c in team_stats.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(team_stats[c])
    ]


def merge_extra_sources(
    team_stats: pd.DataFrame, extra_sources: dict[str, pd.DataFrame] | None
) -> pd.DataFrame:
    """Aggregates each extra source (e.g. PFR advstats, which are player-level)
    to (season, week, team) and merges its numeric columns into team_stats,
    prefixed by source name to avoid collisions. Any numeric column that
    survives ends up automatically rolled/EWMA'd downstream -- no extra
    wiring needed in build_team_game_table.
    """
    if not extra_sources:
        return team_stats

    base_team_col = _pick_team_column(team_stats)
    merged = team_stats.copy()

    for source_name, df in extra_sources.items():
        if df is None or df.empty:
            print(f"  (skipping extra source '{source_name}': empty)")
            continue
        try:
            team_col = _pick_team_column(df)
        except KeyError:
            print(f"  (skipping extra source '{source_name}': no team column found)")
            continue

        numeric_cols = [
            c
            for c in df.columns
            if c not in config.ID_COLUMNS
            and c != team_col
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            print(f"  (skipping extra source '{source_name}': no numeric columns found)")
            continue

        agg = (
            df.groupby(["season", "week", team_col])[numeric_cols]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={team_col: base_team_col, **{c: f"{source_name}_{c}" for c in numeric_cols}})
        )
        merged = merged.merge(agg, on=["season", "week", base_team_col], how="left")
        print(f"  merged extra source '{source_name}': +{len(numeric_cols)} columns")

    return merged


def build_team_game_table(schedules: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with that team's own stats ('_produced')
    and its opponent's same-game stats ('_allowed')."""
    team_col = _pick_team_column(team_stats)
    stat_cols = _stat_columns(team_stats, team_col)

    games = schedules.copy()
    if not config.INCLUDE_POSTSEASON and "game_type" in games.columns:
        games = games[games["game_type"] == "REG"]

    # long table: one row per (game, team), regardless of home/away
    home_side = games[["season", "week", "game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home_side["is_home"] = 1
    away_side = games[["season", "week", "game_id", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away_side["is_home"] = 0
    team_game = pd.concat([home_side, away_side], ignore_index=True)

    # attach the team's own production for that week
    ts = team_stats[["season", "week", team_col] + stat_cols].rename(columns={team_col: "team"})
    team_game = team_game.merge(ts, on=["season", "week", "team"], how="left")

    # attach what the opponent produced that same week == what this team allowed
    ts_allowed = ts.rename(columns={c: f"{c}_allowed" for c in stat_cols})
    ts_allowed = ts_allowed.rename(columns={"team": "opponent"})
    team_game = team_game.merge(ts_allowed, on=["season", "week", "opponent"], how="left")

    produced_cols = {c: f"{c}_produced" for c in stat_cols}
    team_game = team_game.rename(columns=produced_cols)

    return team_game.sort_values(["team", "season", "week"]).reset_index(drop=True)


def add_rate_stats(team_game: pd.DataFrame) -> pd.DataFrame:
    """Derives rate stats (EPA/play, yards/attempt, etc.) from raw counting
    stats, computed per game BEFORE rolling -- see config.RATE_STAT_DEFINITIONS
    for why this matters: load_team_stats() only gives totals (e.g. summed
    EPA across every attempt that game), which conflate efficiency with play
    volume/pace. Dividing per game first, then rolling/EWMA-ing the resulting
    rate, is the correct order of operations -- rolling the raw totals and
    dividing afterward would let high-volume games dominate the average.

    New columns are named {name}_produced / {name}_allowed, matching the
    existing suffix convention, so they flow through add_rolling_features,
    pivot_to_game_level, and feature_engineering.py's diff/matchup logic
    automatically -- no changes needed anywhere else.

    Each definition's "pairing" (default "same") controls which suffix the
    DENOMINATOR uses relative to the numerator's suffix:
      - "same": denominator uses the same suffix as the numerator (offense
        stats -- both numbers come from the same team's own game).
      - "cross": denominator uses the OPPOSITE suffix (defense stats -- the
        numerator is this team's own defensive production, but the relevant
        denominator is the opponent's snap count that game).
    """
    new_cols = {}
    added = 0
    skipped = []

    for definition in config.RATE_STAT_DEFINITIONS:
        pairing = definition.get("pairing", "same")
        for suffix in ("produced", "allowed"):
            denom_suffix = suffix if pairing == "same" else ("allowed" if suffix == "produced" else "produced")

            num_cols = [f"{c}_{suffix}" for c in definition["numerator"]]
            den_cols = [f"{c}_{denom_suffix}" for c in definition["denominator"]]
            missing = [c for c in num_cols + den_cols if c not in team_game.columns]
            if missing:
                skipped.append((definition["name"], suffix, missing))
                continue

            numerator = team_game[num_cols].sum(axis=1, min_count=1)
            denominator = team_game[den_cols].sum(axis=1, min_count=1)
            rate = numerator / denominator.replace(0, np.nan)
            # name stays keyed to the numerator's own suffix -- e.g.
            # def_sack_rate_produced -- so downstream suffix-pattern matching
            # (rolling, matchup/diff features) still works normally even
            # though the denominator was pulled from the opposite suffix.
            new_cols[f"{definition['name']}_{suffix}"] = rate
            added += 1

    if skipped:
        print(f"add_rate_stats: skipped {len(skipped)} definition/suffix combos (missing raw columns), e.g. {skipped[:3]}")
    print(f"add_rate_stats: added {added} rate columns")

    return pd.concat([team_game, pd.DataFrame(new_cols, index=team_game.index)], axis=1)


def add_rolling_features(team_game: pd.DataFrame) -> pd.DataFrame:
    """Adds, for every *_produced / *_allowed stat column, a shifted rolling
    mean and EWMA computed on each team's continuous game log.

    With hundreds of stat columns (team_stats + PFR advstats combined), this
    inserts thousands of new columns. Building them into a dict and doing a
    single pd.concat at the end avoids pandas' one-column-at-a-time
    fragmentation warning/slowdown from repeated out[new_col] = ... inserts.
    """
    stat_cols = [c for c in team_game.columns if c.endswith("_produced") or c.endswith("_allowed")]

    grouped = team_game.groupby("team", sort=False)
    new_cols = {}

    for col in stat_cols:
        shifted = grouped[col].shift(1)  # exclude current game -- no leakage
        shifted_grouped = shifted.groupby(team_game["team"])

        new_cols[f"{col}_roll{config.ROLLING_WINDOW_GAMES}"] = shifted_grouped.transform(
            lambda s: s.rolling(window=config.ROLLING_WINDOW_GAMES, min_periods=1).mean()
        )
        new_cols[f"{col}_ewma"] = shifted_grouped.transform(
            lambda s: s.ewm(halflife=config.EWMA_HALFLIFE_GAMES, min_periods=1).mean()
        )

    # base columns minus the raw current-game production/allowed values --
    # those would leak the outcome into the model; only the rolled/EWMA'd
    # (pre-game) versions should ever be used as predictors.
    base = team_game.drop(columns=stat_cols)
    rolled = pd.DataFrame(new_cols, index=team_game.index)

    return pd.concat([base, rolled], axis=1)


def pivot_to_game_level(team_game_features: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one-row-per-team-per-game back to one-row-per-game with
    home_/away_ prefixes, matching the original training file's layout."""
    home = team_game_features[team_game_features["is_home"] == 1].drop(columns=["is_home", "opponent"])
    away = team_game_features[team_game_features["is_home"] == 0].drop(columns=["is_home", "opponent"])

    feature_cols = [c for c in home.columns if c not in ("season", "week", "game_id", "team")]
    home = home.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}})
    away = away.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}})

    merged = home.merge(away, on=["season", "week", "game_id"], how="inner")

    sched_cols = [
        c
        for c in [
            "game_id",
            "home_team",
            "away_team",
            "spread_line",
            "total_line",
            "home_moneyline",
            "away_moneyline",
            "home_spread_odds",
            "away_spread_odds",
            "home_score",
            "away_score",
            "gameday",
            "weekday",
            "gametime",
            "location",
            "div_game",
            "home_rest",
            "away_rest",
            "roof",
            "surface",
            "temp",
            "wind",
        ]
        if c in schedules.columns
    ]
    merged = merged.merge(
        schedules[sched_cols], on=["game_id", "home_team", "away_team"], how="left"
    )

    return merged


def add_labels(game_level: pd.DataFrame) -> pd.DataFrame:
    """Recreates the label/outcome columns from the original training file."""
    df = game_level.copy()
    played = df["home_score"].notna() & df["away_score"].notna()

    df["fav_home"] = (df["spread_line"] > 0).astype("Int64")
    # Magnitude of the spread, independent of which team is favored. A plain
    # linear term on spread_line (as used in 06_train_curated_model.py's
    # logistic regression) assumes covering a 2.5-point spread and a
    # 10.5-point spread differ by the same fixed amount per point -- it
    # can't represent big favorites behaving differently than small ones
    # (e.g. backdoor covers in blowouts). abs_spread_line gives a linear
    # model a direct way to test that, with its own coefficient and p-value,
    # rather than relying on a tree model to maybe find the nonlinearity on
    # its own.
    df["abs_spread_line"] = df["spread_line"].abs()
    df["margin"] = df["home_score"] - df["away_score"]

    df.loc[played, "home_win"] = (df.loc[played, "margin"] > 0).astype(int)
    df.loc[played, "away_win"] = (df.loc[played, "margin"] < 0).astype(int)

    # spread_line convention (nflverse): positive = home favored by that many
    # points. Home covers if margin > spread_line.
    df.loc[played, "home_cover"] = (df.loc[played, "margin"] > df.loc[played, "spread_line"]).astype(int)
    df.loc[played, "away_cover"] = (df.loc[played, "margin"] < df.loc[played, "spread_line"]).astype(int)

    return df


def build_full_dataset(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
    extra_sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    team_stats = merge_extra_sources(team_stats, extra_sources)
    team_game = build_team_game_table(schedules, team_stats)
    team_game = add_rate_stats(team_game)
    team_game_feat = add_rolling_features(team_game)
    game_level = pivot_to_game_level(team_game_feat, schedules)
    return add_labels(game_level)
