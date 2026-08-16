"""
Write predictions.json from this week's picks (25_live_weekly_scoring.py's
output).

IMPORTANT: this matches orb-analytics-web's REAL, LIVE schema -- confirmed
by reading Orb-Analytics/MLB-Model/main/predictions.json directly and the
actual JS in orb-analytics-web/predictions.html that consumes it -- NOT
the schema documented in orb-analytics-web's README, which turned out to
be stale/aspirational (no `has_pick` field, full team names instead of
abbreviations, `edge` as a decimal instead of a percentage). Building
against the README's example would have produced a file the real site code
can't actually render. The real MLB shape:

    {
      "model": "MLB",
      "generated_at": "2026-07-21T04:44:09.317689+00:00",
      "version": "v2.1",
      "picks": [
        {
          "home_team": "PHI", "away_team": "LAD", "pick": "LAD",
          "has_pick": true, "confidence": 0.4678,
          "home_odds": -127, "away_odds": 125, "line": 125,
          "edge": 2.34, "notes": "XGBoost edge: +2.34%"
        }
      ]
    }

This writer produces the NFL equivalent, with one addition the site's JS
also expects: a `spread` field. MLB's single "line" field does double duty
as both the display number AND the American-odds input to the page's
implied-probability formula, because a moneyline number works fine as
both. A spread sport doesn't have that luxury -- the spread itself (e.g.
-3.5, what a bettor cares about) and that side's American odds at that
spread (e.g. -110, what implied probability is actually computed from) are
two different numbers. `line` here is the picked side's own odds (parallel
to MLB's "line"); `spread` is the picked side's own spread number.

    {
      "model": "NFL",
      "generated_at": "2026-09-09T13:00:00Z",
      "version": "v2.0-2model-devigged-asymmetric-edge",
      "picks": [
        {
          "game_id": "...", "home_team": "KC", "away_team": "BUF", "pick": "KC",
          "has_pick": true, "confidence": 0.58, "spread": -3.5,
          "home_odds": -110, "away_odds": -110, "line": -110,
          "edge": 0.34, "notes": "2-model avg (logit+xgb), de-vigged edge +0.34% (favorite)"
        }
      ]
    }

The predictions page reads this file directly from
Orb-Analytics/NFL-Model's main branch -- committing an updated
predictions.json to main is the entire integration; no deploy or API call
needed on the site side.

"home_team"/"away_team"/"pick" are nflverse's own team abbreviations
(matching MLB's real behavior of using abbreviations, not full names --
the site's JS checks `pick.pick === pick.home_team` for exact string
equality, so these must match). "confidence" = the de-vigged, market-
regressed cover probability for the picked side (25_live_weekly_scoring.py's
"confidence" column, from feature_utils.compute_edges_devigged -- the
2-model logit+xgb average blended against the de-vigged market price).
"edge" is written as a PERCENTAGE NUMBER (2.34 = 2.34%), matching MLB's
real convention -- our internal edge is a decimal fraction (e.g. 0.0034),
so it's multiplied by 100 here.

Run:
    python scripts/26_write_predictions_json.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config


def _picked_team_spread_display(spread_line: float, picked_home: bool) -> float:
    """Convert nflverse's home-perspective spread_line (positive = home
    favored) to the PICKED team's own posted spread (favorite always
    negative, matching standard sportsbook display)."""
    return -spread_line if picked_home else spread_line


def _int_or_none(val):
    return int(val) if pd.notna(val) else None


def main():
    if not config.LIVE_PICKS_CSV.exists():
        raise FileNotFoundError(f"{config.LIVE_PICKS_CSV} not found. Run 25_live_weekly_scoring.py first.")

    picks_df = pd.read_csv(config.LIVE_PICKS_CSV)
    if picks_df.empty:
        print("No picks this week -- writing predictions.json with an empty picks list "
              "(so the site correctly shows 'no picks' rather than stale data).")

    has_home_away_odds = "home_spread_odds" in picks_df.columns and "away_spread_odds" in picks_df.columns
    if not has_home_away_odds:
        print("WARNING: live_picks.csv doesn't have home_spread_odds/away_spread_odds columns -- "
              "writing home_odds/away_odds as null (only the picked side's own odds, in 'line', "
              "will be populated). Check 25_live_weekly_scoring.py's scoring DataFrame if this is "
              "unexpected -- these columns should carry through from schedules via build_full_dataset.")

    has_market_implied = "market_implied_prob_devigged" in picks_df.columns
    has_confidence = "confidence" in picks_df.columns

    picks = []
    for _, row in picks_df.iterrows():
        picked_home = row["predicted_home_cover"] == 1
        picked_abbr = row["home_team"] if picked_home else row["away_team"]

        spread_display = _picked_team_spread_display(row["spread_line"], picked_home)

        # v2 rule (25_live_weekly_scoring.py): "confidence" and "edge" are
        # already the final, de-vigged numbers computed once by
        # feature_utils.compute_edges_devigged -- the SAME basis used to
        # decide whether this game got picked in the first place. Unlike
        # the old v1 rule, there is no longer a separate raw-vig selection
        # edge and de-vigged display edge to reconcile; this is read
        # straight through.
        if has_confidence and has_market_implied:
            confidence = float(row["confidence"])
            market_implied = float(row["market_implied_prob_devigged"])
            display_edge = float(row["edge"])
        else:
            # Fallback for older live_picks.csv files written before the
            # v2 columns existed -- avg_prob_3 was the old 3-model average.
            market_implied = None
            model_prob_for_pick = float(row["avg_prob_3"]) if picked_home else float(1 - row["avg_prob_3"])
            confidence = model_prob_for_pick
            display_edge = float(row["edge"])

        notes = (
            f"2-model avg (logit+xgb), de-vigged edge {row['edge']*100:+.2f}% "
            f"({'favorite' if row['picked_favorite'] else 'underdog'})"
        )

        picks.append({
            "game_id": str(row["game_id"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "pick": picked_abbr,
            "has_pick": True,
            "confidence": round(confidence, 4),
            "spread": round(float(spread_display), 1),
            "home_odds": _int_or_none(row["home_spread_odds"]) if has_home_away_odds else None,
            "away_odds": _int_or_none(row["away_spread_odds"]) if has_home_away_odds else None,
            "line": _int_or_none(row["picked_odds"]),
            # De-vigged market-implied probability for the picked side, so
            # the site can show a "Market Implied" number consistent with
            # confidence/edge instead of recomputing a raw (vig-included)
            # one client-side from the odds (which would reintroduce the
            # same inconsistency this whole change fixes).
            "market_implied": round(market_implied, 4) if market_implied is not None else None,
            "edge": round(display_edge * 100, 2),
            # Whether the PICKED side is the game's actual favorite, derived
            # from spread_line (not from the sign of the odds -- both sides
            # of a spread are usually priced around -110 regardless of who's
            # favored, so odds sign is not a reliable favorite/underdog
            # signal the way it is for a moneyline sport). The site's
            # isFavoritePick()/isUnderdogPick() helpers read this field
            # directly instead of falling back to odds-sign for NFL.
            "is_favorite": bool(row["picked_favorite"]),
            "notes": notes,
        })

    output = {
        "model": "NFL",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": config.MODEL_VERSION,
        "picks": picks,
    }

    with open(config.PREDICTIONS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(picks)} picks to {config.PREDICTIONS_JSON}")
    print(f"\nCommit and push this file to Orb-Analytics/NFL-Model's main branch to publish it:")
    print(f"  git add {config.PREDICTIONS_JSON.name}")
    print(f"  git commit -m \"Week's picks: {output['generated_at']}\"")
    print(f"  git push")


if __name__ == "__main__":
    main()
