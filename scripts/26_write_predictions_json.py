"""
Write predictions.json from this week's picks (25_live_weekly_scoring.py's
output), matching the exact schema orb-analytics-web expects
(Orb-Analytics/orb-analytics-web README, "Predictions Integration"):

    {
      "model": "NFL",
      "generated_at": "2026-07-15T08:00:00Z",
      "version": "v1.0-3way-consensus-low-edge-favorite",
      "picks": [
        {
          "game_id": "...",
          "home_team": "Kansas City Chiefs",
          "away_team": "Buffalo Bills",
          "pick": "Chiefs -3.5",
          "confidence": 0.58,
          "line": -110,
          "notes": "..."
        }
      ]
    }

The predictions page reads this file directly from
Orb-Analytics/NFL-Model's main branch -- committing an updated
predictions.json to main is the entire integration; no deploy or API call
needed on the site side.

"pick" = the picked team's short/nickname + their own posted spread
(favorite always negative, matching standard sportsbook display -- NOT
nflverse's spread_line convention directly, which is home-perspective and
needs a sign flip for away picks; see _picked_team_spread_display below).
"confidence" = the 3-model average probability for the picked side (same
avg_prob_3 used to compute edge in 25).

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

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None


def _load_team_names() -> dict:
    """abbr -> {"full": ..., "short": ...}. Falls back to using the
    abbreviation itself for both if nflreadpy's team table isn't available
    or doesn't have the expected columns -- this should never hard-fail the
    whole run just because a display name lookup didn't work."""
    if nfl is None:
        return {}
    try:
        teams = nfl.load_teams().to_pandas()
    except Exception as e:
        print(f"Could not load team names from nflreadpy ({e}) -- falling back to abbreviations.")
        return {}

    abbr_col = next((c for c in ["team_abbr", "abbr", "team"] if c in teams.columns), None)
    full_col = next((c for c in ["team_name", "full_name", "display_name"] if c in teams.columns), None)
    short_col = next((c for c in ["team_nick", "nickname", "short_name"] if c in teams.columns), None)
    if not abbr_col:
        return {}

    mapping = {}
    for _, row in teams.iterrows():
        abbr = row[abbr_col]
        full = row[full_col] if full_col else abbr
        short = row[short_col] if short_col else full
        mapping[abbr] = {"full": full, "short": short}
    return mapping


def _picked_team_spread_display(spread_line: float, picked_home: bool) -> float:
    """Convert nflverse's home-perspective spread_line (positive = home
    favored) to the PICKED team's own posted spread (favorite always
    negative, matching standard sportsbook display)."""
    return -spread_line if picked_home else spread_line


def main():
    if not config.LIVE_PICKS_CSV.exists():
        raise FileNotFoundError(f"{config.LIVE_PICKS_CSV} not found. Run 25_live_weekly_scoring.py first.")

    picks_df = pd.read_csv(config.LIVE_PICKS_CSV)
    if picks_df.empty:
        print("No picks this week -- writing predictions.json with an empty picks list "
              "(so the site correctly shows 'no picks' rather than stale data).")

    team_names = _load_team_names()

    picks = []
    for _, row in picks_df.iterrows():
        picked_home = row["predicted_home_cover"] == 1
        picked_abbr = row["home_team"] if picked_home else row["away_team"]
        opp_abbr = row["away_team"] if picked_home else row["home_team"]

        home_display = team_names.get(row["home_team"], {}).get("full", row["home_team"])
        away_display = team_names.get(row["away_team"], {}).get("full", row["away_team"])
        picked_short = team_names.get(picked_abbr, {}).get("short", picked_abbr)

        spread_display = _picked_team_spread_display(row["spread_line"], picked_home)
        pick_str = f"{picked_short} {spread_display:+.1f}"

        confidence = float(row["avg_prob_3"]) if picked_home else float(1 - row["avg_prob_3"])

        notes = (
            f"3-way consensus, edge {row['edge']*100:+.1f}% "
            f"({'favorite' if row['picked_favorite'] else 'underdog'})"
        )

        picks.append({
            "game_id": str(row["game_id"]),
            "home_team": home_display,
            "away_team": away_display,
            "pick": pick_str,
            "confidence": round(confidence, 4),
            "line": int(row["picked_odds"]) if pd.notna(row["picked_odds"]) else None,
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
