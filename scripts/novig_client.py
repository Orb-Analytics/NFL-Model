"""
Novig NFL spread-odds client -- adapted from the MLB model's
`fetch_betting_odds()` (etl.py in the MLB-Model repo), which hits Novig's
public GraphQL endpoint (no API key required) filtered to
`game: {league: {_eq: "MLB"}}` and parses moneyline + totals markets from
the response. This does the same thing for NFL spread markets instead.

CONFIRMED against a real `--debug` dump of live Novig NFL preseason odds
(Aug 2026). The actual formats:

  - Moneyline market: description is the bare team abbreviation, e.g. "PIT".
    Not used here.
  - Totals market: "{AWAY_ABBR} @ {HOME_ABBR} t{NUMBER}", e.g.
    "GB @ PIT t37.5". Not used here.
  - Spread market (what we need): NOT "AWAY @ HOME s-3.5" as originally
    guessed. It's "{HOME_ABBR} {home_team's_own_signed_spread}", e.g.
    "JAX -7.5" (JAX favored by 7.5 as the home team), "NE +2.5" (NE
    underdog by 2.5 as the home team). Checked against every game in a real
    debug dump -- the consensus spread market (is_consensus=True) is
    *always* labeled with the home team's own line, never the away team's.
    Outcomes list both sides by their own signed number, e.g. for
    "JAX -7.5": outcomes=[JAX -7.5=0.53, CLE +7.5=0.54].
  - Some consensus markets have one side's price still `None` (not yet
    posted) -- those games are skipped rather than crashed on.
  - Event `description` uses full team display names, e.g. "Cleveland
    Browns @ Jacksonville Jaguars" -- NOT abbreviations. Event matching
    below is done on full names via NFL_TEAM_FULL_NAMES, not by scanning
    for abbreviation substrings (which would rarely match real display
    names).
  - Team abbreviation mismatch found: Novig uses "WSH" for Washington;
    nflverse uses "WAS". Recorded in config.NOVIG_TEAM_ABBR_MAP, though the
    full-name-based event matching below sidesteps needing it for matching
    purposes -- it's kept for any future abbreviation-keyed lookups.

Since spread_line follows nflverse's convention (positive = home favored)
and Novig's consensus market reports the home team's own signed spread
(negative = home favored), the conversion is a single sign flip:
    spread_line = -1 * (number in the consensus market description)

Output shape (stable regardless of how the parsing internals evolve):
    game_id, home_team, away_team, spread_line, home_spread_odds,
    away_spread_odds, fetched_at
spread_line follows nflverse's convention (positive = home favored), home/
away_spread_odds are American odds ints, matching what pivot_to_game_level
expects in home_spread_odds/away_spread_odds.

Run:
    python scripts/novig_client.py --debug                       # dump raw NFL market descriptions, no parsing
    python scripts/novig_client.py --schedule data/raw/schedules.parquet --out data/processed/live_novig_odds.csv
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

import config

NOVIG_QUERY = """
query {
  event(where: {_and: [
      {_or: [
        {status: {_eq: "OPEN_PREGAME"}},
        {status: {_eq: "OPEN_INGAME"}}
      ]},
      {game: {league: {_eq: "%s"}}}
    ]}) {
    id
    description
    game {
      scheduled_start
    }
    markets {
      description
      is_consensus
      outcomes {
        description
        available
      }
    }
  }
}
""" % config.NOVIG_LEAGUE_FILTER

# Confirmed spread-market format: "{TEAM_ABBR} {signed_number}", e.g.
# "JAX -7.5", "NE +2.5". Requires the signed number so it doesn't also match
# the bare-abbreviation moneyline market description (e.g. "PIT").
_SPREAD_RE = re.compile(r'^([A-Z]{2,3}) ([+-]\d+(?:\.\d+)?)$')

# nflverse team abbreviation -> full display name, used to match Novig's
# event `description` field (which uses full names, e.g. "Cleveland Browns
# @ Jacksonville Jaguars") against our schedule (which only has
# abbreviations). Hardcoded rather than fetched at runtime so matching
# doesn't depend on an extra network call succeeding.
NFL_TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _price_to_american(price):
    """Novig probability (0-1) -> American odds int. Same conversion as the
    MLB client -- Novig's `available` field is a win probability, not
    already-formatted odds."""
    try:
        if price is None:
            return None
        price = float(price)
        if price <= 0 or price >= 1:
            return None
        if price == 0.5:
            return 100
        elif price < 0.5:
            return int(round((1 / price - 1) * 100))
        else:
            return int(round(-100 / (1 / price - 1)))
    except Exception:
        return None


def fetch_raw_events() -> list[dict]:
    resp = requests.post(config.NOVIG_GRAPHQL_URL, json={"query": NOVIG_QUERY}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"Novig GraphQL errors: {data['errors']}")
    return data.get("data", {}).get("event", [])


def debug_dump(events: list[dict]) -> None:
    """Print every market description for every event, completely
    unfiltered -- useful if Novig ever changes its market naming and this
    client needs re-verifying."""
    print(f"{len(events)} {config.NOVIG_LEAGUE_FILTER} events returned.\n")
    for ev in events:
        print(f"--- {ev.get('description')} (id={ev.get('id')}, "
              f"start={ev.get('game', {}).get('scheduled_start')}) ---")
        for mkt in ev.get("markets") or []:
            outcomes = mkt.get("outcomes") or []
            outcome_strs = [f"{o.get('description')}={o.get('available')}" for o in outcomes]
            print(f"  market: {mkt.get('description')!r}  consensus={mkt.get('is_consensus')}  "
                  f"outcomes=[{', '.join(outcome_strs)}]")
        print()


def _extract_novig_spread(markets: list[dict]) -> dict:
    """Pulls the consensus spread line + both sides' American odds from one
    event's markets. Returns {} if nothing matched, or if either side's
    price isn't posted yet.

    The consensus spread market is always labeled with the HOME team's own
    signed spread (e.g. "JAX -7.5"), so once we find it we don't need to
    know which abbreviation Novig uses for which side ahead of time: the
    outcome sharing the market's own team code is home's price, the other
    outcome is away's price.
    """
    for mkt in markets:
        if not mkt.get("is_consensus"):
            continue
        mdesc = (mkt.get("description") or "").strip()
        m = _SPREAD_RE.match(mdesc)
        if not m:
            continue

        mkt_team, mkt_num = m.group(1), m.group(2)
        outcomes = mkt.get("outcomes") or []
        home_price = None
        away_price = None
        for oc in outcomes:
            od = (oc.get("description") or "").strip()
            p = oc.get("available")
            if od.startswith(mkt_team + " "):
                home_price = p
            else:
                away_price = p

        if home_price is None or away_price is None:
            # One side not posted yet -- skip rather than write a
            # half-populated row.
            continue

        home_odds = _price_to_american(home_price)
        away_odds = _price_to_american(away_price)
        if home_odds is None or away_odds is None:
            continue

        return {
            # spread_line follows nflverse's convention (positive = home
            # favored); the market reports home's own signed spread
            # (negative = home favored), so flip the sign once.
            "spread_line": -float(mkt_num),
            "home_spread_odds": home_odds,
            "away_spread_odds": away_odds,
        }
    return {}


def fetch_current_lines(schedule_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Fetch this week's live NFL spread lines/odds from Novig, matched to
    the given schedule (must have game_id, home_team, away_team columns,
    e.g. read from data/raw/schedules.parquet).

    Returns a DataFrame with columns: game_id, home_team, away_team,
    spread_line, home_spread_odds, away_spread_odds, fetched_at. Games with
    no Novig match get NaN odds columns (not dropped) so the caller can see
    match coverage.
    """
    events = fetch_raw_events()
    if debug:
        debug_dump(events)

    # Key events by their full-name description, e.g. "Cleveland Browns @
    # Jacksonville Jaguars" -- confirmed real format, not abbreviations.
    event_lookup = {}
    for ev in events:
        desc = ev.get("description", "")
        n_markets = len(ev.get("markets") or [])
        if desc not in event_lookup or n_markets > len(event_lookup[desc].get("markets") or []):
            event_lookup[desc] = ev

    rows = []
    matched = 0
    for _, g in schedule_df.iterrows():
        home_abbr = str(g["home_team"]).strip()
        away_abbr = str(g["away_team"]).strip()
        home_full = NFL_TEAM_FULL_NAMES.get(home_abbr, home_abbr)
        away_full = NFL_TEAM_FULL_NAMES.get(away_abbr, away_abbr)

        row = {
            "game_id": g["game_id"],
            "home_team": home_abbr,
            "away_team": away_abbr,
            "spread_line": None,
            "home_spread_odds": None,
            "away_spread_odds": None,
        }

        novig_key = f"{away_full} @ {home_full}"
        ev = event_lookup.get(novig_key)
        if ev is None:
            # Fallback in case Novig's naming drifts slightly (extra
            # whitespace, punctuation, etc.) -- scan for both full names
            # appearing in the description rather than an exact match.
            for desc, candidate in event_lookup.items():
                if away_full in desc and home_full in desc:
                    ev = candidate
                    break

        if ev and ev.get("markets"):
            odds = _extract_novig_spread(ev["markets"])
            if odds:
                row.update(odds)
                matched += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    print(f"Matched {matched} of {len(df)} scheduled games to Novig odds.")
    if matched < len(df):
        print("Unmatched games will have null spread_line/odds -- run with --debug to inspect "
              "why (event description format may not match what this client expects, or Novig "
              "hasn't posted lines for those games yet).")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true",
                         help="Dump every NFL event's raw market descriptions and exit -- run this "
                              "first against a live slate to confirm the spread-market format.")
    parser.add_argument("--schedule", default=str(config.RAW_SCHEDULES_PATH),
                         help="Path to schedules.parquet (from 01_fetch_historical.py) to match against.")
    parser.add_argument("--out", default=str(config.LIVE_NOVIG_ODDS_CSV),
                         help="Where to write the matched odds CSV.")
    args = parser.parse_args()

    if args.debug:
        events = fetch_raw_events()
        debug_dump(events)
        return

    schedule_path = Path(args.schedule)
    if not schedule_path.exists():
        raise FileNotFoundError(f"{schedule_path} not found. Run 01_fetch_historical.py first.")
    schedule_df = pd.read_parquet(schedule_path)
    # Scope to games that haven't been played yet -- no point fetching live
    # odds for historical games.
    if "home_score" in schedule_df.columns:
        schedule_df = schedule_df[schedule_df["home_score"].isna()]
    if schedule_df.empty:
        print("No upcoming (unplayed) games found in the schedule -- nothing to fetch odds for.")
        return

    df = fetch_current_lines(schedule_df, debug=args.debug)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
