"""
Novig NFL spread-odds client -- adapted from the MLB model's
`fetch_betting_odds()` (etl.py in the MLB-Model repo), which hits Novig's
public GraphQL endpoint (no API key required) filtered to
`game: {league: {_eq: "MLB"}}` and parses moneyline + totals markets from
the response. This does the same thing for NFL spread markets instead.

IMPORTANT -- this needs to be verified against a live response before it's
trusted: the MLB client's market-parsing regexes were built by inspecting
real Novig responses (moneyline markets are described by the team
abbreviation alone, e.g. "MIL"; totals markets are described like
"TOR @ MIL t7.5"). Nobody has inspected a real Novig NFL response yet, so
the spread-market regex below (`_SPREAD_RE`) is a best guess by analogy to
the totals pattern, not a confirmed format. Two things to check the first
time this runs against a live NFL slate:

  1. Run with --debug: it prints every market description for every NFL
     event, unfiltered, so you can see the actual spread market naming
     convention and fix `_SPREAD_RE` / `_extract_novig_spread()` if it
     doesn't match what's really there.
  2. Check whether Novig's team abbreviations match nflverse's for every
     NFL team. The MLB client needed `_NOVIG_ABBR_MAP` for exactly this
     (KAN/CWS/WAS didn't match) -- populate `config.NOVIG_TEAM_ABBR_MAP`
     the same way if any NFL mismatches show up in --debug output.

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

_TEAM_RE = re.compile(r'^[A-Z]{2,3}$')
# Best-guess pattern for a spread market, by analogy to the MLB totals
# market's "AWAY @ HOME t7.5" format -- "s" for spread instead of "t" for
# total. UNCONFIRMED against a real NFL response -- see this file's
# docstring. If --debug shows a different pattern, fix this regex (and
# _extract_novig_spread below) to match it.
_SPREAD_RE = re.compile(r'^([A-Z]{2,3}) @ ([A-Z]{2,3}) s([+-]?\d+\.?\d*)$')


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
    unfiltered -- run this FIRST against a live NFL slate to confirm (or
    fix) the spread-market parsing pattern before trusting fetch_current_lines()."""
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


def _extract_novig_spread(markets: list[dict], home_abbr: str, away_abbr: str) -> dict:
    """Pulls the consensus spread line + both sides' American odds from one
    event's markets. Returns {} if nothing matched (unrecognized format --
    run --debug to see why)."""
    for mkt in markets:
        if not mkt.get("is_consensus"):
            continue
        mdesc = (mkt.get("description") or "").strip()
        m = _SPREAD_RE.match(mdesc)
        if not m:
            continue

        outcomes = mkt.get("outcomes") or []
        prices = {}
        for oc in outcomes:
            od = (oc.get("description") or "").strip()
            p = oc.get("available")
            od = config.NOVIG_TEAM_ABBR_MAP.get(od, od)
            if _TEAM_RE.match(od) and p is not None:
                prices[od] = _price_to_american(p)

        if home_abbr in prices and away_abbr in prices:
            spread_num = float(m.group(3))
            # Normalize to nflverse convention: spread_line = points the
            # HOME team is favored by (positive = home favored). Novig's
            # sign convention for this market is UNCONFIRMED -- verify
            # against --debug output and flip the sign here if backwards.
            return {
                "spread_line": spread_num,
                "home_spread_odds": prices[home_abbr],
                "away_spread_odds": prices[away_abbr],
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

        row = {
            "game_id": g["game_id"],
            "home_team": home_abbr,
            "away_team": away_abbr,
            "spread_line": None,
            "home_spread_odds": None,
            "away_spread_odds": None,
        }

        # Novig events are keyed by display name in the MLB client, but this
        # schedule only has abbreviations -- try the abbreviation-based key
        # first (works if Novig's `description` field uses team codes for
        # NFL the way it does for some markets), falling back to a scan for
        # any event description containing both abbreviations.
        novig_key = f"{away_abbr} @ {home_abbr}"
        ev = event_lookup.get(novig_key)
        if ev is None:
            for desc, candidate in event_lookup.items():
                if away_abbr in desc and home_abbr in desc:
                    ev = candidate
                    break

        if ev and ev.get("markets"):
            odds = _extract_novig_spread(ev["markets"], home_abbr, away_abbr)
            if odds:
                row.update(odds)
                matched += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    print(f"Matched {matched} of {len(df)} scheduled games to Novig odds.")
    if matched < len(df):
        print("Unmatched games will have null spread_line/odds -- run with --debug to inspect "
              "why (event description format may not match what this client expects).")
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
