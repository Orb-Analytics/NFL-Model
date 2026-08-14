"""
Builds predictions_history.json from the historical spread-picks CSV, matching
the real shape of Orb-Analytics/MLB-Model/main/predictions_history.json as
closely as the source data allows.

The source CSV (data/raw/historical_spread_picks.csv) has no edge, confidence,
date, or box-score fields -- only season/week, team nicknames, the posted
spread, the pick, its odds, and the result. Those unavailable fields are
simply omitted per instructions ("edge is missing, that's ok, we can build it
without it for now"); site-side code in predictions.html has matching
guards to hide the Edge/Conf columns and splits when a sport doesn't
report them (see the "activeSport !== 'nfl'" checks added alongside this
script).

Date is not present in the source data (only season + week), so each pick is
assigned the Sunday of its NFL week as a reasonable stand-in -- this is exact
for the majority of games (the Sunday early/late slate) and off by at most a
few days for Thu/Mon games, which is fine for sorting/chart purposes.
"""
import json
from datetime import date, timedelta

import pandas as pd

import config

NICKNAME_TO_ABBR = {
    "49ers": "SF", "49Ers": "SF",
    "Bears": "CHI", "Bengals": "CIN", "Bills": "BUF", "Broncos": "DEN",
    "Browns": "CLE", "Buccaneers": "TB", "Cardinals": "ARI", "Chargers": "LAC",
    "Chiefs": "KC", "Colts": "IND", "Commanders": "WAS", "Cowboys": "DAL",
    "Dolphins": "MIA", "Eagles": "PHI", "Falcons": "ATL", "Giants": "NYG",
    "Jaguars": "JAX", "Jets": "NYJ", "Lions": "DET", "Packers": "GB",
    "Panthers": "CAR", "Patriots": "NE", "Raiders": "LV", "Rams": "LAR",
    "Ravens": "BAL", "Saints": "NO", "Seahawks": "SEA", "Steelers": "PIT",
    "Texans": "HOU", "Titans": "TEN", "Vikings": "MIN",
}

# Sunday of Week 1 for each season on record. Games run Thu-Mon; Sunday is
# used as the representative date for every week since we don't have the
# actual day-of-week per game in the source data.
WEEK1_SUNDAY = {
    2023: date(2023, 9, 10),
    2024: date(2024, 9, 8),
    2025: date(2025, 9, 7),
}

SOURCE_CSV = config.ROOT_DIR / "data" / "raw" / "historical_spread_picks.csv"
OUTPUT_JSON = config.ROOT_DIR / "predictions_history.json"


def _season_year(season_str: str) -> int:
    # "2023-24" -> 2023; also covers the "2023-25" typo row (same season).
    return int(str(season_str).split("-")[0])


def _week_date(season_year: int, week: int) -> str:
    base = WEEK1_SUNDAY[season_year]
    d = base + timedelta(weeks=week - 1)
    return f"{d.month}/{d.day}/{d.year}"


def _result_label(pick_result: float) -> str:
    # Pick Result is 1 = win, 0 = loss, 0.5 = push (confirmed against the
    # real data -- 7 of 264 rows are 0.5, not just 0/1 as first assumed).
    if pick_result == 1:
        return "Win"
    if pick_result == 0.5:
        return "Push"
    return "Loss"


def _units(result: str, odds: int) -> float:
    if result == "Push":
        return 0.0
    if result == "Loss":
        return -1.0
    return round(odds / 100.0, 4) if odds > 0 else round(100.0 / abs(odds), 4)


def main():
    df = pd.read_csv(SOURCE_CSV)

    picks = []
    for _, row in df.iterrows():
        season_year = _season_year(row["season"])
        home_abbr = NICKNAME_TO_ABBR[row["home_team"]]
        away_abbr = NICKNAME_TO_ABBR[row["away_team"]]
        pick_abbr = NICKNAME_TO_ABBR[row["Spread Pick"]]
        picked_home = bool(row["Pick Home?"])
        home_spread = float(row["spread"])
        picked_spread = home_spread if picked_home else -home_spread
        odds = int(row["Odds"])
        result = _result_label(row["Pick Result"])

        picks.append({
            "date": _week_date(season_year, int(row["week"])),
            "season": season_year,
            "week": int(row["week"]),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "pick": pick_abbr,
            "spread": round(picked_spread, 1),
            "confidence": None,
            "line": odds,
            "edge": None,
            "result": result,
            "units": _units(result, odds),
            "home_score": None,
            "away_score": None,
        })

    wins = sum(1 for p in picks if p["result"] == "Win")
    losses = sum(1 for p in picks if p["result"] == "Loss")
    pushes = sum(1 for p in picks if p["result"] == "Push")
    total_units = round(sum(p["units"] for p in picks), 2)
    total_picks = len(picks)
    # Win rate excludes pushes from the denominator (standard convention --
    # a push is a non-event, not a loss).
    win_rate = round((wins / (wins + losses)) * 100, 1) if (wins + losses) else 0

    output = {
        "model": "NFL",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "version": config.MODEL_VERSION,
        "summary": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "total_units": total_units,
            "total_picks": total_picks,
        },
        "picks": picks,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {total_picks} historical picks ({wins}-{losses}-{pushes}, {total_units:+.2f}u) to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
