import os

import duckdb
import httpx
import pandas as pd
from loguru import logger

os.makedirs("data", exist_ok=True)

def fetch_2024_calendar():
    logger.info("📡 Fetching 2024 F1 Calendar from Jolpica API...")
    url = "https://api.jolpi.ca/ergast/f1/2024/races.json"
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

    races = data["MRData"]["RaceTable"]["Races"]
    clean_races = []
    for race in races:
        clean_races.append({
            "season": int(race["season"]),
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "date": race["date"]
        })

    df = pd.DataFrame(clean_races)
    logger.success(f"🏁 Successfully fetched {len(df)} races!")
    return df


def save_to_duckdb(df):
    db_path = "data/f1_historical.duckdb"
    logger.info(f"💾 Saving to DuckDB at {db_path}...")

    # Open connection safely (releases the lock immediately when done)
    with duckdb.connect(db_path) as con:
        # 1. Create the table ONLY if it's missing (No more bulldozing)
        con.execute("CREATE TABLE IF NOT EXISTS race_calendar AS SELECT * FROM df LIMIT 0")

        # 2. Add a Unique Index so we don't insert duplicate races
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_race ON race_calendar(season, round)")

        # 3. Safely insert the new data (ignoring duplicates)
        con.register("stage_df", df)
        con.execute("INSERT OR IGNORE INTO race_calendar SELECT * FROM stage_df")

        result = con.execute("SELECT COUNT(*) FROM race_calendar").fetchall()
        logger.success(f"✅ Verified {result[0][0]} total rows safely stored in DuckDB!")


if __name__ == "__main__":
    df_calendar = fetch_2024_calendar()
    save_to_duckdb(df_calendar)