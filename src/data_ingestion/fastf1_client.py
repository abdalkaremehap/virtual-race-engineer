import os

import fastf1
from loguru import logger

# 1. Setup the Cache (CRITICAL to avoid getting blocked by F1 servers)
cache_dir = "data/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


def fetch_driver_telemetry(year: int, grand_prix: str, session_type: str, driver: str):
    logger.info(f"🏎️ Fetching FastF1 data for {year} {grand_prix} ({session_type})...")
    # Get the session object
    session = fastf1.get_session(year, grand_prix, session_type)
    # Load the data (This will take 30-60 seconds on the very first run as it downloads the massive payload)
    logger.info("Downloading laps, telemetry, and weather... please wait.")
    session.load(weather=False, messages=False)

    # Isolate our specific driver's laps
    driver_laps = session.laps.pick_driver(driver)

    # Find their absolute fastest lap of the race
    fastest_lap = driver_laps.pick_fastest()
    logger.success(f"🏁 {driver}'s Fastest Lap Time: {fastest_lap['LapTime']}")

    # Extract the high-frequency telemetry (Speed, Throttle, Brake, Gear) for that single lap
    telemetry = fastest_lap.get_telemetry()
    logger.success(f"📊 Extracted {len(telemetry)} micro-telemetry data points for this single lap alone!")

    return telemetry


if __name__ == "__main__":
    # Let's pull Max Verstappen's (VER) data from the 2024 Bahrain Race ('R')
    tel_df = fetch_driver_telemetry(2024, 'Bahrain', 'R', 'ALO')
    # Print a tiny sample showing the car accelerating through the gears
    logger.info(f"\nSample Telemetry Data:\n{tel_df[['Time', 'Speed', 'Throttle', 'Brake', 'nGear']].head(10)}")