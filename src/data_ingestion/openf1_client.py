import httpx
import pandas as pd
from loguru import logger


def get_openf1_session_key(year: int, country: str, session_name: str) -> int:
    url = "https://api.openf1.org/v1/sessions"
    # Using 'params' automatically fixes spaces (e.g. Great Britain -> Great%20Britain)
    params = {"year": year, "country_name": country, "session_name": session_name}

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    if not data:
        raise ValueError("Session not found in OpenF1!")

    session_key = data[0]["session_key"]
    logger.success(f"🔑 Found Session Key: {session_key}")
    return session_key


def fetch_race_control_messages(session_key: int):
    logger.info(f"🚩 Fetching FIA Race Control messages for session {session_key}...")
    url = "https://api.openf1.org/v1/race_control"

    with httpx.Client() as client:
        response = client.get(url, params={"session_key": session_key})
        # If API gives a 404 or empty response, return empty table safely
        if response.status_code == 404 or not response.json():
            return pd.DataFrame()

        response.raise_for_status()
        data = response.json()
    df = pd.DataFrame(data)
    flags = df[df['flag'].notnull()] if 'flag' in df.columns else []
    logger.success(f"Fetched {len(df)} total messages. {len(flags)} were Flag events.")
    return df


def fetch_team_radio(session_key: int, driver_number: int):
    logger.info(f"📻 Fetching Team Radio feeds for Driver #{driver_number}...")
    url = "https://api.openf1.org/v1/team_radio"

    with httpx.Client() as client:
        response = client.get(url, params={"session_key": session_key, "driver_number": driver_number})

        # Safely catch 404s for drivers who had no radio
        if response.status_code == 404 or not response.json():
            logger.warning(f"No radio found for driver #{driver_number}.")
            return pd.DataFrame()

        response.raise_for_status()
        data = response.json()
    df = pd.DataFrame(data)
    logger.success(f"Fetched {len(df)} radio transmissions!")
    return df


if __name__ == "__main__":
    # 1. Get the unique key for the 2024 Bahrain Race
    session_key = get_openf1_session_key(2024, "Bahrain", "Race")

    # 2. Get the official Race Control text stream
    rc_df = fetch_race_control_messages(session_key)
    logger.info(f"\nSample Race Control:\n{rc_df[['date', 'category', 'message']].head()}")

    # 3. Get Max Verstappen's (Car #1) radio recordings
    radio_df = fetch_team_radio(session_key, 1)
    if not radio_df.empty:
        logger.info(f"\nSample Team Radio (Audio URLs):\n{radio_df[['date', 'recording_url']].head()}")
