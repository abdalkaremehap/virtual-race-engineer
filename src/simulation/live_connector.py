import asyncio
import os
import sys

import httpx
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion.openf1_client import (
    fetch_race_control_messages,
    get_openf1_session_key,
)
from src.simulation.engine import LiveRaceSimulationEngine


class LiveGrandPrixConnector:
    """
    Phase 7 (Task 7.1): Production Bridge between OpenF1 Real-Time Telemetry API
    and the Asynchronous Simulation Engine (with automatic Replay Fallback).
    """

    def __init__(self, year: int = 2024, country: str = "Bahrain", ego_driver: str = "VER"):
        self.year = year
        self.country = country
        self.ego_driver = ego_driver
        self.engine = LiveRaceSimulationEngine(circuit=country, ego_driver=ego_driver)

    def map_fia_flag_to_track_status(self, message_text: str) -> str:
        msg = message_text.upper()
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg or "VSC DEPLOYED" in msg:
            return "VSC"
        if "SAFETY CAR DEPLOYED" in msg:
            return "SC"
        if "RED FLAG" in msg:
            return "RED"
        if "GREEN" in msg or "CLEAR" in msg or "SAFETY CAR IN THIS LAP" in msg:
            return "GREEN"
        return "GREEN"

    async def poll_openf1_race_control_once(self) -> str:
        """Polls OpenF1 Race Control endpoint to sync live FIA Flag states."""
        try:
            session_key = await asyncio.to_thread(
                get_openf1_session_key, self.year, self.country, "Race"
            )
            rc_df = await asyncio.to_thread(fetch_race_control_messages, session_key)
            if not rc_df.empty and "message" in rc_df.columns:
                latest_msg = str(rc_df.iloc[-1]["message"])
                detected = self.map_fia_flag_to_track_status(latest_msg)
                logger.info(
                    f"🛰️ [OpenF1 Live Feed | Session {session_key}] "
                    f"Latest FIA Message: '{latest_msg}' -> Status: {detected}"
                )
                return detected
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"⚠️ OpenF1 Live API unreachable ({e}). Using local DuckDB replay stream.")
        return "GREEN"

    async def launch(self, duration_sec: float = 4.0):
        logger.info(f"🏁 Connecting Pit Wall to {self.year} {self.country} Grand Prix...")
        await self.poll_openf1_race_control_once()
        await self.engine.run_live_simulation(duration_sec=duration_sec)


if __name__ == "__main__":
    connector = LiveGrandPrixConnector(year=2024, country="Bahrain", ego_driver="VER")
    asyncio.run(connector.launch(duration_sec=4.0))