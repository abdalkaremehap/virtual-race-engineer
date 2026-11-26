"""
Backend service layer powering the Multi-Page FastAPI F1 Pit Wall Web App.

100% reuses the verified Python engines (`AutonomousRaceEngineer`, `TyreL1TensorCache`,
`PitStopSimulator`, `RaceStrategyOptimizer`, `MultiHopFIARulebook`, and `telemetry_daemon`)
with zero Streamlit dependencies and zero changes to physics/strategy calculations.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.race_engineer import AutonomousRaceEngineer
from src.agent.tools import (
    RaceContext,
    RaceEngineerToolkit,
    build_live_data_from_standings_df,
    compute_remaining_stint_pit_breakeven,
    get_driver_used_compounds_up_to_lap,
)
from src.dashboard.radio_voice import format_text_for_f1_speech
from src.dashboard.smooth_broadcast_canvas import (
    build_broadcast_payload,
    render_smooth_broadcast_html,
)
from src.data_ingestion.telemetry_daemon import (
    DRIVER_METADATA_MAP,
    F1_2024_CALENDAR,
    F1_2024_DRIVERS,
    SESSION_DISPLAY_LABELS,
    SESSION_TYPES,
    SUPPORTED_SEASONS,
    compute_championship_standings_till_race,
    ensure_all_grand_prix_in_duckdb,
    fetch_real_fastf1_session,
    generate_circuit_map_coordinates,
    generate_driver_micro_telemetry,
    get_circuit_curvature_profile,
    get_circuit_metadata,
    get_gp_session_laps_df,
    get_real_driver_lap_telemetry,
    get_session_data_source,
    infer_lap_weather_condition,
    load_or_populate_gp_session,
)
from src.ml_surrogates.pit_simulator import PitStopSimulator

DB_PATH = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "circuit_pit_data.json")
_AUTO_FETCH_LOCK = threading.RLock()
_FAILED_FETCH_CACHE: dict[tuple[int, str, str], float] = {}
_FAILED_FETCH_TTL_SEC = 300.0

WEATHER_OPTIONS = [
    "Dry (Clear)",
    "Overcast (Cool)",
    "Light Drizzle (Damp)",
    "Heavy Rain (Wet)",
]

FAMOUS_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "hungary_2022_l39",
        "title": "2022 Hungary GP — Lap 39 (Cold Hards Trap)",
        "subtitle": "LEC leads P1 on 18L Mediums in cool 26.7°C conditions. Ferrari boxed for Hards and lost the race.",
        "season": 2022,
        "circuit": "Hungary",
        "lap": 39,
        "total_laps": 70,
        "ego_driver": "LEC",
        "track_status": "GREEN",
        "track_temp": 26.7,
        "weather": "Overcast (Cool)",
        "target_tyre": "HARD",
        "suggested_question": "Should we box for Hards right now?",
        "rows": [
            {"driver": "LEC", "position": 1, "compound": "MEDIUM", "tyre_life": 18, "gap_to_leader": 0.0, "lap_time_sec": 82.8},
            {"driver": "VER", "position": 2, "compound": "MEDIUM", "tyre_life": 1, "gap_to_leader": 7.2, "lap_time_sec": 81.9},
            {"driver": "RUS", "position": 3, "compound": "MEDIUM", "tyre_life": 2, "gap_to_leader": 11.4, "lap_time_sec": 82.4},
            {"driver": "SAI", "position": 4, "compound": "MEDIUM", "tyre_life": 22, "gap_to_leader": 14.2, "lap_time_sec": 83.1},
            {"driver": "HAM", "position": 5, "compound": "MEDIUM", "tyre_life": 20, "gap_to_leader": 18.0, "lap_time_sec": 83.0},
            {"driver": "PER", "position": 6, "compound": "MEDIUM", "tyre_life": 3, "gap_to_leader": 21.5, "lap_time_sec": 82.6},
            {"driver": "NOR", "position": 7, "compound": "MEDIUM", "tyre_life": 10, "gap_to_leader": 24.0, "lap_time_sec": 83.4},
            {"driver": "ALO", "position": 8, "compound": "HARD", "tyre_life": 18, "gap_to_leader": 31.0, "lap_time_sec": 83.8},
        ],
    },
    {
        "id": "silverstone_2022_l39",
        "title": "2022 British GP — Lap 39 (Safety Car Sprint)",
        "subtitle": "Ocon stops on track — SC deployed! LEC leads on 14-lap Hards while rivals pit for Softs.",
        "season": 2022,
        "circuit": "Great Britain",
        "lap": 39,
        "total_laps": 52,
        "ego_driver": "LEC",
        "track_status": "SC",
        "track_temp": 30.0,
        "weather": "Dry (Clear)",
        "target_tyre": "SOFT",
        "suggested_question": "Safety Car is out! Should we box for Softs right now?",
        "rows": [
            {"driver": "LEC", "position": 1, "compound": "HARD", "tyre_life": 14, "gap_to_leader": 0.0, "lap_time_sec": 102.0},
            {"driver": "SAI", "position": 2, "compound": "HARD", "tyre_life": 19, "gap_to_leader": 3.5, "lap_time_sec": 102.5},
            {"driver": "HAM", "position": 3, "compound": "HARD", "tyre_life": 6, "gap_to_leader": 6.2, "lap_time_sec": 102.3},
            {"driver": "PER", "position": 4, "compound": "MEDIUM", "tyre_life": 12, "gap_to_leader": 15.8, "lap_time_sec": 103.1},
            {"driver": "ALO", "position": 5, "compound": "HARD", "tyre_life": 15, "gap_to_leader": 19.4, "lap_time_sec": 103.4},
            {"driver": "NOR", "position": 6, "compound": "HARD", "tyre_life": 17, "gap_to_leader": 23.0, "lap_time_sec": 103.7},
        ],
    },
    {
        "id": "monaco_2022_l18",
        "title": "2022 Monaco GP — Lap 18 (Skip Intermediates?)",
        "subtitle": "Track drying rapidly — LEC leads on full Wets. Should Ferrari pit for Intermediates or go straight to Slicks?",
        "season": 2022,
        "circuit": "Monaco",
        "lap": 18,
        "total_laps": 64,
        "ego_driver": "LEC",
        "track_status": "GREEN",
        "track_temp": 28.0,
        "weather": "Dry (Clear)",
        "target_tyre": "INTERMEDIATE",
        "suggested_question": "Should we box for Intermediates right now?",
        "rows": [
            {"driver": "LEC", "position": 1, "compound": "WET", "tyre_life": 18, "gap_to_leader": 0.0, "lap_time_sec": 99.0},
            {"driver": "SAI", "position": 2, "compound": "WET", "tyre_life": 18, "gap_to_leader": 4.5, "lap_time_sec": 99.5},
            {"driver": "PER", "position": 3, "compound": "INTERMEDIATE", "tyre_life": 2, "gap_to_leader": 11.2, "lap_time_sec": 88.0},
            {"driver": "VER", "position": 4, "compound": "WET", "tyre_life": 18, "gap_to_leader": 14.5, "lap_time_sec": 99.8},
            {"driver": "NOR", "position": 5, "compound": "INTERMEDIATE", "tyre_life": 1, "gap_to_leader": 22.0, "lap_time_sec": 89.2},
            {"driver": "RUS", "position": 6, "compound": "WET", "tyre_life": 18, "gap_to_leader": 25.0, "lap_time_sec": 100.2},
        ],
    },
    {
        "id": "abudhabi_2021_l53",
        "title": "2021 Abu Dhabi GP — Lap 53 (Free SC Stop)",
        "subtitle": "Latifi crashes! HAM leads VER by 11.5s on 39-lap old Hards under Safety Car with 5 laps left.",
        "season": 2021,
        "circuit": "Abu Dhabi",
        "lap": 53,
        "total_laps": 58,
        "ego_driver": "HAM",
        "track_status": "SC",
        "track_temp": 30.0,
        "weather": "Dry (Clear)",
        "target_tyre": "SOFT",
        "suggested_question": "Safety Car is out! Should we box for fresh Softs?",
        "rows": [
            {"driver": "HAM", "position": 1, "compound": "HARD", "tyre_life": 39, "gap_to_leader": 0.0, "lap_time_sec": 115.0},
            {"driver": "VER", "position": 2, "compound": "HARD", "tyre_life": 17, "gap_to_leader": 13.8, "lap_time_sec": 115.2},
            {"driver": "PER", "position": 3, "compound": "SOFT", "tyre_life": 12, "gap_to_leader": 28.0, "lap_time_sec": 116.0},
            {"driver": "SAI", "position": 4, "compound": "HARD", "tyre_life": 29, "gap_to_leader": 34.0, "lap_time_sec": 116.4},
            {"driver": "TSU", "position": 5, "compound": "HARD", "tyre_life": 25, "gap_to_leader": 42.0, "lap_time_sec": 116.8},
            {"driver": "BOT", "position": 6, "compound": "HARD", "tyre_life": 22, "gap_to_leader": 48.0, "lap_time_sec": 117.0},
        ],
    },
]


class PitWallWebState:
    """Singleton state initialized once when the FastAPI app boots."""

    _instance: PitWallWebState | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        ensure_all_grand_prix_in_duckdb(DB_PATH)
        self.toolkit = RaceEngineerToolkit.get()
        self.tyre_cache = self.toolkit.tyre_cache
        self.engineer = AutonomousRaceEngineer(ego_driver="VER")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.circuits_db: dict[str, Any] = json.load(f)
        self.rag_prewarmed = False
        self.rag_mode = "lazy"

    @classmethod
    def get(cls) -> PitWallWebState:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_rag_prewarm(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return

        def _worker():
            try:
                mh = self.toolkit.multihop_rag
                self.rag_mode = getattr(mh.rulebook, "mode", "remote")
                mh.rulebook.query_cached("safety car", top_k=1)
                mh._get_all_chunks()
                self.rag_prewarmed = True
                logger.success(f"✅ [Web Startup] FIA Multi-Hop RAG pre-warmed ({self.rag_mode}).")
            except Exception as e:
                logger.warning(f"⚠️ [Web Startup] FIA RAG pre-warm skipped: {e}")

        threading.Thread(target=_worker, daemon=True, name="web-fia-rag-prewarm").start()


@lru_cache(maxsize=64)
def get_cached_pit_simulator(circuit: str, pit_box_num: int) -> PitStopSimulator:
    return PitStopSimulator(circuit=circuit, team_pit_box_position=int(pit_box_num), seed=42)


@lru_cache(maxsize=32)
def _build_circuit_svg_path(circuit: str) -> str:
    """Builds a normalized SVG polyline path (`viewBox="0 0 120 74"`) from real FIA GPS track coords."""
    x, y = generate_circuit_map_coordinates(circuit, n_points=140)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    min_x, max_x = float(x_arr.min()), float(x_arr.max())
    min_y, max_y = float(y_arr.min()), float(y_arr.max())
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    pad = 8.0
    scale_x = (120.0 - 2 * pad) / span
    scale_y = (74.0 - 2 * pad) / span
    scale = min(scale_x, scale_y)
    ox = (120.0 - (max_x - min_x) * scale) / 2.0
    oy = (74.0 - (max_y - min_y) * scale) / 2.0
    pts = [
        f"{ox + (xi - min_x) * scale:.1f},{74.0 - (oy + (yi - min_y) * scale):.1f}"
        for xi, yi in zip(x_arr, y_arr)
    ]
    return "M " + " L ".join(pts) + " Z"


def get_season_calendar(season: int) -> list[dict[str, Any]]:
    """Returns all 24 Grand Prix rounds with metadata, GPS SVG preview paths, and live FIA cache badges."""
    season_i = int(season) if int(season) in SUPPORTED_SEASONS else 2024
    state = PitWallWebState.get()
    circuits_cfg = state.circuits_db.get("circuits", {})
    out: list[dict[str, Any]] = []
    for item in F1_2024_CALENDAR:
        cname = item["circuit"]
        cfg = circuits_cfg.get(cname, {})
        src = get_session_data_source(season=season_i, circuit=cname, session_type="Race", db_path=DB_PATH)
        out.append({
            "round": int(item["round"]),
            "circuit": cname,
            "race_name": item["race_name"],
            "country": item.get("country", cname),
            "total_laps": int(item["total_laps"]),
            "track_length_m": int(item.get("track_length_m", 5412)),
            "pit_loss_sec": float(cfg.get("green_flag_pit_loss_sec", 22.0)),
            "abrasion": float(cfg.get("abrasion_factor", 1.0)),
            "overtaking_difficulty": float(cfg.get("overtaking_difficulty", 1.0)),
            "is_real_fia": src == "REAL_FASTF1",
            "data_source": src,
            "svg_path": _build_circuit_svg_path(cname),
        })
    return out


def classify_verdict_badge(reply_text: str, user_query: str = "") -> tuple[str, str]:
    """Extracts the Pit Wall verdict pill label & color (aligned with `_format_radio_reply_html`)."""
    upper_txt = (reply_text or "").upper()
    q_low = (user_query or "").lower()
    asked_pit = bool(
        re.search(r"\b(?:box|pit|pitting|undercut|overcut|strategy|stop|soft|medium|hard|inter|wet)\b", q_low)
    )
    is_pure_battle = ("RIVAL BATTLE" in upper_txt) and not asked_pit

    if is_pure_battle:
        return "⚔️ TACTICAL BATTLE BRIEF", "#FFB300"
    if "SKIP " in upper_txt and "BOX STRAIGHT FOR" in upper_txt:
        return "⚠️ SKIP COMPOUND · SWITCH TO SLICKS", "#FFB300"
    if (
        "WEATHER CROSSOVER" in upper_txt
        or "BOX THIS LAP FOR WET" in upper_txt
        or "BOX THIS LAP FOR INTERMEDIATE" in upper_txt
    ):
        return "🌧️ WEATHER CROSSOVER · BOX NOW", "#00E5FF"
    if "STAY OUT" in upper_txt or "DO NOT BOX" in upper_txt:
        return "🟢 PIT CALL · STAY OUT", "#00FF88"
    if (
        "BOX THIS LAP" in upper_txt
        or "BOX BOX" in upper_txt
        or "BOX FOR UNDERCUT" in upper_txt
        or "UNDERCUT WINDOW OPEN" in upper_txt
    ):
        return "🚨 PIT CALL · BOX THIS LAP", "#FF4B4B"
    if "PUSH TO PASS" in upper_txt or "CATCH" in upper_txt or "RIVAL BATTLE" in upper_txt:
        return "⚔️ TACTICAL BATTLE BRIEF", "#FFB300"
    if "FIA" in upper_txt or "REGULATION" in upper_txt or "PENALTY" in upper_txt:
        return "⚖️ FIA REGULATIONS BRIEF", "#A78BFA"
    return "🎧 PIT WALL TELEMETRY BRIEF", "#38BDF8"


def _compute_degradation_curves(track_temp: float, max_lap: int, circuit: str, weather: str) -> dict[str, Any]:
    state = PitWallWebState.get()
    tc = state.tyre_cache
    max_l = max(10, min(65, int(max_lap)))
    laps = list(range(1, max_l + 1))
    curves: dict[str, list[float]] = {}
    for comp in ("SOFT", "MEDIUM", "HARD"):
        curves[comp] = [
            round(
                float(
                    tc.lookup_degradation(
                        comp, track_temp, age, race_lap=age, weather_condition=weather, circuit=circuit
                    )
                ),
                3,
            )
            for age in laps
        ]
    return {"laps": laps, "curves": curves}


def _compute_mc_rejoin_summary(
    circuit: str,
    ego_driver: str,
    df_lap: pd.DataFrame,
    track_status: str,
    pit_box_num: int,
    base_lap_sec: float,
) -> dict[str, Any]:
    sim = get_cached_pit_simulator(circuit, pit_box_num)
    ego_rows = df_lap[df_lap["driver"] == ego_driver]
    ego_row = ego_rows.iloc[0] if not ego_rows.empty else df_lap.iloc[0]
    ego_gap = float(ego_row["gap_to_leader"])
    traffic = [
        {
            "driver": str(r.driver),
            "gap_to_leader": float(r.gap_to_leader),
            "is_lapped": float(r.gap_to_leader) >= base_lap_sec,
        }
        for r in df_lap.itertuples(index=False)
        if str(r.driver) != str(ego_row["driver"])
    ]
    mc = sim.simulate_rejoin_mc(
        driver=str(ego_row["driver"]),
        current_gap_to_leader=ego_gap,
        traffic=traffic,
        race_condition=track_status.upper(),
        n_sims=10_000,
        base_lap_sec=base_lap_sec,
        silent=True,
    )
    pos_dist = mc.get("position_distribution", {})
    positions = sorted(int(k) for k in pos_dist.keys())
    probs = [round(float(pos_dist[p]) * 100.0, 2) for p in positions]
    return {
        "driver": str(ego_row["driver"]),
        "condition": track_status.upper(),
        "median_pos": int(mc["median_pos"]),
        "expected_pos": round(float(mc["expected_pos"]), 2),
        "median_pit_loss": round(float(mc["median_pit_loss"]), 2),
        "p95_pit_loss": round(float(mc["p95_pit_loss"]), 2),
        "green_flag_pit_loss": round(float(mc["green_flag_pit_loss"]), 2),
        "p_clean_air_pct": round(float(mc["p_clean_air"]) * 100.0, 1),
        "positions": [f"P{p}" for p in positions],
        "probabilities": probs,
    }


def _extract_ego_kpis(
    circuit: str,
    ego_driver: str,
    df_lap: pd.DataFrame,
    track_temp: float,
    weather: str,
    total_laps: int,
    pit_loss_sec: float,
    target_tyre: str,
) -> dict[str, Any]:
    state = PitWallWebState.get()
    ego_rows = df_lap[df_lap["driver"] == ego_driver]
    row = ego_rows.iloc[0] if not ego_rows.empty else df_lap.iloc[0]
    drv = str(row["driver"])
    pos = int(row["position"])
    comp = str(row["compound"]).upper()
    age = int(row["tyre_life"])
    gap = float(row["gap_to_leader"])
    ltime = float(row["lap_time_sec"])
    lap = int(row["lap"])
    laps_rem = max(0, int(total_laps) - lap)

    deg = float(
        state.tyre_cache.lookup_degradation(
            comp, track_temp, age, race_lap=lap, weather_condition=weather, circuit=circuit
        )
    )
    next_5l = sum(
        float(
            state.tyre_cache.lookup_degradation(
                comp, track_temp, age + k, race_lap=lap + k, weather_condition=weather, circuit=circuit
            )
        )
        for k in range(1, 6)
    )
    be = compute_remaining_stint_pit_breakeven(
        ego_compound=comp,
        ego_tyre_life=age,
        track_temp=track_temp,
        laps_remaining=laps_rem,
        pit_loss_sec=pit_loss_sec,
        target_new_tyre=target_tyre,
        race_lap=lap,
        weather_condition=weather,
        circuit=circuit,
    )
    meta = DRIVER_METADATA_MAP.get(drv, {"name": drv, "team": "F1 Racing", "number": 0, "color": "#00E5FF"})
    return {
        "driver": drv,
        "driver_name": str(meta.get("name", drv)),
        "team": str(meta.get("team", "F1 Racing")),
        "number": int(meta.get("number", 0)),
        "color": str(meta.get("color", "#00E5FF")),
        "position": pos,
        "compound": comp,
        "tyre_life": age,
        "gap_to_leader": round(gap, 3),
        "lap_time_sec": round(ltime, 3),
        "current_deg": round(deg, 3),
        "next_5lap_deg": round(next_5l, 2),
        "laps_remaining": laps_rem,
        "net_payback_sec": round(float(be["net_stint_pit_payback_sec"]), 2),
        "recoverable_gain_sec": round(float(be["recoverable_tyre_gain_sec"]), 2),
        "fresh_compound": str(be["fresh_compound"]),
    }


def _df_to_laps_map(df_all: pd.DataFrame, status_override: str | None = None) -> dict[int, list[dict]]:
    laps_map: dict[int, list[dict]] = {}
    if df_all.empty:
        return laps_map
    for r in df_all.itertuples(index=False):
        lap_i = int(r.lap)
        laps_map.setdefault(lap_i, []).append({
            "driver": str(r.driver),
            "position": int(r.position),
            "compound": str(r.compound).upper(),
            "tyre_life": int(r.tyre_life),
            "gap_to_leader": float(r.gap_to_leader),
            "lap_time_sec": float(r.lap_time_sec),
            "track_status": str(status_override).upper() if status_override else str(r.track_status).upper(),
        })
    return laps_map


def get_race_session_bundle(
    *,
    season: int,
    circuit: str,
    session_type: str = "Race",
    lap: int | None = None,
    ego_driver: str = "VER",
    target_tyre: str = "HARD",
    weather_override: str | None = None,
    status_override: str | None = None,
    track_temp_override: float | None = None,
    pit_box_num: int = 1,
    include_canvas_html: bool = True,
    auto_fetch_real: bool = True,
) -> dict[str, Any]:
    """Builds the full Live Pit Wall session payload for `/race/{season}/{circuit}` and `/api/race/.../session`."""
    PitWallWebState.get()
    season_i = int(season) if int(season) in SUPPORTED_SEASONS else 2024
    circuit_meta = get_circuit_metadata(circuit)
    canon_circuit = circuit_meta["circuit"]

    # Always auto-fetch REAL FIA telemetry (OpenF1 for 2023-2024, FastF1 for 2021-2022) on first visit
    # instead of silently serving synthetic/calibrated data! Cached permanently in DuckDB after first load.
    if auto_fetch_real and not os.environ.get("PYTEST_CURRENT_TEST"):
        cache_key = (season_i, canon_circuit, str(session_type))
        with _AUTO_FETCH_LOCK:
            cur_src = get_session_data_source(
                season=season_i, circuit=canon_circuit, session_type=session_type, db_path=DB_PATH
            )
            last_fail = _FAILED_FETCH_CACHE.get(cache_key, 0.0)
            if cur_src != "REAL_FASTF1" and (time.time() - last_fail) > _FAILED_FETCH_TTL_SEC:
                logger.info(
                    f"🛰️ [Web Auto-Fetch] Loading REAL FIA telemetry for {season_i} {canon_circuit} ({session_type})..."
                )
                ok = fetch_real_fastf1_session(
                    circuit=canon_circuit,
                    session_type=session_type,
                    season=season_i,
                    db_path=DB_PATH,
                )
                if not ok:
                    _FAILED_FETCH_CACHE[cache_key] = time.time()
                else:
                    _FAILED_FETCH_CACHE.pop(cache_key, None)

    avail_laps = load_or_populate_gp_session(canon_circuit, session_type, season=season_i, db_path=DB_PATH)
    max_lap = max(avail_laps) if avail_laps else int(circuit_meta["total_laps"])
    active_lap = int(np.clip(lap if lap is not None else min(17, max_lap), 1, max_lap))

    df_all = get_gp_session_laps_df(circuit=canon_circuit, session_type=session_type, season=season_i, db_path=DB_PATH)
    df_lap = df_all[df_all["lap"] == active_lap].sort_values("position").copy()
    if df_lap.empty:
        first_l = int(df_all["lap"].min()) if not df_all.empty else 1
        active_lap = first_l
        df_lap = df_all[df_all["lap"] == active_lap].sort_values("position").copy()

    drivers_in_session = [str(d) for d in df_lap["driver"].tolist()]
    active_ego = ego_driver.upper() if ego_driver.upper() in drivers_in_session else (drivers_in_session[0] if drivers_in_session else "VER")

    raw_status = str(df_lap["track_status"].iloc[0]).upper() if not df_lap.empty else "GREEN"
    active_status = str(status_override).upper() if status_override else raw_status
    raw_temp = float(df_lap["track_temp"].iloc[0]) if not df_lap.empty else float(circuit_meta["track_temp"])
    active_temp = float(track_temp_override) if track_temp_override is not None else raw_temp
    inferred_wx = infer_lap_weather_condition(df_lap)
    active_weather = weather_override if weather_override in WEATHER_OPTIONS else inferred_wx

    if status_override:
        df_lap["track_status"] = active_status
    if track_temp_override is not None:
        df_lap["track_temp"] = active_temp

    ego_row = df_lap[df_lap["driver"] == active_ego].iloc[0]
    ego_comp = str(ego_row["compound"]).upper()
    ego_age = int(ego_row["tyre_life"])

    x_track, y_track = generate_circuit_map_coordinates(canon_circuit, n_points=320)
    norm_curv = get_circuit_curvature_profile(canon_circuit, n_points=320)
    micro_tel = get_real_driver_lap_telemetry(
        season=season_i,
        circuit=canon_circuit,
        session_type=session_type,
        driver=active_ego,
        lap=active_lap,
        compound=ego_comp,
        tyre_life=ego_age,
        track_status=active_status,
    )
    if micro_tel is None or micro_tel.empty:
        micro_tel = generate_driver_micro_telemetry(
            driver=active_ego,
            circuit=canon_circuit,
            lap=active_lap,
            compound=ego_comp,
            tyre_life=ego_age,
        )

    mc_summary = _compute_mc_rejoin_summary(
        circuit=canon_circuit,
        ego_driver=active_ego,
        df_lap=df_lap,
        track_status=active_status,
        pit_box_num=pit_box_num,
        base_lap_sec=float(circuit_meta["base_lap_sec"]),
    )
    pit_loss_now = float(mc_summary["median_pit_loss"])
    laps_map = _df_to_laps_map(df_all, status_override=status_override)

    payload = build_broadcast_payload(
        circuit=canon_circuit,
        race_name=circuit_meta["race_name"],
        season=season_i,
        session_label=SESSION_DISPLAY_LABELS.get(session_type, session_type),
        x_track=x_track,
        y_track=y_track,
        norm_curv=norm_curv,
        track_length_m=float(circuit_meta["track_length_m"]),
        base_lap_sec=float(circuit_meta["base_lap_sec"]),
        total_laps=max_lap,
        initial_lap=active_lap,
        initial_sub_lap=0.35,
        ego_driver=active_ego,
        pit_loss_sec=pit_loss_now,
        track_temp_c=active_temp,
        weather_cond=active_weather,
        laps_by_number=laps_map,
        driver_meta=DRIVER_METADATA_MAP,
        micro_tel_df=micro_tel,
        auto_play=False,
        initial_speed=15.0,
    )

    kpis = _extract_ego_kpis(
        circuit=canon_circuit,
        ego_driver=active_ego,
        df_lap=df_lap,
        track_temp=active_temp,
        weather=active_weather,
        total_laps=max_lap,
        pit_loss_sec=pit_loss_now,
        target_tyre=target_tyre.upper(),
    )
    deg_curves = _compute_degradation_curves(active_temp, max_lap, canon_circuit, active_weather)
    data_source = get_session_data_source(season=season_i, circuit=canon_circuit, session_type=session_type, db_path=DB_PATH)

    result: dict[str, Any] = {
        "season": season_i,
        "circuit": canon_circuit,
        "race_name": circuit_meta["race_name"],
        "session_type": session_type,
        "lap": active_lap,
        "max_lap": max_lap,
        "ego_driver": active_ego,
        "target_tyre": target_tyre.upper(),
        "track_status": active_status,
        "track_temp": round(active_temp, 1),
        "weather": active_weather,
        "inferred_weather": inferred_wx,
        "data_source": data_source,
        "is_real_fia": data_source == "REAL_FASTF1",
        "drivers": drivers_in_session,
        "kpis": kpis,
        "mc_rejoin": mc_summary,
        "tyre_curves": deg_curves,
        "broadcast_payload": payload,
    }
    if include_canvas_html:
        result["canvas_html"] = render_smooth_broadcast_html(payload)
    return result


def fetch_real_fia_session_for_web(season: int, circuit: str, session_type: str = "Race") -> dict[str, Any]:
    """Triggers real FastF1 / OpenF1 FIA session download into DuckDB (bypasses failed-fetch cooldown)."""
    meta = get_circuit_metadata(circuit)
    canon = meta["circuit"]
    cache_key = (int(season), canon, str(session_type))
    with _AUTO_FETCH_LOCK:
        _FAILED_FETCH_CACHE.pop(cache_key, None)
        ok = fetch_real_fastf1_session(circuit=canon, session_type=session_type, season=int(season), db_path=DB_PATH)
    src = get_session_data_source(season=int(season), circuit=canon, session_type=session_type, db_path=DB_PATH)
    return {"ok": bool(ok), "circuit": canon, "season": int(season), "session_type": session_type, "data_source": src}


def ask_pit_wall_radio(
    *,
    question: str,
    season: int = 2024,
    circuit: str = "Bahrain",
    session_type: str = "Race",
    lap: int = 17,
    ego_driver: str = "VER",
    weather: str | None = None,
    track_status: str | None = None,
    track_temp: float | None = None,
    pit_box_num: int = 1,
    custom_rows: list[dict[str, Any]] | None = None,
    total_laps_override: int | None = None,
) -> dict[str, Any]:
    """Invokes the LangGraph AutonomousRaceEngineer with an isolated RaceContext and returns structured JSON."""
    state = PitWallWebState.get()
    meta = get_circuit_metadata(circuit)
    canon_circuit = meta["circuit"]

    if custom_rows:
        df_lap = pd.DataFrame([
            {
                "lap": int(lap),
                "driver": str(r["driver"]).upper(),
                "position": int(r["position"]),
                "compound": str(r["compound"]).upper(),
                "tyre_life": int(r["tyre_life"]),
                "gap_to_leader": float(r["gap_to_leader"]),
                "lap_time_sec": float(r.get("lap_time_sec", meta["base_lap_sec"])),
                "track_status": str(track_status or "GREEN").upper(),
                "track_temp": float(track_temp if track_temp is not None else meta["track_temp"]),
            }
            for r in custom_rows
        ]).sort_values("position")
        total_laps = int(total_laps_override or meta["total_laps"])
        used_dry: tuple[str, ...] = ()
    else:
        avail_laps = load_or_populate_gp_session(canon_circuit, session_type, season=int(season), db_path=DB_PATH)
        total_laps = max(avail_laps) if avail_laps else int(meta["total_laps"])
        active_lap = int(np.clip(lap, 1, total_laps))
        df_all = get_gp_session_laps_df(circuit=canon_circuit, session_type=session_type, season=int(season), db_path=DB_PATH)
        df_lap = df_all[df_all["lap"] == active_lap].sort_values("position").copy()
        if df_lap.empty:
            df_lap = df_all[df_all["lap"] == int(df_all["lap"].min())].sort_values("position").copy()
        if track_status:
            df_lap["track_status"] = str(track_status).upper()
        if track_temp is not None:
            df_lap["track_temp"] = float(track_temp)
        used_dry = get_driver_used_compounds_up_to_lap(
            season=int(season),
            circuit=canon_circuit,
            session_type=session_type,
            driver=ego_driver.upper(),
            up_to_lap=active_lap,
            db_path=DB_PATH,
        )

    active_ego = ego_driver.upper()
    if active_ego not in df_lap["driver"].values:
        active_ego = str(df_lap["driver"].iloc[0])

    active_wx = weather if weather in WEATHER_OPTIONS else infer_lap_weather_condition(df_lap)
    active_status = str(df_lap["track_status"].iloc[0]).upper()
    active_lap_num = int(df_lap["lap"].iloc[0])

    # Compute teammate gap ahead if same team
    tm_gap_ahead: float | None = None
    ego_team = DRIVER_METADATA_MAP.get(active_ego, {}).get("team")
    if ego_team:
        ego_gap = float(df_lap[df_lap["driver"] == active_ego]["gap_to_leader"].iloc[0])
        for r in df_lap.itertuples(index=False):
            if str(r.driver) != active_ego and DRIVER_METADATA_MAP.get(str(r.driver), {}).get("team") == ego_team:
                diff = ego_gap - float(r.gap_to_leader)
                if diff > 0:
                    tm_gap_ahead = round(diff, 2)
                break

    ctx = RaceContext(
        driver=active_ego,
        circuit=canon_circuit,
        total_race_laps=total_laps,
        team_pit_box_position=int(pit_box_num),
        weather_condition=active_wx,
        used_dry_compounds=used_dry,
        teammate_gap_ahead_sec=tm_gap_ahead,
        teammate_pitting=False,
        live_data=build_live_data_from_standings_df(
            df_lap,
            circuit=canon_circuit,
            total_race_laps=total_laps,
            weather_condition=active_wx,
            used_dry_compounds=used_dry,
            teammate_gap_ahead_sec=tm_gap_ahead,
            teammate_pitting=False,
        ),
    )

    res = state.engineer.ask_engineer(question.strip(), context=ctx)
    reply_txt = str(res.get("engineer_radio_reply", ""))
    human_call = str(res.get("human_radio_call", "")).strip() or reply_txt
    verdict_label, verdict_color = classify_verdict_badge(reply_txt, user_query=question)

    strat_rank = (res.get("pit_sim_data") or {}).get("strategy_ranking") or {}
    d_meta = DRIVER_METADATA_MAP.get(active_ego, {"name": active_ego, "team": "F1 Racing", "number": 0, "color": "#00E5FF"})

    return {
        "msg_id": f"radio_{time.time_ns()}",
        "driver": active_ego,
        "driver_num": int(d_meta.get("number", 0)),
        "team_color": str(d_meta.get("color", "#00E5FF")),
        "lap_tag": f"L{active_lap_num}/{total_laps} · {active_status}",
        "question": question.strip(),
        "human_radio_call": human_call,
        "spoken_text": format_text_for_f1_speech(human_call),
        "engineer_radio_reply": reply_txt,
        "verdict_label": verdict_label,
        "verdict_color": verdict_color,
        "guardrail_passed": bool(res.get("guardrail_passed", True)),
        "guardrail_notes": list(res.get("guardrail_notes", [])),
        "elapsed_ms": float(res.get("elapsed_ms", 0.0)),
        "required_tools": list(res.get("required_tools", [])),
        "strategy_ranking": {
            "optimal": strat_rank.get("optimal"),
            "optimal_beats_requested_by_sec": float(strat_rank.get("optimal_beats_requested_by_sec", 0.0)),
            "top_options": (strat_rank.get("top_options") or [])[:4],
        } if strat_rank else None,
        "legal_data": res.get("legal_data") or None,
    }


def run_sandbox_simulation(
    *,
    circuit: str,
    lap: int,
    total_laps: int,
    ego_driver: str,
    track_status: str,
    track_temp: float,
    weather: str,
    target_tyre: str,
    rows: list[dict[str, Any]],
    pit_box_num: int = 1,
) -> dict[str, Any]:
    """Simulates a custom What-If scenario grid and returns 60 FPS canvas HTML + MC rejoin + tyre curves."""
    PitWallWebState.get()
    meta = get_circuit_metadata(circuit)
    canon = meta["circuit"]
    df_lap = pd.DataFrame([
        {
            "lap": int(lap),
            "driver": str(r["driver"]).upper(),
            "position": int(r["position"]),
            "compound": str(r["compound"]).upper(),
            "tyre_life": int(r["tyre_life"]),
            "gap_to_leader": float(r["gap_to_leader"]),
            "lap_time_sec": float(r.get("lap_time_sec", meta["base_lap_sec"])),
            "track_status": str(track_status).upper(),
            "track_temp": float(track_temp),
        }
        for r in rows
    ]).sort_values("position")

    active_ego = ego_driver.upper() if ego_driver.upper() in df_lap["driver"].values else str(df_lap["driver"].iloc[0])
    ego_row = df_lap[df_lap["driver"] == active_ego].iloc[0]

    x_track, y_track = generate_circuit_map_coordinates(canon, n_points=320)
    norm_curv = get_circuit_curvature_profile(canon, n_points=320)
    micro_tel = generate_driver_micro_telemetry(
        driver=active_ego,
        circuit=canon,
        lap=int(lap),
        compound=str(ego_row["compound"]).upper(),
        tyre_life=int(ego_row["tyre_life"]),
    )
    mc = _compute_mc_rejoin_summary(
        circuit=canon,
        ego_driver=active_ego,
        df_lap=df_lap,
        track_status=track_status.upper(),
        pit_box_num=pit_box_num,
        base_lap_sec=float(meta["base_lap_sec"]),
    )
    pit_loss_now = float(mc["median_pit_loss"])
    laps_map = _df_to_laps_map(df_lap)
    payload = build_broadcast_payload(
        circuit=canon,
        race_name=f"CUSTOM SANDBOX · {meta['race_name']}",
        season=2024,
        session_label="Scenario Lab",
        x_track=x_track,
        y_track=y_track,
        norm_curv=norm_curv,
        track_length_m=float(meta["track_length_m"]),
        base_lap_sec=float(meta["base_lap_sec"]),
        total_laps=int(total_laps),
        initial_lap=int(lap),
        initial_sub_lap=0.45,
        ego_driver=active_ego,
        pit_loss_sec=pit_loss_now,
        track_temp_c=float(track_temp),
        weather_cond=weather,
        laps_by_number=laps_map,
        driver_meta=DRIVER_METADATA_MAP,
        micro_tel_df=micro_tel,
        auto_play=False,
        initial_speed=15.0,
    )
    kpis = _extract_ego_kpis(
        circuit=canon,
        ego_driver=active_ego,
        df_lap=df_lap,
        track_temp=float(track_temp),
        weather=weather,
        total_laps=int(total_laps),
        pit_loss_sec=pit_loss_now,
        target_tyre=target_tyre.upper(),
    )
    curves = _compute_degradation_curves(float(track_temp), int(total_laps), canon, weather)
    return {
        "circuit": canon,
        "lap": int(lap),
        "total_laps": int(total_laps),
        "ego_driver": active_ego,
        "kpis": kpis,
        "mc_rejoin": mc,
        "tyre_curves": curves,
        "canvas_html": render_smooth_broadcast_html(payload),
    }


def get_championship_bundle(season: int, circuit: str) -> dict[str, Any]:
    """Returns official WDC & WCC championship standings up to the selected season & Grand Prix."""
    PitWallWebState.get()
    season_i = int(season) if int(season) in SUPPORTED_SEASONS else 2024
    meta = get_circuit_metadata(circuit)
    canon = meta["circuit"]
    avail = load_or_populate_gp_session(canon, "Race", season=season_i, db_path=DB_PATH)
    final_lap = max(avail) if avail else int(meta["total_laps"])
    df_all = get_gp_session_laps_df(circuit=canon, session_type="Race", season=season_i, db_path=DB_PATH)
    df_final = df_all[df_all["lap"] == final_lap].sort_values("position").copy()
    champ = compute_championship_standings_till_race(
        season=season_i,
        circuit=canon,
        current_race_standings_df=df_final,
    )
    wdc_rows = []
    for idx_r, r in enumerate(champ["wdc"].itertuples(index=False), start=1):
        d_code = str(r.driver)
        dm = DRIVER_METADATA_MAP.get(d_code, {"color": "#38bdf8", "number": 0})
        pos_val = int(getattr(r, "position", getattr(r, "rank", idx_r)))
        wdc_rows.append({
            "position": pos_val,
            "driver": d_code,
            "driver_name": str(getattr(r, "driver_name", dm.get("name", d_code))),
            "team": str(r.team),
            "color": str(dm.get("color", "#38bdf8")),
            "prior_points": float(getattr(r, "prior_points", 0.0)),
            "live_race_points": float(getattr(r, "live_race_points", 0.0)),
            "total_points": float(getattr(r, "total_points", 0.0)),
            "gap_to_leader": float(getattr(r, "gap_to_leader", 0.0)),
        })
    wcc_rows = []
    team_colors = {m["team"]: m["color"] for m in DRIVER_METADATA_MAP.values()}
    for idx_t, r in enumerate(champ["wcc"].itertuples(index=False), start=1):
        t_name = str(r.team)
        pos_val = int(getattr(r, "position", getattr(r, "rank", idx_t)))
        wcc_rows.append({
            "position": pos_val,
            "team": t_name,
            "color": str(team_colors.get(t_name, "#38bdf8")),
            "prior_points": float(getattr(r, "prior_points", 0.0)),
            "live_race_points": float(getattr(r, "live_race_points", 0.0)),
            "total_points": float(getattr(r, "total_points", 0.0)),
            "gap_to_leader": float(getattr(r, "gap_to_leader", 0.0)),
        })
    return {
        "season": season_i,
        "circuit": canon,
        "race_name": meta["race_name"],
        "round_number": int(champ.get("round_number", meta["round"])),
        "wdc": wdc_rows,
        "wcc": wcc_rows,
    }


def search_fia_rules(question: str, top_k: int = 3) -> dict[str, Any]:
    """Direct 2-Hop FIA Sporting Regulations lookup with spoken summary."""
    state = PitWallWebState.get()
    q = question.strip()
    rag_res = state.toolkit.multihop_rag.ask_multihop(q, top_k=max(1, min(5, int(top_k))))
    human_summary = state.engineer._build_human_radio_call(
        driver_code="VER",
        radio_message=q,
        target_rival=None,
        pit_ctx=None,
        battle_data=None,
        tyre_data=None,
        legal_data=rag_res,
    )
    return {
        "question": q,
        "offline": bool(rag_res.get("offline", False)),
        "error": rag_res.get("error"),
        "elapsed_ms": float(rag_res.get("elapsed_ms", 0.0)),
        "cache_hit": bool(rag_res.get("cache_hit", False)),
        "human_summary": human_summary,
        "spoken_text": format_text_for_f1_speech(human_summary),
        "hop1_primary_rules": rag_res.get("hop1_primary_rules", []),
        "cited_articles": rag_res.get("cited_articles", []),
        "hop2_resolved_references": rag_res.get("hop2_resolved_references", []),
    }


def get_system_health() -> dict[str, Any]:
    state = PitWallWebState.get()
    return {
        "status": "ok",
        "l1_tensor_shape": list(state.tyre_cache.tensor.shape),
        "l1_tensor_kb": round(state.tyre_cache.tensor.nbytes / 1024.0, 2),
        "circuits_count": len(F1_2024_CALENDAR),
        "seasons": list(SUPPORTED_SEASONS),
        "rag_prewarmed": bool(state.rag_prewarmed),
        "rag_mode": state.rag_mode,
    }
