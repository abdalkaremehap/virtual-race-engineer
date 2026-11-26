from functools import lru_cache
import html
import json
import os
import re
import sys
import threading
import time

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
from src.dashboard.radio_voice import render_f1_radio_voice_bar_html
from src.dashboard.smooth_broadcast_canvas import (
    build_broadcast_payload,
    get_streamlit_anti_blink_css,
    render_smooth_broadcast_html,
)
from src.data_ingestion.telemetry_daemon import (
    DRIVER_METADATA_MAP,
    F1_2024_CALENDAR,
    F1_2024_DRIVERS,
    FIA_RACE_POINTS_TABLE,
    SESSION_DISPLAY_LABELS,
    SESSION_TYPES,
    SUPPORTED_SEASONS,
    compute_championship_standings_till_race,
    duckdb_connect,
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
    sync_gp_lap_to_live_table,
)
from src.ml_surrogates.pit_simulator import PitStopSimulator

st.set_page_config(
    page_title="F1 Virtual Race Engineer — Pit Wall Command Center",
    page_icon="🏎️",
    layout="wide",
)
st.markdown(get_streamlit_anti_blink_css(), unsafe_allow_html=True)


@st.cache_resource
def load_core_engines():
    """Loads the 56.86 KB L1 Tensor Cache, DuckDB Calendar/Sessions, and LangGraph Race Engineer once into RAM."""
    db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    ensure_all_grand_prix_in_duckdb(db_path)
    tk = RaceEngineerToolkit.get()
    tyre_cache = tk.tyre_cache
    engineer = AutonomousRaceEngineer(ego_driver="VER")
    config_path = os.path.join(PROJECT_ROOT, "config", "circuit_pit_data.json")
    with open(config_path, "r") as f:
        circuits_db = json.load(f)

    # Pre-warm SentenceTransformer + Qdrant Multi-Hop FIA RAG silently in background so the first SC/FIA query is instant
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        def _prewarm_rag_bg():
            try:
                mh = tk.multihop_rag
                mh.rulebook.query_cached("safety car", top_k=1)
                mh._get_all_chunks()
            except Exception:
                pass

        threading.Thread(target=_prewarm_rag_bg, daemon=True, name="fia-rag-prewarm").start()

    return tyre_cache, engineer, circuits_db, db_path


@lru_cache(maxsize=64)
def get_cached_pit_simulator(circuit: str, pit_box_num: int) -> PitStopSimulator:
    """Reuses PitStopSimulator instances across 4 Hz ticks instead of re-reading circuit_pit_data.json from disk."""
    return PitStopSimulator(circuit=circuit, team_pit_box_position=int(pit_box_num), seed=42)


@lru_cache(maxsize=64)
def get_cached_degradation_curves(track_temp_rounded: float, max_lap: int) -> dict[str, list[float]]:
    """Vectorized L1 tensor slice for SOFT/MEDIUM/HARD curves cached per (temp, max_lap)."""
    laps = np.arange(1, int(max_lap) + 1)
    t_idx = int(np.clip(round(track_temp_rounded) - tyre_cache.TEMP_MIN, 0, tyre_cache.TEMP_MAX - tyre_cache.TEMP_MIN))
    l_idx = np.clip(laps, 0, tyre_cache.MAX_LAP).astype(np.intp)
    return {
        comp: tyre_cache.tensor[tyre_cache.COMPOUND_TO_IDX[comp], t_idx, l_idx].astype(float).tolist()
        for comp in ("SOFT", "MEDIUM", "HARD")
    }


@lru_cache(maxsize=16)
def get_cached_session_laps_map(
    season: int,
    circuit: str,
    session_type: str,
    track_status_override: str | None,
    db_path: str,
) -> dict[int, list[dict]]:
    """Loads all laps of a Grand Prix session once for zero-blink 60 FPS client-side canvas playback."""
    df_all = get_gp_session_laps_df(circuit=circuit, session_type=session_type, season=season, db_path=db_path)
    laps_map: dict[int, list[dict]] = {}
    if df_all.empty:
        return laps_map
    for r in df_all.itertuples(index=False):
        lap_i = int(r.lap)
        laps_map.setdefault(lap_i, []).append({
            "driver": str(r.driver),
            "position": int(r.position),
            "compound": str(r.compound),
            "tyre_life": int(r.tyre_life),
            "gap_to_leader": float(r.gap_to_leader),
            "lap_time_sec": float(r.lap_time_sec),
            "track_status": str(track_status_override) if track_status_override else str(r.track_status),
        })
    return laps_map


tyre_cache, engineer, circuits_db, DB_PATH = load_core_engines()

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "selected_season" not in st.session_state:
    st.session_state.selected_season = 2024
if "selected_circuit" not in st.session_state:
    st.session_state.selected_circuit = "Bahrain"
if "selected_session" not in st.session_state:
    st.session_state.selected_session = "Race"
if "current_lap" not in st.session_state:
    st.session_state.current_lap = 17
if "p1_current_lap" not in st.session_state:
    st.session_state.p1_current_lap = int(st.session_state.current_lap)
if "lap_progress_pct" not in st.session_state:
    st.session_state.lap_progress_pct = 35
if "sub_lap_progress" not in st.session_state:
    st.session_state.sub_lap_progress = round(float(st.session_state.lap_progress_pct) / 100.0, 4)
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "playback_speed_label" not in st.session_state:
    st.session_state.playback_speed_label = "15x Race Pace"
if "p2_is_playing" not in st.session_state:
    st.session_state.p2_is_playing = False
if "p2_sub_lap_progress" not in st.session_state:
    st.session_state.p2_sub_lap_progress = 0.50
if "env_override_active" not in st.session_state:
    st.session_state.env_override_active = False
if "track_status_override" not in st.session_state:
    st.session_state.track_status_override = "GREEN"
if "track_temp_override" not in st.session_state:
    st.session_state.track_temp_override = 32.0
if "weather_condition" not in st.session_state:
    st.session_state.weather_condition = "Dry (Clear)"
if "radio_history" not in st.session_state:
    st.session_state.radio_history = []
if "p2_radio_history" not in st.session_state:
    st.session_state.p2_radio_history = []
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False
if "custom_scenario_rows" not in st.session_state:
    st.session_state.custom_scenario_rows = [
        {"pos": 1, "driver": "VER", "compound": "SOFT", "tyre_life": 18, "interval_sec": 0.0, "base_lap_sec": 94.2},
        {"pos": 2, "driver": "HAM", "compound": "HARD", "tyre_life": 6, "interval_sec": 2.4, "base_lap_sec": 93.8},
        {"pos": 3, "driver": "NOR", "compound": "MEDIUM", "tyre_life": 10, "interval_sec": 1.1, "base_lap_sec": 93.5},
        {"pos": 4, "driver": "LEC", "compound": "SOFT", "tyre_life": 17, "interval_sec": 1.8, "base_lap_sec": 94.3},
        {"pos": 5, "driver": "SAI", "compound": "HARD", "tyre_life": 7, "interval_sec": 2.2, "base_lap_sec": 93.9},
        {"pos": 6, "driver": "RUS", "compound": "MEDIUM", "tyre_life": 12, "interval_sec": 3.5, "base_lap_sec": 94.1},
        {"pos": 7, "driver": "PIA", "compound": "HARD", "tyre_life": 8, "interval_sec": 2.7, "base_lap_sec": 94.0},
        {"pos": 8, "driver": "ALO", "compound": "SOFT", "tyre_life": 19, "interval_sec": 4.2, "base_lap_sec": 94.7},
    ]


# =====================================================================
# SHARED VISUALIZATION & HTML HELPERS (ZERO PYARROW DEPENDENCY)
# =====================================================================
PLAYBACK_SPEED_MULTIPLIERS: dict[str, float] = {
    "1x Real-Time": 1.0,
    "5x Fast": 5.0,
    "15x Race Pace": 15.0,
    "30x High-Speed": 30.0,
}


def compute_dynamic_standings_at_sub_lap(
    df_standings: pd.DataFrame,
    circuit: str,
    sub_lap_progress: float,
) -> pd.DataFrame:
    """
    Vectorized sub-lap modulator: updates each car's live speed and sub-lap gap breathing
    at `sub_lap_progress` (0.0 .. 1.0) using cached circuit curvature (zero np.gradient per tick).
    """
    if df_standings.empty:
        return df_standings
    df_live = df_standings.copy()
    x_track, _, norm_curv = get_circuit_curvature_profile(circuit, n_points=240)
    n_pts = len(x_track)

    meta = get_circuit_metadata(circuit)
    base_lap_sec = float(meta["base_lap_sec"])
    lap_scale = max(60.0, base_lap_sec)
    leader_frac = float(sub_lap_progress) % 1.0
    t_status = str(df_live["track_status"].iloc[0]) if "track_status" in df_live.columns else "GREEN"
    status_cap = 1.0 if t_status == "GREEN" else (0.72 if t_status == "VSC" else 0.58)

    positions = df_live["position"].to_numpy(dtype=np.int64)
    drivers = df_live["driver"].astype(str).tolist()
    base_gaps = df_live["gap_to_leader"].to_numpy(dtype=np.float64)
    ages = df_live["tyre_life"].to_numpy(dtype=np.float64)
    laps = df_live["lap"].to_numpy(dtype=np.float64)

    car_fracs = (leader_frac - (base_gaps / lap_scale)) % 1.0
    pt_indices = np.rint(car_fracs * (n_pts - 1)).astype(np.intp) % n_pts
    curv_here = norm_curv[pt_indices]

    pace_offsets = np.array(
        [DRIVER_METADATA_MAP.get(d, F1_2024_DRIVERS[0])["pace_offset"] for d in drivers],
        dtype=np.float64,
    )
    pace_factors = 1.0 - (pace_offsets * 0.008)
    deg_penalties = np.minimum(0.15, ages * 0.0025)
    inst_speeds = np.clip(
        (335.0 - 235.0 * (curv_here ** 0.75)) * pace_factors * (1.0 - deg_penalties) * status_cap,
        75.0,
        345.0,
    )

    phases = 2.0 * np.pi * leader_frac + positions * 0.55
    jitters = 0.045 * np.sin(phases) + 0.025 * (0.5 - curv_here)
    dyn_gaps = np.where(positions == 1, 0.0, np.maximum(0.05 * positions, base_gaps + jitters))
    dyn_gaps = np.round(dyn_gaps, 3)

    elapsed_race_secs = np.maximum(0.0, (laps - 1.0 + leader_frac) * base_lap_sec + dyn_gaps)
    new_ts = [
        f"{int(es // 3600):02d}:{int((es % 3600) // 60):02d}:{es % 60:06.3f}"
        for es in elapsed_race_secs
    ]

    df_live["speed_kmh"] = np.round(inst_speeds, 1)
    df_live["gap_to_leader"] = dyn_gaps
    df_live["lap_timestamp"] = new_ts
    return df_live


def render_wdc_standings_html(wdc_df: pd.DataFrame, ego_driver: str) -> str:
    """Renders an F1 Broadcast World Drivers' Championship (WDC) standings table with live projected points."""
    rows_html = []
    for _, r in wdc_df.iterrows():
        rank = int(r["rank"])
        drv = str(r["driver"])
        name = str(r["name"])
        team = str(r["team"])
        color = str(r["color"])
        race_pos = int(r["race_position"])
        prior_pts = int(r["prior_points"])
        live_pts = int(r["live_race_points"])
        tot_pts = int(r["total_points"])
        gap = int(r["gap_to_leader"])
        wins = int(r["wins"])
        podiums = int(r["podiums"])

        is_ego = drv == ego_driver
        row_bg = "rgba(0, 229, 255, 0.18)" if is_ego else "rgba(255, 255, 255, 0.03)"
        border_left = f"4px solid {'#00FF66' if is_ego else color}"
        live_badge = (
            f"<span style='background:#0f5132; color:#75b798; padding:2px 7px; border-radius:10px; font-weight:bold; font-size:11px;'>+{live_pts} PTS (P{race_pos})</span>"
            if live_pts > 0
            else f"<span style='color:#6e7681; font-size:11px;'>+0 (P{race_pos})</span>"
        )
        gap_str = "LEADER" if rank == 1 else f"-{gap} pts"

        rows_html.append(
            f"<tr style='background:{row_bg}; border-left:{border_left}; border-bottom:1px solid #2a2e39;'>"
            f"<td style='padding:6px 8px; font-weight:bold; color:#ffd700;'>P{rank}</td>"
            f"<td style='padding:6px 8px; font-weight:800; color:#fff;'>{drv} <span style='color:#ccc; font-weight:normal;'>({name})</span></td>"
            f"<td style='padding:6px 8px;'><span style='display:inline-block; width:10px; height:10px; background:{color}; border-radius:2px; margin-right:6px;'></span><span style='color:#bbb;'>{team}</span></td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#9ecbff;'>{prior_pts}</td>"
            f"<td style='padding:6px 8px; font-family:monospace;'>{live_badge}</td>"
            f"<td style='padding:6px 8px; font-family:monospace; font-weight:900; color:#00ff88; font-size:14px;'>{tot_pts} PTS</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#ff7b7b;'>{gap_str}</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#aaa; font-size:12px;'>{wins}W / {podiums}Pod</td>"
            f"</tr>"
        )

    return (
        "<div style='border:1px solid #2e3440; border-radius:8px; overflow-y:auto; max-height:380px;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:13px; text-align:left;'>"
        "<thead><tr style='background:#1a1d24; color:#8b949e; border-bottom:2px solid #30363d; position:sticky; top:0;'>"
        "<th style='padding:8px;'>Rank</th><th style='padding:8px;'>Driver</th><th style='padding:8px;'>Constructor</th>"
        "<th style='padding:8px;'>Prior Pts</th><th style='padding:8px;'>Live Race Proj</th><th style='padding:8px;'>Total WDC Pts</th>"
        "<th style='padding:8px;'>Gap to P1</th><th style='padding:8px;'>Wins / Podiums</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def render_wcc_standings_html(wcc_df: pd.DataFrame, ego_team: str) -> str:
    """Renders an F1 Broadcast World Constructors' Championship (WCC) standings table with team badges & live points."""
    rows_html = []
    for _, r in wcc_df.iterrows():
        rank = int(r["rank"])
        team = str(r["team"])
        color = str(r["color"])
        drivers_str = str(r["drivers_str"])
        prior_pts = int(r["prior_points"])
        live_pts = int(r["live_race_points"])
        tot_pts = int(r["total_points"])
        gap = int(r["gap_to_leader"])
        wins = int(r["wins"])
        podiums = int(r["podiums"])

        is_ego_team = team == ego_team
        row_bg = "rgba(0, 229, 255, 0.16)" if is_ego_team else "rgba(255, 255, 255, 0.03)"
        border_left = f"5px solid {color}"
        live_badge = (
            f"<span style='background:#0f5132; color:#75b798; padding:2px 7px; border-radius:10px; font-weight:bold; font-size:11px;'>+{live_pts} PTS</span>"
            if live_pts > 0
            else "<span style='color:#6e7681; font-size:11px;'>+0 PTS</span>"
        )
        gap_str = "LEADER" if rank == 1 else f"-{gap} pts"

        rows_html.append(
            f"<tr style='background:{row_bg}; border-left:{border_left}; border-bottom:1px solid #2a2e39;'>"
            f"<td style='padding:7px 8px; font-weight:bold; color:#ffd700;'>P{rank}</td>"
            f"<td style='padding:7px 8px; font-weight:800; color:#fff;'><span style='display:inline-block; width:12px; height:12px; background:{color}; border-radius:3px; margin-right:8px;'></span>{team}</td>"
            f"<td style='padding:7px 8px; color:#ccc; font-family:monospace;'>{drivers_str}</td>"
            f"<td style='padding:7px 8px; font-family:monospace; color:#9ecbff;'>{prior_pts}</td>"
            f"<td style='padding:7px 8px; font-family:monospace;'>{live_badge}</td>"
            f"<td style='padding:7px 8px; font-family:monospace; font-weight:900; color:#00ff88; font-size:14px;'>{tot_pts} PTS</td>"
            f"<td style='padding:7px 8px; font-family:monospace; color:#ff7b7b;'>{gap_str}</td>"
            f"<td style='padding:7px 8px; font-family:monospace; color:#aaa; font-size:12px;'>{wins}W / {podiums}Pod</td>"
            f"</tr>"
        )

    return (
        "<div style='border:1px solid #2e3440; border-radius:8px; overflow-y:auto; max-height:380px;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:13px; text-align:left;'>"
        "<thead><tr style='background:#1a1d24; color:#8b949e; border-bottom:2px solid #30363d; position:sticky; top:0;'>"
        "<th style='padding:8px;'>Rank</th><th style='padding:8px;'>Constructor / Team</th><th style='padding:8px;'>Drivers</th>"
        "<th style='padding:8px;'>Prior Pts</th><th style='padding:8px;'>Live Race Proj</th><th style='padding:8px;'>Total WCC Pts</th>"
        "<th style='padding:8px;'>Gap to P1</th><th style='padding:8px;'>Wins / Podiums</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def render_telemetry_cards_grid_html(
    ego_driver: str,
    ego_pos: int,
    ego_compound: str,
    ego_tyre_life: int,
    frag_lap: int,
    max_session_lap: int,
    live_sensor: pd.Series,
    current_deg: float,
    future_5lap_loss: float,
    mc_res: dict,
    uc_res: dict,
    target_tyre: str,
) -> str:
    """
    Renders a sleek 3x2 dark-glass F1 Pit Wall Telemetry & Strategy Card Grid
    with color-coded accent borders, mini progress bars, and zero text truncation.
    """
    comp_colors = {
        "SOFT": ("#ff2800", "#ffffff"),
        "MEDIUM": ("#ffd700", "#111111"),
        "HARD": ("#e6edf3", "#111111"),
        "INTERMEDIATE": ("#39b54a", "#ffffff"),
        "WET": ("#0072bc", "#ffffff"),
    }
    comp_bg, comp_fg = comp_colors.get(ego_compound.upper(), ("#00e5ff", "#111111"))

    spd = float(live_sensor["Speed"])
    gear = int(live_sensor["nGear"])
    rpm = int(live_sensor["RPM"])
    drs_open = bool(int(live_sensor["DRS"]))
    thr = float(live_sensor["Throttle"])
    brk = float(live_sensor["Brake"])
    cursor_s = float(live_sensor["TimeSec"])

    deg_col = "#ff4d4f" if current_deg >= 2.15 else ("#ffa94d" if current_deg >= 1.2 else "#00ff88")
    deg_pct = int(min(100, max(5, (current_deg / 3.5) * 100)))
    clean_pct = float(mc_res["p_clean_air"]) * 100.0
    uc_prob = float(uc_res["win_probability"]) * 100.0
    uc_margin = float(uc_res["median_margin_sec"])
    uc_col = "#00ff88" if uc_prob >= 55.0 else ("#ffd700" if uc_prob >= 25.0 else "#ff6b6b")

    drs_pill = (
        "<span style='background:#0f5132; color:#00ff88; border:1px solid #00ff88; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:800;'>DRS OPEN</span>"
        if drs_open
        else "<span style='background:#1f2736; color:#8b949e; border:1px solid #30363d; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700;'>DRS CLOSED</span>"
    )

    card_base = (
        "background:linear-gradient(180deg, #141a26 0%, #0f141e 100%); "
        "border:1px solid #263042; border-radius:10px; padding:12px 14px; "
        "display:flex; flex-direction:column; justify-content:space-between; gap:6px;"
    )
    lbl_css = "font-size:11px; font-weight:800; letter-spacing:0.6px; color:#8b949e; text-transform:uppercase;"
    val_css = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:20px; font-weight:900; color:#f0f6fc; line-height:1.15;"
    sub_css = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; color:#9ecbff;"

    return (
        "<div style='display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; margin:4px 0 8px 0;'>"
        # Card 1: Position & Tyre
        f"<div style='{card_base} border-left:4px solid {comp_bg};'>"
        f"  <div style='{lbl_css}'>🏎️ {ego_driver} Position &amp; Tyre</div>"
        f"  <div style='display:flex; align-items:center; justify-content:space-between;'>"
        f"    <span style='{val_css}'>P{ego_pos} · {ego_compound}</span>"
        f"    <span style='background:{comp_bg}; color:{comp_fg}; width:26px; height:26px; line-height:26px; text-align:center; border-radius:50%; font-weight:900; font-size:13px;'>{ego_compound[0]}</span>"
        f"  </div>"
        f"  <div style='{sub_css}'>{ego_tyre_life} Laps Old &nbsp;·&nbsp; Lap {frag_lap}/{max_session_lap}</div>"
        f"</div>"
        # Card 2: Live Car Sensors
        f"<div style='{card_base} border-left:4px solid #00e5ff;'>"
        f"  <div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"    <span style='{lbl_css}'>⚡ Live Car Sensors</span>{drs_pill}"
        f"  </div>"
        f"  <div style='{val_css}'>{spd:.0f} km/h <span style='color:#ffd700;'>| Gear {gear}</span></div>"
        f"  <div style='{sub_css}'>{rpm:,} RPM</div>"
        f"</div>"
        # Card 3: Pedals & Lap Cursor
        f"<div style='{card_base} border-left:4px solid #00ff66;'>"
        f"  <div style='{lbl_css}'>🦶 Pedals &amp; Lap Cursor</div>"
        f"  <div style='display:flex; flex-direction:column; gap:5px; margin-top:2px;'>"
        f"    <div style='display:flex; align-items:center; gap:8px; font-family:monospace; font-size:12px;'>"
        f"      <span style='color:#00ff66; width:34px; font-weight:800;'>THR</span>"
        f"      <div style='flex:1; height:7px; background:#1f2736; border-radius:4px; overflow:hidden;'><div style='width:{thr:.0f}%; height:100%; background:#00ff66;'></div></div>"
        f"      <span style='color:#e6edf3; width:36px; text-align:right; font-weight:700;'>{thr:.0f}%</span>"
        f"    </div>"
        f"    <div style='display:flex; align-items:center; gap:8px; font-family:monospace; font-size:12px;'>"
        f"      <span style='color:#ff4d4f; width:34px; font-weight:800;'>BRK</span>"
        f"      <div style='flex:1; height:7px; background:#1f2736; border-radius:4px; overflow:hidden;'><div style='width:{brk:.0f}%; height:100%; background:#ff4d4f;'></div></div>"
        f"      <span style='color:#e6edf3; width:36px; text-align:right; font-weight:700;'>{brk:.0f}%</span>"
        f"    </div>"
        f"  </div>"
        f"  <div style='{sub_css}'>Lap Time Cursor: <b>{cursor_s:.2f}s</b></div>"
        f"</div>"
        # Card 4: L1 Tensor Tyre Wear
        f"<div style='{card_base} border-left:4px solid {deg_col};'>"
        f"  <div style='{lbl_css}'>🧊 L1 Tensor Tyre Wear</div>"
        f"  <div style='{val_css} color:{deg_col};'>+{current_deg:.2f} s/lap</div>"
        f"  <div style='height:6px; background:#1f2736; border-radius:4px; overflow:hidden;'><div style='width:{deg_pct}%; height:100%; background:{deg_col};'></div></div>"
        f"  <div style='{sub_css}'>Projected Next 5 Laps: <b style='color:{deg_col};'>+{future_5lap_loss:.1f}s</b></div>"
        f"</div>"
        # Card 5: 10k MC Pit Rejoin
        f"<div style='{card_base} border-left:4px solid #38bdf8;'>"
        f"  <div style='{lbl_css}'>⏱️ 10k MC Pit Rejoin</div>"
        f"  <div style='{val_css}'>Rejoin P{mc_res['median_pos']} <span style='color:#38bdf8; font-size:15px;'>({mc_res['median_pit_loss']:.1f}s loss)</span></div>"
        f"  <div style='height:6px; background:#1f2736; border-radius:4px; overflow:hidden;'><div style='width:{clean_pct:.0f}%; height:100%; background:#38bdf8;'></div></div>"
        f"  <div style='{sub_css}'><b>{clean_pct:.1f}% Clean Air</b> &nbsp;·&nbsp; P95: {mc_res['p95_pit_loss']:.1f}s</div>"
        f"</div>"
        # Card 6: Undercut Target
        f"<div style='{card_base} border-left:4px solid #c084fc;'>"
        f"  <div style='{lbl_css}'>⚔️ Undercut → {target_tyre}</div>"
        f"  <div style='{val_css} color:{uc_col};'>{uc_prob:.1f}% Win Prob</div>"
        f"  <div style='height:6px; background:#1f2736; border-radius:4px; overflow:hidden;'><div style='width:{min(100, max(3, uc_prob)):.0f}%; height:100%; background:{uc_col};'></div></div>"
        f"  <div style='{sub_css}'>Median Margin: <b style='color:{uc_col};'>{uc_margin:+.2f}s</b></div>"
        f"</div>"
        "</div>"
    )


def render_f1_timing_tower_html(
    df: pd.DataFrame,
    ego_driver: str,
    track_temp: float,
    weather_condition: str = "Dry (Clear)",
) -> str:
    """Renders an F1 TV Broadcast-style HTML Timing Tower with timestamps & sensor readouts without needing pyarrow."""
    badge_colors = {
        "SOFT": ("#ff2800", "#ffffff"),
        "MEDIUM": ("#ffd700", "#111111"),
        "HARD": ("#f0f0f0", "#111111"),
        "INTERMEDIATE": ("#39b54a", "#ffffff"),
        "WET": ("#0072bc", "#ffffff"),
    }
    rows_html = []
    prev_gap = 0.0
    for _, r in df.iterrows():
        drv = str(r["driver"])
        pos = int(r["position"])
        comp = str(r["compound"]).upper()
        age = int(r["tyre_life"])
        gap = float(r["gap_to_leader"])
        interval = 0.0 if pos == 1 else max(0.0, gap - prev_gap)
        prev_gap = gap
        ltime = float(r["lap_time_sec"])
        spd = float(r.get("speed_kmh", 312.0))
        lap_num = int(r["lap"])
        ts_str = str(r.get("lap_timestamp", f"00:{lap_num*1:02d}:{ltime:05.2f}"))
        deg = tyre_cache.lookup_degradation(
            comp, track_temp, age, race_lap=lap_num, weather_condition=weather_condition
        )
        bg_badge, fg_badge = badge_colors.get(comp, ("#00e5ff", "#111111"))
        d_meta = DRIVER_METADATA_MAP.get(drv, {"color": "#888888", "team": "F1"})

        is_ego = drv == ego_driver
        row_bg = "rgba(0, 229, 255, 0.18)" if is_ego else "rgba(255, 255, 255, 0.03)"
        border_left = f"4px solid {'#00FF66' if is_ego else d_meta['color']}"
        gap_str = "LEADER" if pos == 1 else f"+{gap:.3f}s"
        int_str = "—" if pos == 1 else f"+{interval:.3f}s"

        rows_html.append(
            f"<tr style='background:{row_bg}; border-left:{border_left}; border-bottom:1px solid #2a2e39;'>"
            f"<td style='padding:6px 8px; font-weight:bold; color:#aaa;'>P{pos}</td>"
            f"<td style='padding:6px 8px; font-weight:800; color:#fff;'>{drv} <span style='color:#888; font-size:11px; font-weight:normal;'>{d_meta['team'][:10]}</span></td>"
            f"<td style='padding:6px 8px;'><span style='background:{bg_badge}; color:{fg_badge}; padding:2px 7px; border-radius:10px; font-weight:bold; font-size:11px;'>{comp[0]}</span> <span style='color:#ccc; font-size:12px;'>{age}L</span></td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#00ff88;'>{gap_str}</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#9ecbff;'>{int_str}</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#ddd;'>{ltime:.3f}s</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#ffd700;'>{spd:.0f} km/h</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#ff7b7b; font-weight:bold;'>+{deg:.2f}s/L</td>"
            f"<td style='padding:6px 8px; font-family:monospace; color:#8b949e; font-size:12px;'>{ts_str}</td>"
            f"</tr>"
        )

    return (
        "<div style='border:1px solid #2e3440; border-radius:8px; overflow-y:auto; max-height:420px;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:13px; text-align:left;'>"
        "<thead><tr style='background:#1a1d24; color:#8b949e; border-bottom:2px solid #30363d; position:sticky; top:0;'>"
        "<th style='padding:8px;'>Pos</th><th style='padding:8px;'>Driver</th><th style='padding:8px;'>Tyre</th>"
        "<th style='padding:8px;'>Gap</th><th style='padding:8px;'>Interval</th><th style='padding:8px;'>Last Lap</th>"
        "<th style='padding:8px;'>Speed</th><th style='padding:8px;'>L1 Deg</th><th style='padding:8px;'>Timestamp</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def _format_radio_reply_html(reply_text: str, user_query: str = "") -> tuple[str, str, str]:
    """
    Converts the LangGraph Race Engineer markdown reply into clean F1 Pit Wall Comms HTML
    (replacing raw monospace st.code dumps) and extracts a hero strategy callout pill.
    Returns `(verdict_label, verdict_color, body_html)`.
    """
    upper_txt = reply_text.upper()
    q_low = (user_query or "").lower()
    asked_pit = bool(re.search(r"\b(?:box|pit|pitting|undercut|overcut|strategy|stop|soft|medium|hard|inter|wet)\b", q_low))
    is_pure_battle = ("RIVAL BATTLE" in upper_txt) and not asked_pit

    if is_pure_battle:
        verdict_label, verdict_color = "⚔️ TACTICAL BATTLE BRIEF", "#FFB300"
    elif "SKIP " in upper_txt and "BOX STRAIGHT FOR" in upper_txt:
        verdict_label, verdict_color = "⚠️ SKIP COMPOUND · SWITCH TO SLICKS", "#FFB300"
    elif "WEATHER CROSSOVER" in upper_txt or "BOX THIS LAP FOR WET" in upper_txt or "BOX THIS LAP FOR INTERMEDIATE" in upper_txt:
        verdict_label, verdict_color = "🌧️ WEATHER CROSSOVER · BOX NOW", "#00E5FF"
    elif "STAY OUT" in upper_txt or "DO NOT BOX" in upper_txt:
        verdict_label, verdict_color = "🟢 PIT CALL · STAY OUT", "#00FF88"
    elif "BOX THIS LAP" in upper_txt or "BOX BOX" in upper_txt or "BOX FOR UNDERCUT" in upper_txt or "UNDERCUT WINDOW OPEN" in upper_txt:
        verdict_label, verdict_color = "🚨 PIT CALL · BOX THIS LAP", "#FF4B4B"
    elif "PUSH TO PASS" in upper_txt or "CATCH" in upper_txt or "RIVAL BATTLE" in upper_txt:
        verdict_label, verdict_color = "⚔️ TACTICAL BATTLE BRIEF", "#FFB300"
    elif "FIA" in upper_txt or "REGULATION" in upper_txt or "PENALTY" in upper_txt:
        verdict_label, verdict_color = "⚖️ FIA REGULATIONS BRIEF", "#A78BFA"
    else:
        verdict_label, verdict_color = "🎧 PIT WALL TELEMETRY BRIEF", "#38BDF8"

    lines_out = []
    for raw_line in reply_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#f8fafc;'>\1</b>", esc)
        esc = re.sub(
            r"`([^`]+)`",
            r"<code style='background:#0b1220;color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:11px;'>\1</code>",
            esc,
        )
        if esc.startswith("- ") or esc.startswith("• "):
            esc = esc[2:]
            lines_out.append(
                f"<div style='display:flex;gap:6px;margin:4px 0;line-height:1.45;'>"
                f"<span style='color:{verdict_color};font-weight:900;'>▸</span>"
                f"<span>{esc}</span></div>"
            )
        else:
            lines_out.append(f"<div style='margin:4px 0;line-height:1.45;'>{esc}</div>")

    return verdict_label, verdict_color, "".join(lines_out)


@st.fragment
def render_pit_wall_radio_console(
    *,
    key_prefix: str,
    history_state_key: str,
    ego_driver: str,
    current_lap: int,
    max_laps: int,
    track_status: str,
    target_tyre: str,
    build_context_fn,
) -> None:
    """
    Renders the signature docked F1 Pit Wall Radio Comms Console inside an isolated `@st.fragment`:
    - Clicking Quick-Call chips or SEND reruns ONLY this radio console (~40-80ms), never reloading the 60 FPS Canvas!
    - Shows a crisp, human F1 Race Engineer radio call up front with an instant 0ms HTML `<details>`
      Clarification dropdown inside each bubble for full telemetry, 10k MC, strategy ranking & FIA legal proof.
    """
    d_meta = DRIVER_METADATA_MAP.get(
        ego_driver,
        {"name": ego_driver, "team": "F1 Racing", "number": 1, "color": "#00E5FF"},
    )
    team_color = str(d_meta.get("color", "#00E5FF"))
    drv_num = int(d_meta.get("number", 0))
    team_name = str(d_meta.get("team", "F1 Racing"))
    status_color = "#00FF88" if track_status == "GREEN" else ("#FFD600" if track_status == "VSC" else "#FF4B4B")

    # Header styled like an official F1 Team Radio broadcast graphic + 0ms native Clarification drawer styles
    st.markdown(
        f"""
        <style>
        @keyframes f1Pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.35; transform: scale(0.82); }}
        }}
        @keyframes f1Wave {{
            0%, 100% {{ transform: scaleY(0.35); }}
            50% {{ transform: scaleY(1.0); }}
        }}
        .f1-radio-hdr {{
            background: linear-gradient(135deg, #111827 0%, #0b101a 100%);
            border: 1px solid #263145;
            border-left: 4px solid {team_color};
            border-radius: 10px;
            padding: 9px 12px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }}
        .f1-radio-feed {{
            background: linear-gradient(180deg, #0b1019 0%, #0e1420 100%);
            border: 1px solid #232d40;
            border-radius: 10px;
            padding: 10px;
            height: 326px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .f1-wave-bars {{
            display: inline-flex;
            align-items: center;
            gap: 2px;
            height: 16px;
        }}
        .f1-wave-bars span {{
            width: 2.5px;
            height: 14px;
            border-radius: 2px;
            background: {team_color};
            display: inline-block;
            animation: f1Wave 1.1s ease-in-out infinite;
        }}
        .f1-wave-bars span:nth-child(2) {{ animation-delay: 0.18s; }}
        .f1-wave-bars span:nth-child(3) {{ animation-delay: 0.36s; }}
        .f1-wave-bars span:nth-child(4) {{ animation-delay: 0.54s; }}
        .f1-clarify-drawer {{
            margin-top: 8px;
            border-top: 1px dashed #24344d;
            padding-top: 6px;
        }}
        .f1-clarify-drawer > summary {{
            list-style: none;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid #334155;
            color: #93c5fd;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 700;
            user-select: none;
            transition: background 0.15s ease, border-color 0.15s ease;
        }}
        .f1-clarify-drawer > summary::-webkit-details-marker {{
            display: none;
        }}
        .f1-clarify-drawer > summary:hover {{
            background: rgba(51, 65, 85, 0.95);
            border-color: #38bdf8;
            color: #e0f2fe;
        }}
        .f1-clarify-drawer > summary::before {{
            content: "▸";
            display: inline-block;
            transition: transform 0.15s ease;
            color: #38bdf8;
        }}
        .f1-clarify-drawer[open] > summary::before {{
            transform: rotate(90deg);
        }}
        .f1-clarify-body {{
            margin-top: 7px;
            background: rgba(9, 14, 24, 0.88);
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 8px 9px;
            font-size: 11px;
            color: #cbd5e1;
        }}
        .f1-strat-rank-box {{
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid #1e293b;
        }}
        </style>
        <div class="f1-radio-hdr">
            <div style="min-width:0;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:#ff2800;box-shadow:0 0 8px #ff2800;display:inline-block;animation:f1Pulse 1.4s infinite;"></span>
                    <span style="font-size:10.5px;font-weight:900;letter-spacing:1.1px;color:#ff5252;">PIT WALL RADIO COMMS</span>
                </div>
                <div style="font-size:14px;font-weight:800;color:#f8fafc;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    📻 {html.escape(ego_driver)} #{drv_num} <span style="color:#94a3b8;font-weight:600;font-size:12px;">· {html.escape(team_name)}</span>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                <span style="background:rgba(15,23,42,0.9);border:1px solid {status_color};color:{status_color};padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:800;font-family:monospace;">
                    L{current_lap}/{max_laps} · {html.escape(track_status)}
                </span>
                <div class="f1-wave-bars" title="Encrypted Team Radio Channel">
                    <span></span><span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 One-Click Tactical Quick-Call Chips (2x2 grid)
    pending_q: str | None = None
    qc1, qc2 = st.columns(2, gap="small")
    if qc1.button("⚔️ Catch Rival?", key=f"{key_prefix}_qc_catch", use_container_width=True):
        pending_q = "Can I catch the car ahead or should we box to undercut?"
    if qc2.button(f"📦 Box → {target_tyre}?", key=f"{key_prefix}_qc_box", use_container_width=True):
        pending_q = f"Should we box for {target_tyre}s right now?"

    qc3, qc4 = st.columns(2, gap="small")
    if qc3.button("🛞 Tyre Wear & Cliff", key=f"{key_prefix}_qc_wear", use_container_width=True):
        pending_q = f"How much time am I losing per lap on these tyres right now, and should we undercut onto {target_tyre}s?"
    if qc4.button("⚖️ FIA SC Rules", key=f"{key_prefix}_qc_fia", use_container_width=True):
        pending_q = "Safety Car is out — should we box now, and what is the FIA penalty if I overtake under the Safety Car?"

    # Bottom Push-To-Talk Form rendered after transcript; handle submission cleanly via fragment-scoped rerun
    form_submitted_q = st.session_state.pop(f"_{key_prefix}_pending_submit", None)
    active_question = pending_q or form_submitted_q

    if active_question and str(active_question).strip():
        with st.spinner("📡 Pit Wall Engineer analyzing telemetry, 10k MC & FIA rules..."):
            ctx = build_context_fn()
            agent_state = engineer.ask_engineer(str(active_question).strip(), context=ctx)
            agent_state = dict(agent_state)
            agent_state["_msg_id"] = f"{key_prefix}_{time.time_ns()}"
            agent_state["_user_query"] = str(active_question).strip()
            agent_state["_lap_tag"] = f"L{current_lap}/{max_laps} · {track_status}"
            agent_state["_driver_code"] = ego_driver
            agent_state["_driver_num"] = drv_num
            agent_state["_team_color"] = team_color
            history = list(st.session_state.get(history_state_key, []))
            history.insert(0, agent_state)
            st.session_state[history_state_key] = history[:15]

    history = st.session_state.get(history_state_key, [])

    # Scrollable F1 Team Radio Transcript Feed (newest transmission on top for zero-scroll visibility)
    if not history:
        feed_inner = (
            f"<div style='margin:auto;text-align:center;padding:28px 14px;color:#94a3b8;'>"
            f"<div style='font-size:28px;margin-bottom:8px;'>🎧</div>"
            f"<div style='font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:6px;'>"
            f"Channel Open — {html.escape(ego_driver)} #{drv_num} Race Engineer Ready</div>"
            f"<div style='font-size:11.5px;line-height:1.5;color:#8b949e;'>"
            f"Tap a tactical quick-call above or transmit any strategy, battle, tyre, or FIA rulebook question below."
            f"</div>"
            f"<div style='margin-top:12px;display:inline-flex;gap:6px;flex-wrap:wrap;justify-content:center;'>"
            f"<span style='background:#151d2e;border:1px solid #29354d;border-radius:999px;padding:2px 8px;font-size:10px;color:#7dd3fc;'>L1 Tensor Wear</span>"
            f"<span style='background:#151d2e;border:1px solid #29354d;border-radius:999px;padding:2px 8px;font-size:10px;color:#34d399;'>10k Monte Carlo</span>"
            f"<span style='background:#151d2e;border:1px solid #29354d;border-radius:999px;padding:2px 8px;font-size:10px;color:#c4b5fd;'>2-Hop FIA RAG</span>"
            f"</div></div>"
        )
    else:
        bubbles_html = []
        for idx_msg, entry in enumerate(history):
            q_text = str(entry.get("_user_query") or entry.get("radio_message") or "Radio check")
            drv_c = str(entry.get("_driver_code") or entry.get("driver") or ego_driver)
            drv_n = int(entry.get("_driver_num", drv_num))
            t_col = str(entry.get("_team_color") or team_color)
            lap_tag = str(entry.get("_lap_tag") or f"L{current_lap}/{max_laps}")
            guard_ok = bool(entry.get("guardrail_passed", True))
            elapsed_ms = entry.get("elapsed_ms", 0)
            reply_txt = str(entry.get("engineer_radio_reply", ""))
            human_call = str(entry.get("human_radio_call", "")).strip()
            verdict_lbl, verdict_col, body_html = _format_radio_reply_html(reply_txt, user_query=q_text)

            latest_chip = (
                "<span style='background:#1d4ed8;color:#eff6ff;padding:1px 6px;border-radius:4px;font-size:9.5px;font-weight:800;margin-left:6px;'>LATEST</span>"
                if idx_msg == 0
                else ""
            )
            guard_pill = (
                "<span style='background:rgba(0,230,118,0.14);border:1px solid #00e676;color:#69f0ae;padding:1px 6px;border-radius:999px;font-size:9.5px;font-weight:800;'>🛡️ VERIFIED</span>"
                if guard_ok
                else "<span style='background:rgba(255,75,75,0.16);border:1px solid #ff4b4b;color:#ff8a80;padding:1px 6px;border-radius:999px;font-size:9.5px;font-weight:800;'>❌ GUARDRAIL</span>"
            )

            # Driver bubble (right-aligned)
            bubbles_html.append(
                f"<div style='align-self:flex-end;max-width:92%;background:linear-gradient(135deg,#172033 0%,#121a29 100%);"
                f"border:1px solid #2b3850;border-right:3px solid {t_col};border-radius:10px 10px 2px 10px;padding:7px 10px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:3px;'>"
                f"<span style='font-size:10px;font-weight:800;color:{t_col};letter-spacing:0.5px;'>🎙️ {html.escape(drv_c)} #{drv_n} · DRIVER</span>"
                f"<span style='font-size:9.5px;color:#64748b;font-family:monospace;'>{html.escape(lap_tag)}{latest_chip}</span>"
                f"</div>"
                f"<div style='font-size:12px;color:#f1f5f9;line-height:1.4;'>“{html.escape(q_text)}”</div>"
                f"</div>"
            )

            # Optional Strategy Optimizer mini-pill showing #1 Optimal Strategy vs Requested
            strat_pill = ""
            strat_rank = (entry.get("pit_sim_data") or {}).get("strategy_ranking")
            if strat_rank and strat_rank.get("optimal"):
                opt_lbl = html.escape(str(strat_rank["optimal"]["label"]))
                delta_opt = float(strat_rank.get("optimal_beats_requested_by_sec", 0.0))
                delta_str = f" (-{delta_opt:.1f}s faster)" if delta_opt >= 0.5 else ""
                strat_pill = (
                    f"<div style='display:inline-block;margin-left:6px;background:rgba(56,189,248,0.12);"
                    f"border:1px solid #38bdf8;color:#7dd3fc;padding:2px 7px;border-radius:6px;"
                    f"font-size:10px;font-weight:800;margin-bottom:6px;'>🧠 OPTIMAL: {opt_lbl}{delta_str}</div>"
                )

            # Build Top-3 Strategy Options proof table for inside the Clarification dropdown
            strat_table_html = ""
            if strat_rank and strat_rank.get("top_options"):
                top_opts = strat_rank["top_options"][:3]
                best_time = float(top_opts[0].get("expected_remaining_time_sec", 0.0))
                rows_html = []
                for rank_i, opt_item in enumerate(top_opts, 1):
                    delta_s = float(opt_item.get("expected_remaining_time_sec", best_time)) - best_time
                    delta_badge = (
                        "<span style='color:#34d399;font-weight:800;'>BEST</span>"
                        if rank_i == 1
                        else f"<span style='color:#94a3b8;'>+{delta_s:.1f}s</span>"
                    )
                    rows_html.append(
                        f"<div style='display:flex;justify-content:space-between;gap:8px;padding:2px 0;font-size:10.5px;'>"
                        f"<span style='color:#e2e8f0;'>#{rank_i} {html.escape(str(opt_item.get('label', '')))}</span>"
                        f"<span style='font-family:monospace;white-space:nowrap;'>{delta_badge}</span></div>"
                    )
                strat_table_html = (
                    f"<div class='f1-strat-rank-box'>"
                    f"<div style='font-size:9.5px;font-weight:800;color:#7dd3fc;letter-spacing:0.6px;margin-bottom:3px;'>TOP RANKED STRATEGIES (REMAINING RACE TIME)</div>"
                    f"{''.join(rows_html)}</div>"
                )

            # Guardrail notes proof
            g_notes = entry.get("guardrail_notes") or []
            guard_notes_html = ""
            if g_notes:
                notes_str = " · ".join(html.escape(str(n)) for n in g_notes[:3])
                guard_notes_html = (
                    f"<div style='margin-top:6px;padding-top:5px;border-top:1px solid #1e293b;font-size:10px;color:#86efac;'>"
                    f"🛡️ {notes_str}</div>"
                )

            # Primary human-friendly F1 voice + instant 0ms collapsible Clarification dropdown
            if human_call:
                primary_html = (
                    f"<div style='font-size:12.6px;color:#f8fafc;line-height:1.5;font-weight:500;'>"
                    f"“{html.escape(human_call)}”</div>"
                    f"<details class='f1-clarify-drawer'>"
                    f"<summary>📊 More Clarification — Telemetry, 10k MC &amp; Strategy Proof</summary>"
                    f"<div class='f1-clarify-body'>{body_html}{strat_table_html}{guard_notes_html}</div>"
                    f"</details>"
                )
            else:
                primary_html = f"<div style='font-size:11.8px;color:#cbd5e1;'>{body_html}</div>"

            # Race Engineer Pit Wall bubble (left-aligned)
            bubbles_html.append(
                f"<div style='align-self:flex-start;max-width:98%;background:linear-gradient(135deg,#0f1826 0%,#101c2d 100%);"
                f"border:1px solid #22324a;border-left:3px solid {verdict_col};border-radius:10px 10px 10px 2px;padding:8px 10px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:5px;'>"
                f"<span style='font-size:10px;font-weight:900;color:#38bdf8;letter-spacing:0.5px;'>🎧 RACE ENGINEER · PIT WALL</span>"
                f"<div style='display:flex;align-items:center;gap:5px;'>{guard_pill}"
                f"<span style='font-size:9.5px;color:#94a3b8;font-family:monospace;'>{elapsed_ms}ms</span></div>"
                f"</div>"
                f"<div style='display:flex;flex-wrap:wrap;align-items:center;'>"
                f"<div style='display:inline-block;background:rgba(15,23,42,0.95);border:1px solid {verdict_col};color:{verdict_col};"
                f"padding:2px 8px;border-radius:6px;font-size:10.5px;font-weight:800;letter-spacing:0.4px;margin-bottom:6px;'>{verdict_lbl}</div>"
                f"{strat_pill}</div>"
                f"{primary_html}"
                f"</div>"
            )
        feed_inner = "".join(bubbles_html)

    st.markdown(f"<div class='f1-radio-feed'>{feed_inner}</div>", unsafe_allow_html=True)

    # Pit Wall Spoken Voice Bar (F1 Dual-Tone Radio Beep + UK Race Engineer Voice reading `human_radio_call`)
    voice_auto = st.toggle(
        "🔊 Pit Wall Voice (Auto-Read Radio)",
        value=True,
        key=f"{key_prefix}_voice_auto_toggle",
        help="Automatically plays the F1 Pit Wall radio beep and speaks the Race Engineer's call aloud using your browser's UK voice.",
    )
    if history:
        latest_entry = history[0]
        latest_spoken = str(
            latest_entry.get("human_radio_call")
            or latest_entry.get("engineer_radio_reply")
            or ""
        ).strip()
        latest_msg_id = str(latest_entry.get("_msg_id") or f"{key_prefix}_0_{len(history)}")
        latest_drv = str(latest_entry.get("_driver_code") or ego_driver)
        latest_col = str(latest_entry.get("_team_color") or team_color)
        components.html(
            render_f1_radio_voice_bar_html(
                speech_text=latest_spoken,
                message_id=latest_msg_id,
                auto_play=bool(voice_auto),
                driver_code=latest_drv,
                team_color=latest_col,
            ),
            height=44,
        )

    # Bottom Push-To-Talk Radio Input Bar (fragment-scoped rerun = zero full-page reload / zero 60 FPS Canvas blink!)
    with st.form(key=f"{key_prefix}_radio_ptt_form", clear_on_submit=True):
        in_c1, in_c2 = st.columns([2.45, 1.35], gap="small")
        typed_q = in_c1.text_input(
            "Driver Radio Message",
            placeholder="Radio message...",
            label_visibility="collapsed",
            key=f"{key_prefix}_radio_text_input",
        )
        send_clicked = in_c2.form_submit_button("🎙️ SEND", type="primary", use_container_width=True)
        if send_clicked and typed_q.strip():
            st.session_state[f"_{key_prefix}_pending_submit"] = typed_q.strip()
            st.rerun(scope="fragment")

    # Optional FIA Multi-Hop Evidence & Clear Radio utility row
    if history:
        latest = history[0]
        has_legal = bool(latest.get("legal_data") and latest["legal_data"].get("hop1_primary_rules"))
        if has_legal:
            with st.expander("⚖️ Inspect FIA Rulebook Evidence (Hop 1 + Hop 2)", expanded=False):
                h1 = latest["legal_data"]["hop1_primary_rules"][0]
                st.markdown(f"**📘 [Hop 1 — Primary Rule | Page {h1['page']} ({h1['score']}% Match)]:**\n> {h1['text']}")
                for h2 in latest["legal_data"].get("hop2_resolved_references", []):
                    st.markdown(
                        f"**🔗 [Hop 2 — Cross-Ref Article {h2['article']} | Page {h2['page']}]:**\n> {h2['text']}"
                    )
        if st.button("🗑️ Clear Radio Log", key=f"{key_prefix}_clear_radio_btn", use_container_width=True):
            st.session_state[history_state_key] = []
            st.rerun(scope="fragment")


def generate_driver_radio_and_team_decision(
    row: pd.Series,
    df_standings: pd.DataFrame,
    circuit: str,
    track_status: str,
    track_temp: float,
    pit_box_num: int,
    target_tyre: str,
    total_laps: int = 57,
    weather_condition: str = "Dry (Clear)",
) -> dict:
    """
    Computes the clicked driver's live radio feed, pit-wall telemetry diagnosis,
    remaining-stint pit breakeven physics, and 10,000-run Monte Carlo team strategy decision.
    """
    drv = str(row["driver"])
    pos = int(row["position"])
    comp = str(row["compound"]).upper()
    age = int(row["tyre_life"])
    gap = float(row["gap_to_leader"])
    ltime = float(row["lap_time_sec"])
    lap = int(row["lap"])
    laps_remaining = max(0, int(total_laps) - lap)
    d_meta = DRIVER_METADATA_MAP.get(drv, {"name": drv, "team": "F1 Racing", "number": 0})

    deg = tyre_cache.lookup_degradation(
        comp, track_temp, age, race_lap=lap, weather_condition=weather_condition, circuit=circuit
    )
    next_5l = sum(
        tyre_cache.lookup_degradation(
            comp, track_temp, age + k, race_lap=lap + k, weather_condition=weather_condition, circuit=circuit
        )
        for k in range(1, 6)
    )
    weather_penalty = tyre_cache.get_weather_compound_penalty(comp, weather_condition)
    weather_opt_tyre = tyre_cache.recommended_compound_for_weather(
        weather_condition=weather_condition,
        laps_remaining=laps_remaining,
        current_compound=comp,
        track_temp=track_temp,
    )
    effective_target_tyre = (
        weather_opt_tyre
        if weather_penalty >= 4.0 and tyre_cache.get_weather_compound_penalty(target_tyre, weather_condition) > 0.0
        else target_tyre
    )

    pit_sim = get_cached_pit_simulator(circuit, pit_box_num)
    base_lap_sec = float(get_circuit_metadata(circuit)["base_lap_sec"])
    traffic = [
        {
            "driver": r.driver,
            "gap_to_leader": float(r.gap_to_leader),
            "is_lapped": float(r.gap_to_leader) >= base_lap_sec,
        }
        for r in df_standings.itertuples(index=False)
        if r.driver != drv
    ]
    mc = pit_sim.simulate_rejoin_mc(
        drv,
        gap,
        traffic,
        race_condition=track_status,
        n_sims=10_000,
        silent=True,
        base_lap_sec=base_lap_sec,
    )

    car_ahead = df_standings[df_standings["position"] == pos - 1].iloc[0] if pos > 1 and not df_standings[df_standings["position"] == pos - 1].empty else None
    car_behind = df_standings[df_standings["position"] == pos + 1].iloc[0] if not df_standings[df_standings["position"] == pos + 1].empty else None

    gap_ahead = round(gap - float(car_ahead["gap_to_leader"]), 2) if car_ahead is not None else 0.0
    gap_behind = round(float(car_behind["gap_to_leader"]) - gap, 2) if car_behind is not None else 99.0

    new_tyre_deg = tyre_cache.lookup_degradation(
        effective_target_tyre, track_temp, 1, race_lap=lap + 1, weather_condition=weather_condition, circuit=circuit
    ) + tyre_cache.get_cold_track_activation_penalty(
        effective_target_tyre, track_temp, stint_lap=1, weather_condition=weather_condition
    )
    uc = pit_sim.calculate_undercut_mc(
        rival_gap=max(0.3, gap_ahead if pos > 1 else gap_behind),
        new_tyre=effective_target_tyre,
        old_tyre=comp,
        race_condition=track_status,
        n_sims=10_000,
        old_tyre_deg=deg,
        new_tyre_deg=new_tyre_deg,
        silent=True,
    )

    be = compute_remaining_stint_pit_breakeven(
        ego_compound=comp,
        ego_tyre_life=age,
        track_temp=track_temp,
        laps_remaining=laps_remaining,
        pit_loss_sec=float(mc["median_pit_loss"]),
        target_new_tyre=effective_target_tyre,
        race_lap=lap,
        weather_condition=weather_condition,
        circuit=circuit,
        p_clean_air=float(mc["p_clean_air"]),
        positions_dropped=max(0, int(mc["median_pos"]) - pos),
        race_condition=track_status,
    )
    rec_gain = be["recoverable_tyre_gain_sec"]
    net_payback = be["net_stint_pit_payback_sec"]

    # Synthesize realistic Driver -> Pit Wall Radio & Pit Wall Team Decision
    if weather_penalty >= 4.0 and net_payback > 0.0:
        driver_msg = (
            f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"Track is {weather_condition} — zero grip on {comp}s "
            f"(+{weather_penalty:.1f}s/L wet loss), we need {effective_target_tyre}s NOW!\""
        )
        team_call = (
            f"🌧️ BOX THIS LAP FOR {effective_target_tyre} (WEATHER CROSSOVER) — Running {comp}s in {weather_condition} "
            f"loses +{weather_penalty:.1f}s/lap (+{net_payback:.1f}s net stint gain). Rejoin P{mc['median_pos']}!"
        )
    elif net_payback <= 0.0 and (age >= 15 or deg >= 2.0 or track_status in ("SC", "VSC") or laps_remaining <= 12):
        driver_msg = (
            f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"Tyres are {age}L old ({comp}, +{deg:.2f}s/L deg) "
            f"with {laps_remaining} laps to go — should we box or hold track position?\""
        )
        team_call = (
            f"🛑 STAY OUT & HOLD P{pos} (DO NOT BOX) — Only {laps_remaining}L left: {mc['median_pit_loss']:.1f}s pit loss "
            f"exceeds +{rec_gain:.1f}s fresh-tyre gain (net {net_payback:+.1f}s) and would drop you P{pos}->P{mc['median_pos']}!"
        )
    elif track_status == "SC" and net_payback > 0.0:
        sc_saved = float(
            mc.get(
                "pit_loss_saved_vs_green",
                max(0.0, float(pit_sim.circuit_data["normal_loss"]) - float(mc["median_pit_loss"])),
            )
        )
        driver_msg = f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"Safety Car deployed! Tyres are {age} laps old ({comp}), are we boxing this lap?\""
        team_call = f"🚨 BOX BOX NOW FOR {target_tyre} — Cheap Safety Car stop saves {sc_saved:.1f}s (+{net_payback:.1f}s net stint gain)! Rejoin P{mc['median_pos']}."
    elif track_status == "VSC" and net_payback > 0.0:
        driver_msg = f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"VSC on my dash, maintaining positive delta. Pit window open?\""
        team_call = f"🟡 BOX UNDER VSC FOR {target_tyre} — Pit loss reduced to {mc['median_pit_loss']:.2f}s (+{net_payback:.1f}s net stint gain). Rejoin P{mc['median_pos']} ({mc['p_clean_air']*100:.0f}% Clean Air)."
    elif (age >= 18 or deg >= 2.10) and net_payback > 0.0:
        driver_msg = f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"Rear tyres are overheating, losing traction out of slow corners (+{deg:.2f}s/lap deg)!\""
        team_call = (
            f"🟠 PIT WINDOW OPEN (BOX THIS LAP FOR {target_tyre}) — Projected {laps_remaining}L fresh-tyre gain +{rec_gain:.1f}s "
            f"exceeds {mc['median_pit_loss']:.1f}s pit loss (net +{net_payback:.1f}s). Undercut win prob: {uc['win_probability']*100:.1f}%."
        )
    elif car_ahead is not None and gap_ahead < 1.2:
        driver_msg = f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"I'm right in {car_ahead['driver']}'s DRS (+{gap_ahead:.2f}s). Give me full deployment to overtake!\""
        team_call = f"🟢 PUSH TO PASS {car_ahead['driver']} (STRAT MODE 3) — You have DRS & +{deg:.2f}s/L wear. If stuck for 2 laps, we evaluate the undercut."
    else:
        driver_msg = f"📻 [{drv} #{d_meta['number']} -> {d_meta['team']}]: \"Balance feels solid in clean air. Last lap {ltime:.3f}s. What's our target pace?\""
        team_call = f"🟢 STAY OUT & EXTEND STINT — Tyres are healthy ({age}L {comp}, +{deg:.2f}s/L). Rejoin today would be P{mc['median_pos']}, hold track position."

    return {
        "driver": drv,
        "name": d_meta["name"],
        "team": d_meta["team"],
        "number": d_meta["number"],
        "lap": lap,
        "laps_remaining": laps_remaining,
        "position": pos,
        "compound": comp,
        "tyre_life": age,
        "deg": deg,
        "next_5l_loss": next_5l,
        "recoverable_tyre_gain_sec": rec_gain,
        "net_stint_pit_payback_sec": net_payback,
        "gap_ahead": gap_ahead,
        "ahead_driver": car_ahead["driver"] if car_ahead is not None else "NONE (LEADER)",
        "gap_behind": gap_behind,
        "behind_driver": car_behind["driver"] if car_behind is not None else "NONE",
        "mc_rejoin": mc,
        "mc_undercut": uc,
        "driver_radio_msg": driver_msg,
        "team_decision": team_call,
    }


@lru_cache(maxsize=128)
def _cached_driver_decision_impl(
    driver: str,
    circuit: str,
    track_status: str,
    track_temp_rounded: float,
    pit_box_num: int,
    target_tyre: str,
    total_laps: int,
    weather_condition: str,
    standings_sig: tuple[tuple, ...],
) -> dict:
    """Memoizes the 20,000-sim Monte Carlo decision bundle per canonical lap state so 4 Hz ticks stay O(1)."""
    df_sig = pd.DataFrame(
        standings_sig,
        columns=["lap", "driver", "position", "compound", "tyre_life", "gap_to_leader", "lap_time_sec"],
    )
    row = df_sig[df_sig["driver"] == driver].iloc[0]
    return generate_driver_radio_and_team_decision(
        row=row,
        df_standings=df_sig,
        circuit=circuit,
        track_status=track_status,
        track_temp=track_temp_rounded,
        pit_box_num=pit_box_num,
        target_tyre=target_tyre,
        total_laps=total_laps,
        weather_condition=weather_condition,
    )


def get_cached_driver_decision(
    driver: str,
    df_base_standings: pd.DataFrame,
    circuit: str,
    track_status: str,
    track_temp: float,
    pit_box_num: int,
    target_tyre: str,
    total_laps: int = 57,
    weather_condition: str = "Dry (Clear)",
) -> dict:
    """Builds a hashable signature from canonical lap standings and fetches/caches the MC decision bundle."""
    standings_sig = tuple(
        (
            int(r.lap),
            str(r.driver),
            int(r.position),
            str(r.compound).upper(),
            int(r.tyre_life),
            round(float(r.gap_to_leader), 3),
            round(float(r.lap_time_sec), 3),
        )
        for r in df_base_standings.itertuples(index=False)
    )
    return _cached_driver_decision_impl(
        driver=str(driver),
        circuit=str(circuit),
        track_status=str(track_status).upper(),
        track_temp_rounded=round(float(track_temp), 1),
        pit_box_num=int(pit_box_num),
        target_tyre=str(target_tyre).upper(),
        total_laps=int(total_laps),
        weather_condition=str(weather_condition),
        standings_sig=standings_sig,
    )


@lru_cache(maxsize=64)
def _cached_championship_view_impl(
    season: int,
    circuit: str,
    pos_sig: tuple[tuple[str, int], ...],
    ego_driver: str,
    ego_team_name: str,
) -> dict:
    """Caches WDC/WCC DataFrames and rendered broadcast HTML until race order or selection changes."""
    df_pos = pd.DataFrame(pos_sig, columns=["driver", "position"])
    champ_data = compute_championship_standings_till_race(
        season=season,
        circuit=circuit,
        current_race_standings_df=df_pos,
    )
    wdc_html = render_wdc_standings_html(champ_data["wdc"], ego_driver)
    wcc_html = render_wcc_standings_html(champ_data["wcc"], ego_team_name)
    return {
        "champ_data": champ_data,
        "wdc_html": wdc_html,
        "wcc_html": wcc_html,
    }


def get_cached_championship_view(
    season: int,
    circuit: str,
    df_standings: pd.DataFrame,
    ego_driver: str,
    ego_team_name: str,
) -> dict:
    pos_sig = tuple(
        (str(r.driver), int(r.position))
        for r in df_standings.itertuples(index=False)
    )
    return _cached_championship_view_impl(
        season=int(season),
        circuit=str(circuit),
        pos_sig=pos_sig,
        ego_driver=str(ego_driver),
        ego_team_name=str(ego_team_name),
    )


# =====================================================================
# TOP NAVIGATION: PAGE 1 (LIVE & HISTORICAL GP) vs PAGE 2 (CUSTOM SCENARIO BUILDER)
# =====================================================================
st.sidebar.title("🏎️ PIT WALL COMMAND")
app_mode = st.sidebar.radio(
    "🧭 Select Command Center Mode:",
    [
        "📡 Page 1: Live & Historical Grand Prix Replay",
        "🧪 Page 2: 100% Custom Scenario Builder",
    ],
    index=0,
)

st.sidebar.markdown("---")

# =====================================================================
# PAGE 1: LIVE & HISTORICAL GRAND PRIX REPLAY + CIRCUIT GRAPHICS + DRIVER INSPECTOR
# =====================================================================
if app_mode.startswith("📡 Page 1"):
    st.sidebar.subheader("🌍 Season, Grand Prix & Session Selector")
    selected_season = st.sidebar.selectbox(
        "📅 Select F1 Season:",
        SUPPORTED_SEASONS,
        index=SUPPORTED_SEASONS.index(st.session_state.selected_season) if st.session_state.selected_season in SUPPORTED_SEASONS else 0,
    )
    calendar_names = [f"R{r['round']:02d}: {r['circuit']} ({r['race_name']})" for r in F1_2024_CALENDAR]
    circuit_keys = [r["circuit"] for r in F1_2024_CALENDAR]

    curr_idx = circuit_keys.index(st.session_state.selected_circuit) if st.session_state.selected_circuit in circuit_keys else 0
    chosen_cal_idx = st.sidebar.selectbox(
        f"🏁 Select {selected_season} Grand Prix (All 24 Races):",
        range(len(calendar_names)),
        format_func=lambda i: calendar_names[i],
        index=curr_idx,
    )
    selected_circuit = circuit_keys[chosen_cal_idx]
    circuit_meta = F1_2024_CALENDAR[chosen_cal_idx]

    selected_session = st.sidebar.selectbox(
        "⏱️ Select Session Type (Practice 1-3, Qualifying, Sprint, Race):",
        SESSION_TYPES,
        format_func=lambda s: SESSION_DISPLAY_LABELS.get(s, s),
        index=SESSION_TYPES.index(st.session_state.selected_session) if st.session_state.selected_session in SESSION_TYPES else 5,
    )

    # Auto-fetch authentic FastF1 timing by default in interactive sessions (offline-safe during pytest)
    _default_auto_ff1 = not bool(os.environ.get("PYTEST_CURRENT_TEST"))
    auto_fetch_real = st.sidebar.checkbox(
        "Auto-Fetch Real FastF1 Data on Race Switch",
        value=st.session_state.get("auto_fetch_real_fastf1", _default_auto_ff1),
        key="auto_fetch_real_fastf1",
        help="Automatically downloads authentic FIA lap-by-lap timing & tyre stints from FastF1 when opening an uncached session (~15s once, then saved permanently in DuckDB).",
    )

    # Ensure selected Season, Grand Prix & Session are populated in DuckDB
    _pre_src = get_session_data_source(selected_season, selected_circuit, selected_session, db_path=DB_PATH)
    if auto_fetch_real and _pre_src not in ("FASTF1_REAL", "OPENF1_REAL"):
        with st.sidebar.status(
            f"Fetching real FIA timing (FastF1 / OpenF1) for {selected_season} {selected_circuit} {selected_session}...",
            expanded=False,
        ):
            available_laps = load_or_populate_gp_session(
                selected_circuit,
                selected_session,
                season=selected_season,
                db_path=DB_PATH,
                allow_auto_fastf1=True,
            )
            get_cached_session_laps_map.cache_clear()
    else:
        available_laps = load_or_populate_gp_session(
            selected_circuit,
            selected_session,
            season=selected_season,
            db_path=DB_PATH,
            allow_auto_fastf1=False,
        )
    max_session_lap = max(available_laps) if available_laps else circuit_meta["total_laps"]
    session_data_src = get_session_data_source(selected_season, selected_circuit, selected_session, db_path=DB_PATH)

    if session_data_src == "FASTF1_REAL":
        _src_badge = '<span style="color:#00E676;font-weight:800;">🟢 REAL FASTF1 FIA TELEMETRY</span>'
    elif session_data_src == "OPENF1_REAL":
        _src_badge = '<span style="color:#00E676;font-weight:800;">🟢 REAL OPENF1 FIA TELEMETRY</span>'
    elif session_data_src == "JOLPICA_CALIBRATED":
        _src_badge = '<span style="color:#FFD600;font-weight:800;">🟡 JOLPICA-CALIBRATED MODEL</span>'
    else:
        _src_badge = '<span style="color:#B0BEC5;font-weight:800;">⚪ SYNTHETIC PHYSICS MODEL</span>'

    st.sidebar.markdown(
        f"""
        <div style="background:rgba(17,22,33,0.9);border:1px solid #2A3244;border-radius:8px;
                    padding:8px 10px;margin:6px 0 8px 0;font-size:11.5px;line-height:1.55;color:#CBD5E1;">
            <div>📡 <b>Telemetry:</b> {_src_badge}</div>
            <div>🗺️ <b>Track Map:</b> <span style="color:#38BDF8;font-weight:700;">Real FIA GPS Outline</span></div>
            <div>🏆 <b>WDC/WCC:</b> <span style="color:#38BDF8;font-weight:700;">Official 2021–2024 Jolpica</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if session_data_src not in ("FASTF1_REAL", "OPENF1_REAL"):
        if st.sidebar.button(
            "📡 Fetch Real FastF1 Telemetry",
            key="sb_fetch_fastf1_btn",
            use_container_width=True,
            help=f"Download authentic FIA lap-by-lap timing & tyre data for {selected_season} {selected_circuit} {selected_session} via FastF1.",
        ):
            with st.sidebar.status(
                f"Downloading {selected_season} {selected_circuit} {selected_session} via FastF1...",
                expanded=False,
            ):
                ok_ff1, msg_ff1 = fetch_real_fastf1_session(
                    circuit=selected_circuit,
                    session_type=selected_session,
                    season=selected_season,
                    db_path=DB_PATH,
                )
            get_cached_session_laps_map.cache_clear()
            _cached_driver_decision_impl.cache_clear()
            _cached_championship_view_impl.cache_clear()
            st.session_state["_fastf1_fetch_status"] = (ok_ff1, msg_ff1)
            st.rerun()

    if st.session_state.get("_fastf1_fetch_status"):
        _ff1_ok, _ff1_msg = st.session_state.pop("_fastf1_fetch_status")
        if _ff1_ok:
            st.sidebar.success(_ff1_msg)
        else:
            st.sidebar.warning(_ff1_msg)

    if (
        selected_season != st.session_state.selected_season
        or selected_circuit != st.session_state.selected_circuit
        or selected_session != st.session_state.selected_session
    ):
        st.session_state.selected_season = selected_season
        st.session_state.selected_circuit = selected_circuit
        st.session_state.selected_session = selected_session
        st.session_state.current_lap = min(17, max_session_lap)
        st.session_state.p1_current_lap = int(st.session_state.current_lap)
        st.session_state.track_temp_override = float(circuit_meta["track_temp"])

    # Historical Lap Scrubber & Live Step Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Historical Replay & Live Transport")
    selected_lap = st.sidebar.slider(
        f"⏪ Scrub Lap (1 .. {max_session_lap})",
        min_value=1,
        max_value=int(max_session_lap),
        value=min(int(st.session_state.current_lap), int(max_session_lap)),
    )
    if selected_lap != st.session_state.get("_last_p1_slider_lap", selected_lap):
        st.session_state.current_lap = selected_lap
        st.session_state.p1_current_lap = selected_lap
    st.session_state._last_p1_slider_lap = int(st.session_state.current_lap)

    lap_progress_pct = st.sidebar.slider(
        "📍 On-Track Micro-Timestamp (% of Lap)",
        min_value=0,
        max_value=99,
        value=int(st.session_state.lap_progress_pct),
        help="Moves all 20 cars around the circuit map and updates live speed/throttle/RPM sensor cursors.",
    )
    if lap_progress_pct != st.session_state.get("_last_p1_slider_pct", lap_progress_pct):
        st.session_state.lap_progress_pct = lap_progress_pct
        st.session_state.sub_lap_progress = round(lap_progress_pct / 100.0, 4)
    st.session_state._last_p1_slider_pct = int(st.session_state.lap_progress_pct)

    speed_options = list(PLAYBACK_SPEED_MULTIPLIERS.keys())
    play_btn_label = "⏸️ PAUSE AUTO-SYNC" if st.session_state.is_playing else "▶️ AUTO-SYNC LAPS"
    if st.sidebar.button(
        play_btn_label,
        key="top_video_play_pause_btn",
        type="primary" if not st.session_state.is_playing else "secondary",
        use_container_width=True,
    ):
        st.session_state.is_playing = not st.session_state.is_playing
        st.rerun()

    t_c1, t_c2, t_c3 = st.sidebar.columns(3)
    if t_c1.button("⏮️ -1 Lap", key="top_video_prev_lap_btn", use_container_width=True):
        st.session_state.current_lap = max(1, int(st.session_state.current_lap) - 1)
        st.session_state.p1_current_lap = st.session_state.current_lap
        st.session_state._last_p1_slider_lap = st.session_state.current_lap
        st.rerun()
    if t_c2.button("⏭️ +1 Lap", key="top_video_next_lap_btn", use_container_width=True):
        st.session_state.current_lap = min(int(max_session_lap), int(st.session_state.current_lap) + 1)
        st.session_state.p1_current_lap = st.session_state.current_lap
        st.session_state._last_p1_slider_lap = st.session_state.current_lap
        st.rerun()
    if t_c3.button("⏩ +25%", key="top_video_scrub_btn", use_container_width=True):
        new_sub = float(st.session_state.sub_lap_progress) + 0.25
        if new_sub >= 1.0:
            st.session_state.sub_lap_progress = round(new_sub % 1.0, 4)
            st.session_state.current_lap = min(int(max_session_lap), int(st.session_state.current_lap) + 1)
            st.session_state.p1_current_lap = st.session_state.current_lap
            st.session_state._last_p1_slider_lap = st.session_state.current_lap
        else:
            st.session_state.sub_lap_progress = round(new_sub, 4)
        st.session_state.lap_progress_pct = int(round(st.session_state.sub_lap_progress * 100)) % 100
        st.session_state._last_p1_slider_pct = st.session_state.lap_progress_pct
        st.rerun()

    selected_speed_label = st.sidebar.selectbox(
        "⏩ Sync Playback Speed",
        speed_options,
        index=speed_options.index(st.session_state.playback_speed_label)
        if st.session_state.playback_speed_label in speed_options
        else 2,
        key="top_video_speed_select",
    )
    st.session_state.playback_speed_label = selected_speed_label
    speed_mult = PLAYBACK_SPEED_MULTIPLIERS[selected_speed_label]

    # Environment Override Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌦️ Environment & Race Control Overrides")
    env_override = st.sidebar.checkbox(
        "Enable Manual Environment Override",
        value=st.session_state.env_override_active,
        help="Override the historical track temperature, weather, and FIA flag status (GREEN / VSC / SC).",
    )
    st.session_state.env_override_active = env_override

    track_temp = st.sidebar.slider(
        "🌡️ Track Temperature (°C)",
        20.0,
        60.0,
        float(st.session_state.track_temp_override),
        0.5,
    )
    st.session_state.track_temp_override = track_temp

    weather_cond = st.sidebar.selectbox(
        "☁️ Weather / Grip Condition",
        ["Dry (Clear)", "Overcast (Cool)", "Light Drizzle (Damp)", "Heavy Rain (Wet)"],
        index=0,
    )
    st.session_state.weather_condition = weather_cond

    f1, f2, f3 = st.sidebar.columns(3)
    if f1.button("🟢 GREEN", use_container_width=True):
        st.session_state.env_override_active = True
        st.session_state.track_status_override = "GREEN"
        st.rerun()
    if f2.button("🟡 VSC", use_container_width=True):
        st.session_state.env_override_active = True
        st.session_state.track_status_override = "VSC"
        st.rerun()
    if f3.button("🚨 SC", use_container_width=True):
        st.session_state.env_override_active = True
        st.session_state.track_status_override = "SC"
        st.rerun()

    # Sync active Grand Prix session lap to live_telemetry_laps in DuckDB
    df_standings = sync_gp_lap_to_live_table(
        circuit=selected_circuit,
        session_type=selected_session,
        lap=st.session_state.current_lap,
        season=selected_season,
        track_status_override=st.session_state.track_status_override if st.session_state.env_override_active else None,
        track_temp_override=track_temp if st.session_state.env_override_active else None,
        db_path=DB_PATH,
    )

    # Auto-sync weather condition from real lap compounds & track temp unless manual override is ON
    inferred_wx = infer_lap_weather_condition(df_standings)
    if not env_override:
        weather_cond = inferred_wx
        st.session_state.weather_condition = weather_cond

    driver_options = df_standings["driver"].tolist() if not df_standings.empty else [d["driver"] for d in F1_2024_DRIVERS]
    if "sidebar_ego_driver" not in st.session_state or st.session_state.sidebar_ego_driver not in driver_options:
        st.session_state.sidebar_ego_driver = driver_options[0]
    if st.session_state.get("_pill_driver_pending") in driver_options:
        st.session_state.sidebar_ego_driver = st.session_state.pop("_pill_driver_pending")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎧 Active Driver & Strategy Target")
    ego_driver = st.sidebar.selectbox(
        "🏎️ Inspect Driver (Click to view Radio & Team Decision):",
        driver_options,
        key="sidebar_ego_driver",
    )
    pit_box_num = st.sidebar.slider("🔧 Team Pit Box Position (1=Entry, 10=Exit)", 1, 10, 1)
    target_tyre = st.sidebar.selectbox("🛞 Target Next Pit Compound", ["MEDIUM", "HARD", "SOFT", "INTERMEDIATE", "WET"], index=1)

    current_lap = int(df_standings["lap"].iloc[0]) if not df_standings.empty else st.session_state.current_lap
    current_status = str(df_standings["track_status"].iloc[0]) if not df_standings.empty else "GREEN"
    active_temp = float(df_standings["track_temp"].iloc[0]) if not df_standings.empty else track_temp

    # =====================================================================
    # SPLIT-SCREEN COCKPIT: 60 FPS BROADCAST STAGE (LEFT) + PIT WALL RADIO COMMS (RIGHT)
    # =====================================================================
    _x_tr, _y_tr, _curv_tr = get_circuit_curvature_profile(selected_circuit, n_points=240)
    _p1_laps_map = get_cached_session_laps_map(
        season=int(selected_season),
        circuit=str(selected_circuit),
        session_type=str(selected_session),
        track_status_override=st.session_state.track_status_override if st.session_state.env_override_active else None,
        db_path=DB_PATH,
    )
    _ego_row_top = df_standings[df_standings["driver"] == ego_driver].iloc[0] if not df_standings.empty and not df_standings[df_standings["driver"] == ego_driver].empty else None
    _ego_comp_top = str(_ego_row_top["compound"]).upper() if _ego_row_top is not None else "SOFT"
    _ego_age_top = int(_ego_row_top["tyre_life"]) if _ego_row_top is not None else 12
    _p1_micro_tel = get_real_driver_lap_telemetry(
        season=int(selected_season),
        circuit=selected_circuit,
        session_type=selected_session,
        driver=ego_driver,
        lap=current_lap,
        compound=_ego_comp_top,
        tyre_life=_ego_age_top,
        track_status=current_status,
        n_points=240,
        allow_fastf1_download=False,
    )
    _p1_pit_sim_top = get_cached_pit_simulator(selected_circuit, pit_box_num)
    _p1_pit_loss_top = float(_p1_pit_sim_top.circuit_data["normal_loss"]) * (
        0.46 if current_status == "SC" else (0.68 if current_status == "VSC" else 1.0)
    )
    _p1_canvas_payload = build_broadcast_payload(
        circuit=selected_circuit,
        race_name=circuit_meta["race_name"],
        season=int(selected_season),
        session_label=SESSION_DISPLAY_LABELS.get(selected_session, selected_session),
        x_track=_x_tr,
        y_track=_y_tr,
        norm_curv=_curv_tr,
        track_length_m=float(circuit_meta.get("track_length_m", 5412.0)),
        base_lap_sec=float(circuit_meta["base_lap_sec"]),
        total_laps=int(max_session_lap),
        initial_lap=int(st.session_state.p1_current_lap),
        initial_sub_lap=float(st.session_state.sub_lap_progress),
        ego_driver=ego_driver,
        pit_loss_sec=_p1_pit_loss_top,
        track_temp_c=active_temp,
        weather_cond=weather_cond,
        laps_by_number=_p1_laps_map,
        driver_meta=DRIVER_METADATA_MAP,
        micro_tel_df=_p1_micro_tel,
        auto_play=False,
        initial_speed=PLAYBACK_SPEED_MULTIPLIERS.get(st.session_state.playback_speed_label, 15.0),
    )

    col_p1_stage, col_p1_comms = st.columns([1.85, 1.0], gap="medium")
    with col_p1_stage:
        components.html(render_smooth_broadcast_html(_p1_canvas_payload), height=642, scrolling=False)

    with col_p1_comms:
        def _build_p1_radio_ctx() -> RaceContext:
            snap = st.session_state.get("_p1_live_snapshot")
            active_df = snap["df"] if snap is not None and not snap["df"].empty else df_standings
            active_lap_num = int(active_df["lap"].iloc[0]) if not active_df.empty else int(current_lap)
            used_dry = get_driver_used_compounds_up_to_lap(
                season=int(selected_season),
                circuit=selected_circuit,
                session_type=selected_session,
                driver=ego_driver,
                up_to_lap=active_lap_num,
                db_path=DB_PATH,
            )
            # Detect if teammate is within 4.5s ahead (potential double-stack queuing)
            tm_gap_ahead = None
            ego_team = DRIVER_METADATA_MAP.get(ego_driver, {}).get("team")
            if ego_team and not active_df.empty and not active_df[active_df["driver"] == ego_driver].empty:
                ego_g = float(active_df[active_df["driver"] == ego_driver].iloc[0]["gap_to_leader"])
                for r in active_df.itertuples(index=False):
                    if r.driver != ego_driver and DRIVER_METADATA_MAP.get(r.driver, {}).get("team") == ego_team:
                        diff = ego_g - float(r.gap_to_leader)
                        if 0.0 <= diff < 4.5:
                            tm_gap_ahead = round(diff, 2)
                        break
            return RaceContext(
                driver=ego_driver,
                circuit=selected_circuit,
                total_race_laps=int(max_session_lap),
                team_pit_box_position=int(pit_box_num),
                weather_condition=weather_cond,
                used_dry_compounds=used_dry,
                teammate_gap_ahead_sec=tm_gap_ahead,
                teammate_pitting=False,
                live_data=build_live_data_from_standings_df(
                    active_df,
                    circuit=selected_circuit,
                    total_race_laps=int(max_session_lap),
                    weather_condition=weather_cond,
                    used_dry_compounds=used_dry,
                    teammate_gap_ahead_sec=tm_gap_ahead,
                    teammate_pitting=False,
                ),
            )

        render_pit_wall_radio_console(
            key_prefix="p1",
            history_state_key="radio_history",
            ego_driver=ego_driver,
            current_lap=int(st.session_state.p1_current_lap),
            max_laps=int(max_session_lap),
            track_status=current_status,
            target_tyre=target_tyre,
            build_context_fn=_build_p1_radio_ctx,
        )

    # =====================================================================
    # REAL-TIME DYNAMIC MOVING GRAPHICS & CHANGING TELEMETRY (@st.fragment)
    # =====================================================================
    _p1_run_every = 0.25 if st.session_state.is_playing else None

    @st.fragment(run_every=_p1_run_every)
    def render_page1_live_telemetry_fragment():
        if st.session_state.is_playing:
            base_lap_s = max(60.0, float(circuit_meta["base_lap_sec"]))
            delta_frac = (0.25 * float(speed_mult)) / base_lap_s
            next_sub = float(st.session_state.sub_lap_progress) + delta_frac
            if next_sub >= 1.0:
                laps_inc = int(next_sub // 1.0)
                st.session_state.sub_lap_progress = round(next_sub % 1.0, 4)
                next_lap = int(st.session_state.p1_current_lap) + laps_inc
                if next_lap > int(max_session_lap):
                    next_lap = 1
                st.session_state.p1_current_lap = next_lap
                st.session_state.current_lap = next_lap
                st.session_state._last_p1_slider_lap = next_lap
            else:
                st.session_state.sub_lap_progress = round(next_sub, 4)
            st.session_state.lap_progress_pct = int(round(st.session_state.sub_lap_progress * 100)) % 100
            st.session_state._last_p1_slider_pct = st.session_state.lap_progress_pct

        frag_lap = int(st.session_state.p1_current_lap)
        frag_sub = float(st.session_state.sub_lap_progress)
        frag_pct = round(frag_sub * 100.0, 1)

        ovr_status = st.session_state.track_status_override if st.session_state.env_override_active else None
        ovr_temp = round(float(track_temp), 1) if st.session_state.env_override_active else None
        snap_key = (selected_season, selected_circuit, selected_session, frag_lap, ovr_status, ovr_temp)
        cached_snap = st.session_state.get("_p1_live_snapshot")

        if cached_snap is not None and cached_snap.get("key") == snap_key:
            df_base_lap = cached_snap["df"]
        elif not df_standings.empty and int(df_standings["lap"].iloc[0]) == frag_lap:
            df_base_lap = df_standings
        else:
            df_base_lap = sync_gp_lap_to_live_table(
                circuit=selected_circuit,
                session_type=selected_session,
                lap=frag_lap,
                season=selected_season,
                track_status_override=ovr_status,
                track_temp_override=ovr_temp,
                db_path=DB_PATH,
            )

        frag_status = str(df_base_lap["track_status"].iloc[0]) if not df_base_lap.empty else "GREEN"
        frag_temp = float(df_base_lap["track_temp"].iloc[0]) if not df_base_lap.empty else track_temp
        st.session_state._p1_live_snapshot = {
            "key": snap_key,
            "lap": frag_lap,
            "status": frag_status,
            "temp": frag_temp,
            "df": df_base_lap,
        }

        df_live = compute_dynamic_standings_at_sub_lap(
            df_standings=df_base_lap,
            circuit=selected_circuit,
            sub_lap_progress=frag_sub,
        )

        # Extract Selected Driver Telemetry & Team Decision (cached per canonical lap state for 4 Hz speed)
        ego_row = df_base_lap[df_base_lap["driver"] == ego_driver].iloc[0] if not df_base_lap.empty else None
        if ego_row is not None:
            decision_bundle = get_cached_driver_decision(
                driver=ego_driver,
                df_base_standings=df_base_lap,
                circuit=selected_circuit,
                track_status=frag_status,
                track_temp=frag_temp,
                pit_box_num=pit_box_num,
                target_tyre=target_tyre,
                total_laps=int(max_session_lap),
                weather_condition=weather_cond,
            )
            mc_res = decision_bundle["mc_rejoin"]
            uc_res = decision_bundle["mc_undercut"]
            ego_compound = decision_bundle["compound"]
            ego_tyre_life = decision_bundle["tyre_life"]
            ego_gap = float(ego_row["gap_to_leader"])
            ego_pos = decision_bundle["position"]
            current_deg = decision_bundle["deg"]
            future_5lap_loss = decision_bundle["next_5l_loss"]
        else:
            ego_compound, ego_tyre_life, ego_gap, ego_pos = "SOFT", 15, 0.0, 1
            current_deg, future_5lap_loss = 1.2, 6.5
            default_norm_loss = float(circuits_db.get(selected_circuit, {}).get("normal_loss", 23.0))
            mc_res = {"median_pit_loss": default_norm_loss, "p95_pit_loss": default_norm_loss + 2.0, "elapsed_ms": 0.8, "median_pos": 4, "expected_pos": 4.1, "p_clean_air": 0.8}
            uc_res = {"win_probability": 0.65, "median_margin_sec": 0.8}
            decision_bundle = None

        # Generate Micro-Telemetry Sensor Readings for the Selected Driver at Current Sub-Lap Progress
        micro_tel_df = generate_driver_micro_telemetry(
            driver=ego_driver,
            circuit=selected_circuit,
            lap=frag_lap,
            compound=ego_compound,
            tyre_life=ego_tyre_life,
            track_status=frag_status,
        )
        pt_idx = int(np.clip(int(frag_sub * (len(micro_tel_df) - 1)), 0, len(micro_tel_df) - 1))
        live_sensor = micro_tel_df.iloc[pt_idx]

        # Row 1: Collapsible 3x2 Dark-Glass F1 Telemetry & Strategy Card Grid
        with st.expander(
            f"📊 {ego_driver} Telemetry & Strategy Cards — P{ego_pos} ({ego_compound} {ego_tyre_life}L) · "
            f"{live_sensor['Speed']:.0f} km/h G{int(live_sensor['nGear'])} · "
            f"+{current_deg:.2f}s/L Deg · Rejoin P{mc_res['median_pos']} ({mc_res['p_clean_air']*100:.0f}% Clean Air)",
            expanded=False,
        ):
            st.markdown(
                render_telemetry_cards_grid_html(
                    ego_driver=ego_driver,
                    ego_pos=ego_pos,
                    ego_compound=ego_compound,
                    ego_tyre_life=ego_tyre_life,
                    frag_lap=frag_lap,
                    max_session_lap=int(max_session_lap),
                    live_sensor=live_sensor,
                    current_deg=current_deg,
                    future_5lap_loss=future_5lap_loss,
                    mc_res=mc_res,
                    uc_res=uc_res,
                    target_tyre=target_tyre,
                ),
                unsafe_allow_html=True,
            )

        # =====================================================================
        # ROW 2: CLICKED DRIVER RADIO INSPECTOR & TEAM PIT WALL DECISION PANEL
        # =====================================================================
        st.markdown("---")
        st.subheader("🎧 Interactive Driver Inspector — Live Radio Feed & Team Strategy Decision")
        drv_cols = st.columns(min(10, len(driver_options)))
        for i, d_code in enumerate(driver_options[:10]):
            if drv_cols[i].button(
                f"{'🟢 ' if d_code == ego_driver else ''}{d_code}",
                key=f"pill_top_{d_code}",
                use_container_width=True,
            ):
                st.session_state._pill_driver_pending = d_code
                st.rerun()

        if len(driver_options) > 10:
            drv_cols_2 = st.columns(min(10, len(driver_options) - 10))
            for i, d_code in enumerate(driver_options[10:20]):
                if drv_cols_2[i].button(
                    f"{'🟢 ' if d_code == ego_driver else ''}{d_code}",
                    key=f"pill_bot_{d_code}",
                    use_container_width=True,
                ):
                    st.session_state._pill_driver_pending = d_code
                    st.rerun()

        active_inspected_driver = ego_driver if ego_driver in driver_options else driver_options[0]
        if active_inspected_driver == ego_driver and decision_bundle is not None:
            inspected_bundle = decision_bundle
        else:
            inspected_bundle = get_cached_driver_decision(
                driver=active_inspected_driver,
                df_base_standings=df_base_lap,
                circuit=selected_circuit,
                track_status=frag_status,
                track_temp=frag_temp,
                pit_box_num=pit_box_num,
                target_tyre=target_tyre,
                total_laps=int(max_session_lap),
                weather_condition=weather_cond,
            )

        insp_c1, insp_c2 = st.columns([1.0, 1.0])
        with insp_c1:
            st.info(
                f"**🎙️ LIVE DRIVER RADIO ({inspected_bundle['name']} — #{inspected_bundle['number']} {inspected_bundle['team']} | P{inspected_bundle['position']})**\n\n"
                f"{inspected_bundle['driver_radio_msg']}\n\n"
                f"• **Car Ahead**: `{inspected_bundle['ahead_driver']}` (+{inspected_bundle['gap_ahead']:.2f}s) | "
                f"**Car Behind**: `{inspected_bundle['behind_driver']}` (+{inspected_bundle['gap_behind']:.2f}s)"
            )
        with insp_c2:
            st.success(
                f"**🧠 TEAM PIT WALL DECISION ({inspected_bundle['team']} Strategy Computer)**\n\n"
                f"**{inspected_bundle['team_decision']}**\n\n"
                f"• **10k Monte Carlo Rejoin**: `P{inspected_bundle['mc_rejoin']['median_pos']}` "
                f"(Pit Loss `{inspected_bundle['mc_rejoin']['median_pit_loss']:.2f}s`, Clean Air `{inspected_bundle['mc_rejoin']['p_clean_air']*100:.1f}%`) | "
                f"**Undercut Win Prob**: `{inspected_bundle['mc_undercut']['win_probability']*100:.1f}%`"
            )

        # =====================================================================
        # ROW 3: SIDE-BY-SIDE ANALYTICS (L1 TENSOR DEGRADATION & 10K MONTE CARLO REJOIN)
        # =====================================================================
        st.markdown("---")
        col_deg, col_mc = st.columns([1.05, 1.05], gap="medium")

        with col_deg:
            st.subheader(f"📈 L1 Tensor Tyre Degradation Curves ({frag_temp:.1f}°C Track Temp)")
            laps_axis = np.arange(1, int(max_session_lap) + 1)
            fig_deg = go.Figure()
            compound_styles = {
                "SOFT": ("#FF3333", "solid"),
                "MEDIUM": ("#FFD700", "dash"),
                "HARD": ("#00E5FF", "dot"),
            }
            deg_curves = get_cached_degradation_curves(round(frag_temp, 1), int(max_session_lap))
            for comp, (color, dash_style) in compound_styles.items():
                curve = deg_curves[comp]
                fig_deg.add_trace(
                    go.Scatter(
                        x=laps_axis,
                        y=curve,
                        mode="lines",
                        name=f"{comp} Compound",
                        line=dict(color=color, width=3, dash=dash_style),
                    )
                )
            fig_deg.add_trace(
                go.Scatter(
                    x=[ego_tyre_life],
                    y=[current_deg],
                    mode="markers+text",
                    name=f"{ego_driver} Now ({ego_tyre_life}L)",
                    text=[f"  {ego_driver} (+{current_deg:.2f}s)"],
                    textposition="middle right",
                    marker=dict(size=13, color="#00FF66", symbol="diamond", line=dict(width=2, color="white")),
                )
            )
            fig_deg.add_hline(
                y=2.15,
                line_dash="dot",
                line_color="#FF8800",
                annotation_text="Pit Window Crossover Threshold (+2.15s/lap)",
            )
            fig_deg.update_layout(
                template="plotly_dark",
                height=390,
                margin=dict(l=20, r=20, t=25, b=20),
                xaxis_title="Tyre Age (Laps)",
                yaxis_title="Relative Degradation Delta (+s/lap)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                uirevision="vre_deg_constant",
            )
            st.plotly_chart(fig_deg, use_container_width=True)

        with col_mc:
            st.subheader(f"🎯 10,000-Sim Monte Carlo Pit Rejoin Window — Lap {frag_lap}/{max_session_lap} ({frag_status})")
            pit_sim_mc = get_cached_pit_simulator(selected_circuit, pit_box_num)
            pit_samples = pit_sim_mc._sample_pit_loss_vectorized(frag_status, 10_000) + ego_gap
            fig_mc = go.Figure()
            fig_mc.add_trace(
                go.Histogram(
                    x=pit_samples,
                    nbinsx=45,
                    name=f"{ego_driver} Rejoin Distribution",
                    marker_color="#00E5FF",
                    opacity=0.75,
                )
            )
            for car in df_base_lap.head(10).itertuples(index=False):
                if car.driver != ego_driver:
                    fig_mc.add_vline(
                        x=float(car.gap_to_leader),
                        line_dash="dash",
                        line_color="#FF4B4B",
                        annotation_text=f"{car.driver} (+{float(car.gap_to_leader):.1f}s)",
                        annotation_position="top",
                    )
            fig_mc.update_layout(
                template="plotly_dark",
                height=390,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Track Gap to Leader Upon Pit Exit (Seconds)",
                yaxis_title="Monte Carlo Outcomes (out of 10,000)",
                showlegend=False,
                uirevision="vre_mc_constant",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        # =====================================================================
        # ROW 4: WORLD DRIVERS' (WDC) & CONSTRUCTORS' (WCC) CHAMPIONSHIP STANDINGS TILL THIS RACE
        # =====================================================================
        st.markdown("---")
        ego_team_name = DRIVER_METADATA_MAP.get(ego_driver, {"team": "Red Bull Racing"})["team"]
        champ_view = get_cached_championship_view(
            season=selected_season,
            circuit=selected_circuit,
            df_standings=df_base_lap,
            ego_driver=ego_driver,
            ego_team_name=ego_team_name,
        )
        champ_data = champ_view["champ_data"]
        wdc_df = champ_data["wdc"]
        wcc_df = champ_data["wcc"]

        st.subheader(
            f"🏆 {selected_season} FIA World Championship Standings Till Round {champ_data['round']}/{champ_data['total_rounds']}: "
            f"{champ_data['race_name']} (Live Projected Points)"
        )
        wdc_leader = wdc_df.iloc[0]
        wcc_leader = wcc_df.iloc[0]
        ego_wdc_row = wdc_df[wdc_df["driver"] == ego_driver].iloc[0] if not wdc_df[wdc_df["driver"] == ego_driver].empty else wdc_leader
        ego_wcc_row = wcc_df[wcc_df["team"] == ego_team_name].iloc[0] if not wcc_df[wcc_df["team"] == ego_team_name].empty else wcc_leader

        ch1, ch2, ch3, ch4 = st.columns(4)
        ch1.metric(
            "👑 WDC Leader (Drivers' Title)",
            f"P1 {wdc_leader['driver']} ({wdc_leader['total_points']} pts)",
            f"+{wdc_leader['live_race_points']} pts live ({wdc_leader['wins']} Wins)",
        )
        ch2.metric(
            f"🏎️ {ego_driver} WDC Standing",
            f"P{int(ego_wdc_row['rank'])} ({int(ego_wdc_row['total_points'])} pts)",
            "WDC LEADER" if int(ego_wdc_row["gap_to_leader"]) == 0 else f"-{int(ego_wdc_row['gap_to_leader'])} pts to P1 (+{int(ego_wdc_row['live_race_points'])} live)",
        )
        ch3.metric(
            "🏆 WCC Leader (Constructors' Title)",
            f"P1 {wcc_leader['team']} ({wcc_leader['total_points']} pts)",
            f"+{wcc_leader['live_race_points']} pts live ({wcc_leader['drivers_str']})",
        )
        ch4.metric(
            f"🛡️ {ego_team_name} WCC Standing",
            f"P{int(ego_wcc_row['rank'])} ({int(ego_wcc_row['total_points'])} pts)",
            "WCC LEADER" if int(ego_wcc_row["gap_to_leader"]) == 0 else f"-{int(ego_wcc_row['gap_to_leader'])} pts to P1 (+{int(ego_wcc_row['live_race_points'])} live)",
        )

        tab_wdc, tab_wcc = st.tabs([
            f"🏎️ Drivers' Championship (WDC) — Round 1 to Round {champ_data['round']} ({selected_circuit})",
            f"🛠️ Constructors' Championship (WCC) — Round 1 to Round {champ_data['round']} ({selected_circuit})",
        ])
        with tab_wdc:
            st.markdown(champ_view["wdc_html"], unsafe_allow_html=True)
        with tab_wcc:
            st.markdown(champ_view["wcc_html"], unsafe_allow_html=True)

    render_page1_live_telemetry_fragment()


# =====================================================================
# PAGE 2: 100% USER-CREATED HYPOTHETICAL SCENARIO SANDBOX
# =====================================================================
else:
    st.title("🧪 100% Custom F1 Scenario Builder & Hypothetical Strategy Sandbox")
    st.markdown(
        "Design **any hypothetical Grand Prix scenario** from scratch: choose the circuit, lap/timestamp, "
        "track temperature, FIA flag condition, exact driver order, tyre compounds/ages, and exact time intervals "
        "between drivers — then launch the **Live Circuit Map, 10,000 Monte Carlo Pit Simulator, and Race Engineer AI** on your custom grid!"
    )

    def _sync_scenario_widget_keys(rows: list[dict]):
        """Synchronizes per-row expander widget session_state keys with `rows` before widgets instantiate."""
        st.session_state.custom_scenario_rows = rows
        st.session_state.scen_num_cars_slider = len(rows)
        for idx, r in enumerate(rows):
            st.session_state[f"scen_drv_{idx}"] = r["driver"]
            st.session_state[f"scen_comp_{idx}"] = r["compound"]
            st.session_state[f"scen_age_{idx}"] = int(r["tyre_life"])
            if idx > 0:
                st.session_state[f"scen_int_{idx}"] = float(max(0.05, r["interval_sec"]))
            st.session_state[f"scen_lap_{idx}"] = float(r["base_lap_sec"])

    if st.session_state.get("_pending_scenario_rows") is not None:
        _sync_scenario_widget_keys(st.session_state.pop("_pending_scenario_rows"))
    if st.session_state.get("_pending_scen_lap") is not None:
        st.session_state.scen_lap_slider = int(st.session_state.pop("_pending_scen_lap"))

    # Sidebar Environment & Circuit Config for Custom Scenario
    st.sidebar.subheader("🛠️ Custom Scenario Environment")
    circuit_list = list(circuits_db.keys())
    scen_circuit = st.sidebar.selectbox("🏁 Scenario Circuit", circuit_list, index=0)
    scen_meta = get_circuit_metadata(scen_circuit)

    if "scen_lap_slider" not in st.session_state:
        st.session_state.scen_lap_slider = min(42, int(scen_meta["total_laps"]))
    else:
        st.session_state.scen_lap_slider = min(int(st.session_state.scen_lap_slider), int(scen_meta["total_laps"]))

    scen_lap = st.sidebar.slider("⏱️ Scenario Lap Number", 1, int(scen_meta["total_laps"]), key="scen_lap_slider")
    st.session_state.p2_current_lap = int(scen_lap)
    p2_default_pct = int(round(float(st.session_state.p2_sub_lap_progress) * 100)) % 100
    scen_progress = st.sidebar.slider("📍 On-Track Timestamp (% of Lap)", 0, 99, p2_default_pct)
    if scen_progress != st.session_state.get("_last_p2_slider_pct", scen_progress):
        st.session_state.p2_sub_lap_progress = round(scen_progress / 100.0, 4)
    st.session_state._last_p2_slider_pct = int(round(float(st.session_state.p2_sub_lap_progress) * 100)) % 100
    scen_status = st.sidebar.selectbox("🚩 Track Flag Condition", ["GREEN", "VSC", "SC"], index=0)
    scen_temp = st.sidebar.slider("🌡️ Track Temperature (°C)", 20.0, 60.0, 34.0, 0.5)
    scen_weather = st.sidebar.selectbox(
        "☁️ Scenario Weather / Grip",
        ["Dry (Clear)", "Overcast (Cool)", "Light Drizzle (Damp)", "Heavy Rain (Wet)"],
        index=0,
    )
    scen_pit_box = st.sidebar.slider("🔧 Ego Team Pit Box (1=Entry, 10=Exit)", 1, 10, 1)
    scen_target_tyre = st.sidebar.selectbox("🛞 Target Pit Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=0)

    all_drv_codes = [d["driver"] for d in F1_2024_DRIVERS]
    if "scen_num_cars_slider" not in st.session_state:
        st.session_state.scen_num_cars_slider = len(st.session_state.custom_scenario_rows)
    num_cars = st.sidebar.slider("🏎️ Number of Cars in Scenario Grid", 2, 20, key="scen_num_cars_slider")

    # Adjust custom_scenario_rows length if slider changed
    while len(st.session_state.custom_scenario_rows) < num_cars:
        used = {r["driver"] for r in st.session_state.custom_scenario_rows}
        next_drv = next((c for c in all_drv_codes if c not in used), "BOT")
        new_idx = len(st.session_state.custom_scenario_rows)
        new_row = {
            "pos": new_idx + 1,
            "driver": next_drv,
            "compound": "MEDIUM",
            "tyre_life": 10,
            "interval_sec": 2.0,
            "base_lap_sec": round(scen_meta["base_lap_sec"] + 0.5, 3),
        }
        st.session_state.custom_scenario_rows.append(new_row)
        st.session_state[f"scen_drv_{new_idx}"] = new_row["driver"]
        st.session_state[f"scen_comp_{new_idx}"] = new_row["compound"]
        st.session_state[f"scen_age_{new_idx}"] = new_row["tyre_life"]
        if new_idx > 0:
            st.session_state[f"scen_int_{new_idx}"] = new_row["interval_sec"]
        st.session_state[f"scen_lap_{new_idx}"] = new_row["base_lap_sec"]
    if len(st.session_state.custom_scenario_rows) > num_cars:
        st.session_state.custom_scenario_rows = st.session_state.custom_scenario_rows[:num_cars]

    # Quick Preset Scenario Templates
    st.subheader("⚡ Step 1: Configure Custom Driver Order, Time Intervals & Tyre States")
    p_col1, p_col2, p_col3 = st.columns(3)
    if p_col1.button("🏆 Preset: Final 10-Lap Shootout (HAM vs VER vs NOR)", use_container_width=True):
        st.session_state._pending_scenario_rows = [
            {"pos": 1, "driver": "VER", "compound": "HARD", "tyre_life": 24, "interval_sec": 0.0, "base_lap_sec": 94.4},
            {"pos": 2, "driver": "HAM", "compound": "SOFT", "tyre_life": 4, "interval_sec": 3.2, "base_lap_sec": 93.3},
            {"pos": 3, "driver": "NOR", "compound": "MEDIUM", "tyre_life": 8, "interval_sec": 1.4, "base_lap_sec": 93.6},
            {"pos": 4, "driver": "LEC", "compound": "HARD", "tyre_life": 22, "interval_sec": 4.5, "base_lap_sec": 94.3},
            {"pos": 5, "driver": "PIA", "compound": "MEDIUM", "tyre_life": 11, "interval_sec": 2.1, "base_lap_sec": 93.9},
            {"pos": 6, "driver": "RUS", "compound": "HARD", "tyre_life": 23, "interval_sec": 3.8, "base_lap_sec": 94.5},
        ]
        st.rerun()
    if p_col2.button("🚨 Preset: Late Safety Car Pack Compression", use_container_width=True):
        st.session_state._pending_scenario_rows = [
            {"pos": 1, "driver": "LEC", "compound": "HARD", "tyre_life": 28, "interval_sec": 0.0, "base_lap_sec": 94.6},
            {"pos": 2, "driver": "VER", "compound": "HARD", "tyre_life": 27, "interval_sec": 0.6, "base_lap_sec": 94.4},
            {"pos": 3, "driver": "NOR", "compound": "SOFT", "tyre_life": 2, "interval_sec": 0.5, "base_lap_sec": 93.1},
            {"pos": 4, "driver": "HAM", "compound": "SOFT", "tyre_life": 2, "interval_sec": 0.7, "base_lap_sec": 93.2},
            {"pos": 5, "driver": "SAI", "compound": "MEDIUM", "tyre_life": 14, "interval_sec": 0.8, "base_lap_sec": 94.0},
            {"pos": 6, "driver": "ALO", "compound": "SOFT", "tyre_life": 3, "interval_sec": 0.6, "base_lap_sec": 93.5},
        ]
        st.rerun()
    if p_col3.button("⚔️ Preset: DRS Train Undercut Battle (6 Cars Within 4s)", use_container_width=True):
        st.session_state._pending_scenario_rows = [
            {"pos": 1, "driver": "NOR", "compound": "MEDIUM", "tyre_life": 16, "interval_sec": 0.0, "base_lap_sec": 94.1},
            {"pos": 2, "driver": "VER", "compound": "MEDIUM", "tyre_life": 16, "interval_sec": 0.7, "base_lap_sec": 93.9},
            {"pos": 3, "driver": "LEC", "compound": "MEDIUM", "tyre_life": 15, "interval_sec": 0.8, "base_lap_sec": 94.0},
            {"pos": 4, "driver": "HAM", "compound": "HARD", "tyre_life": 12, "interval_sec": 0.9, "base_lap_sec": 93.8},
            {"pos": 5, "driver": "RUS", "compound": "SOFT", "tyre_life": 19, "interval_sec": 0.7, "base_lap_sec": 94.5},
            {"pos": 6, "driver": "SAI", "compound": "HARD", "tyre_life": 10, "interval_sec": 0.8, "base_lap_sec": 93.9},
        ]
        st.rerun()

    # Interactive PyArrow-Free Grid Builder Controls
    with st.expander("✏️ Edit Custom Grid Order, Drivers, Tyre Compounds, Tyre Ages & Time Intervals", expanded=True):
        updated_rows = []
        seen_drivers: set[str] = set()
        cum_gap = 0.0
        for idx, r in enumerate(st.session_state.custom_scenario_rows):
            pos_num = idx + 1
            c_pos, c_drv, c_comp, c_age, c_int, c_lap = st.columns([0.6, 1.2, 1.3, 1.2, 1.5, 1.5])
            c_pos.markdown(f"**P{pos_num}**")
            comp_list = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

            if f"scen_drv_{idx}" not in st.session_state:
                st.session_state[f"scen_drv_{idx}"] = r["driver"] if r["driver"] in all_drv_codes else all_drv_codes[idx % len(all_drv_codes)]
            if f"scen_comp_{idx}" not in st.session_state:
                st.session_state[f"scen_comp_{idx}"] = r["compound"] if r["compound"] in comp_list else "SOFT"
            if f"scen_age_{idx}" not in st.session_state:
                st.session_state[f"scen_age_{idx}"] = int(r["tyre_life"])

            drv_val = c_drv.selectbox(
                f"Driver P{pos_num}",
                all_drv_codes,
                key=f"scen_drv_{idx}",
                label_visibility="collapsed",
            )
            if drv_val in seen_drivers:
                drv_val = next((c for c in all_drv_codes if c not in seen_drivers), drv_val)
            seen_drivers.add(drv_val)

            comp_val = c_comp.selectbox(
                f"Compound P{pos_num}",
                comp_list,
                key=f"scen_comp_{idx}",
                label_visibility="collapsed",
            )
            age_val = c_age.number_input(
                f"Tyre Age (L) P{pos_num}",
                min_value=1,
                max_value=70,
                step=1,
                key=f"scen_age_{idx}",
            )
            if pos_num == 1:
                int_val = 0.0
                c_int.markdown("`LEADER (0.000s)`")
            else:
                if f"scen_int_{idx}" not in st.session_state:
                    st.session_state[f"scen_int_{idx}"] = float(max(0.05, r["interval_sec"]))
                int_val = c_int.number_input(
                    f"Gap to P{pos_num-1} (s)",
                    min_value=0.05,
                    max_value=60.0,
                    step=0.1,
                    key=f"scen_int_{idx}",
                )
            cum_gap = 0.0 if pos_num == 1 else round(cum_gap + int_val, 3)

            # Calculate physics-grounded lap time from base + L1 Tensor degradation
            deg_val = tyre_cache.lookup_degradation(comp_val, scen_temp, int(age_val))
            d_meta = DRIVER_METADATA_MAP.get(drv_val, F1_2024_DRIVERS[0])
            default_lap = round(scen_meta["base_lap_sec"] + d_meta["pace_offset"] + deg_val * 0.65, 3)
            if f"scen_lap_{idx}" not in st.session_state:
                st.session_state[f"scen_lap_{idx}"] = float(r.get("base_lap_sec", default_lap))
            lap_val = c_lap.number_input(
                f"Lap Time (s) P{pos_num}",
                min_value=55.0,
                max_value=160.0,
                step=0.1,
                key=f"scen_lap_{idx}",
            )
            updated_rows.append({
                "pos": pos_num,
                "driver": drv_val,
                "compound": comp_val,
                "tyre_life": int(age_val),
                "interval_sec": float(int_val),
                "gap_to_leader": cum_gap,
                "base_lap_sec": float(lap_val),
            })
        st.session_state.custom_scenario_rows = updated_rows

    # Persist custom scenario into live_telemetry_laps so LangGraph AI and all tools use it immediately!
    def _build_custom_scenario_df(rows: list[dict], lap_num: int, sync_db: bool = True) -> pd.DataFrame:
        df_rows = []
        tuples = []
        for r in rows:
            tot_sec = int(lap_num) * float(scen_meta["base_lap_sec"]) + float(r["gap_to_leader"])
            mins = int(tot_sec // 60)
            secs = tot_sec % 60
            ts_str = f"{mins // 60:02d}:{mins % 60:02d}:{secs:06.3f}"
            spd = round(322.0 - int(r["pos"]) * 0.9, 1)
            df_rows.append({
                "lap": int(lap_num),
                "driver": str(r["driver"]),
                "position": int(r["pos"]),
                "compound": str(r["compound"]),
                "tyre_life": int(r["tyre_life"]),
                "gap_to_leader": float(r["gap_to_leader"]),
                "lap_time_sec": float(r["base_lap_sec"]),
                "track_status": str(scen_status),
                "track_temp": float(scen_temp),
                "speed_kmh": spd,
                "lap_timestamp": ts_str,
            })
            tuples.append((
                int(lap_num),
                str(r["driver"]),
                int(r["pos"]),
                str(r["compound"]),
                int(r["tyre_life"]),
                float(r["gap_to_leader"]),
                float(r["base_lap_sec"]),
                str(scen_status),
                float(scen_temp),
            ))

        if sync_db and tuples:
            with duckdb_connect(DB_PATH) as con:
                con.execute("DELETE FROM live_telemetry_laps")
                con.executemany(
                    "INSERT OR REPLACE INTO live_telemetry_laps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    tuples,
                )
        return pd.DataFrame(df_rows)

    df_custom = _build_custom_scenario_df(st.session_state.custom_scenario_rows, int(scen_lap), sync_db=True)
    scen_drivers = df_custom["driver"].tolist()
    scen_ego = st.sidebar.selectbox("🏎️ Ego Driver to Command in Scenario:", scen_drivers, index=0)

    # Simulate +1 Lap Forward & Top Video Stream Controller for Custom Scenario
    def _advance_custom_scenario_rows(rows: list[dict]) -> list[dict]:
        next_rows = []
        cum_times = []
        for r in rows:
            new_age = int(r["tyre_life"]) + 1
            deg = tyre_cache.lookup_degradation(r["compound"], scen_temp, new_age)
            new_ltime = round(float(r["base_lap_sec"]) + 0.08 * deg, 3)
            tot_t = float(r["gap_to_leader"]) + new_ltime
            cum_times.append((tot_t, r["driver"], r["compound"], new_age, new_ltime))
        cum_times.sort(key=lambda x: x[0])
        leader_t = cum_times[0][0]
        prev_g = 0.0
        for idx, (tot_t, drv, comp, age, ltime) in enumerate(cum_times, 1):
            if idx == 1:
                g = 0.0
            elif scen_status == "SC":
                g = round(max(prev_g + 0.45, (tot_t - leader_t) * 0.65), 3)
            else:
                g = round(max(prev_g + 0.05, tot_t - leader_t), 3)
            interval = 0.0 if idx == 1 else round(max(0.05, g - prev_g), 3)
            prev_g = g
            next_rows.append({
                "pos": idx,
                "driver": drv,
                "compound": comp,
                "tyre_life": age,
                "interval_sec": interval,
                "gap_to_leader": g,
                "base_lap_sec": ltime,
            })
        return next_rows

    # Sidebar Scenario Replay Controls (keeps main canvas clutter-free)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Scenario Replay & Auto-Step")
    p2_speed_options = list(PLAYBACK_SPEED_MULTIPLIERS.keys())
    p2_play_label = "⏸️ PAUSE SCENARIO SYNC" if st.session_state.p2_is_playing else "▶️ AUTO-SIMULATE LAPS"
    if st.sidebar.button(
        p2_play_label,
        key="p2_video_play_pause_btn",
        type="primary" if not st.session_state.p2_is_playing else "secondary",
        use_container_width=True,
    ):
        st.session_state.p2_is_playing = not st.session_state.p2_is_playing
        st.rerun()

    p2_sb1, p2_sb2, p2_sb3 = st.sidebar.columns(3)
    if p2_sb1.button("⏮️ -1L", key="p2_video_prev_lap_btn", use_container_width=True):
        st.session_state._pending_scen_lap = max(1, int(st.session_state.p2_current_lap) - 1)
        st.rerun()
    if p2_sb2.button("⏭️ +1L", key="p2_video_next_lap_btn", use_container_width=True):
        st.session_state._pending_scenario_rows = _advance_custom_scenario_rows(st.session_state.custom_scenario_rows)
        st.session_state._pending_scen_lap = min(int(scen_meta["total_laps"]), int(st.session_state.p2_current_lap) + 1)
        st.rerun()
    if p2_sb3.button("⏩ +25%", key="p2_video_scrub_btn", use_container_width=True):
        new_p2_sub = float(st.session_state.p2_sub_lap_progress) + 0.25
        if new_p2_sub >= 1.0:
            st.session_state.p2_sub_lap_progress = round(new_p2_sub % 1.0, 4)
            st.session_state._pending_scenario_rows = _advance_custom_scenario_rows(st.session_state.custom_scenario_rows)
            st.session_state._pending_scen_lap = min(int(scen_meta["total_laps"]), int(st.session_state.p2_current_lap) + 1)
        else:
            st.session_state.p2_sub_lap_progress = round(new_p2_sub, 4)
        st.session_state._last_p2_slider_pct = int(round(st.session_state.p2_sub_lap_progress * 100)) % 100
        st.rerun()

    p2_speed_label = st.sidebar.selectbox(
        "⏩ Scenario Speed Multiplier",
        p2_speed_options,
        index=p2_speed_options.index(st.session_state.playback_speed_label)
        if st.session_state.playback_speed_label in p2_speed_options
        else 2,
        key="p2_video_speed_select",
    )
    st.session_state.playback_speed_label = p2_speed_label
    p2_speed_mult = PLAYBACK_SPEED_MULTIPLIERS[p2_speed_label]

    # =====================================================================
    # SPLIT-SCREEN COCKPIT: 60 FPS SCENARIO ROLLOUT (LEFT) + PIT WALL RADIO COMMS (RIGHT)
    # =====================================================================
    _p2_x, _p2_y, _p2_curv = get_circuit_curvature_profile(scen_circuit, n_points=240)
    _p2_laps_map: dict[int, list[dict]] = {}
    _sim_rows = list(st.session_state.custom_scenario_rows)
    _p2_start_lap = int(st.session_state.get("p2_current_lap", scen_lap))
    _p2_end_lap = min(int(scen_meta["total_laps"]), _p2_start_lap + 15)
    for _l_num in range(_p2_start_lap, _p2_end_lap + 1):
        _p2_laps_map[_l_num] = [
            {
                "driver": str(r["driver"]),
                "position": int(r["pos"]),
                "compound": str(r["compound"]).upper(),
                "tyre_life": int(r["tyre_life"]),
                "gap_to_leader": float(r["gap_to_leader"]),
                "lap_time_sec": float(r["base_lap_sec"]),
                "track_status": str(scen_status),
            }
            for r in _sim_rows
        ]
        if _l_num < _p2_end_lap:
            _sim_rows = _advance_custom_scenario_rows(_sim_rows)

    _p2_ego_info = next((r for r in st.session_state.custom_scenario_rows if r["driver"] == scen_ego), st.session_state.custom_scenario_rows[0])
    _p2_micro_tel = generate_driver_micro_telemetry(
        driver=scen_ego,
        circuit=scen_circuit,
        lap=_p2_start_lap,
        compound=str(_p2_ego_info["compound"]),
        tyre_life=int(_p2_ego_info["tyre_life"]),
        track_status=scen_status,
    )
    _p2_pit_sim_top = get_cached_pit_simulator(scen_circuit, scen_pit_box)
    _p2_pit_loss_top = float(_p2_pit_sim_top.circuit_data["normal_loss"]) * (
        0.46 if scen_status == "SC" else (0.68 if scen_status == "VSC" else 1.0)
    )
    _p2_canvas_payload = build_broadcast_payload(
        circuit=scen_circuit,
        race_name=scen_meta["race_name"],
        season=int(st.session_state.selected_season),
        session_label="Custom Scenario Rollout",
        x_track=_p2_x,
        y_track=_p2_y,
        norm_curv=_p2_curv,
        track_length_m=float(scen_meta.get("track_length_m", 5412.0)),
        base_lap_sec=float(scen_meta["base_lap_sec"]),
        total_laps=int(scen_meta["total_laps"]),
        initial_lap=_p2_start_lap,
        initial_sub_lap=float(st.session_state.p2_sub_lap_progress),
        ego_driver=scen_ego,
        pit_loss_sec=_p2_pit_loss_top,
        track_temp_c=float(scen_temp),
        weather_cond=scen_weather,
        laps_by_number=_p2_laps_map,
        driver_meta=DRIVER_METADATA_MAP,
        micro_tel_df=_p2_micro_tel,
        auto_play=False,
        initial_speed=PLAYBACK_SPEED_MULTIPLIERS.get(st.session_state.playback_speed_label, 15.0),
    )

    col_p2_stage, col_p2_comms = st.columns([1.85, 1.0], gap="medium")
    with col_p2_stage:
        components.html(render_smooth_broadcast_html(_p2_canvas_payload), height=642, scrolling=False)

    with col_p2_comms:
        def _build_p2_radio_ctx() -> RaceContext:
            live_p2_lap = int(st.session_state.get("p2_current_lap", scen_lap))
            live_p2_df = _build_custom_scenario_df(
                st.session_state.custom_scenario_rows,
                live_p2_lap,
                sync_db=False,
            )
            return RaceContext(
                driver=scen_ego,
                circuit=scen_circuit,
                total_race_laps=int(scen_meta["total_laps"]),
                team_pit_box_position=int(scen_pit_box),
                weather_condition=scen_weather,
                live_data=build_live_data_from_standings_df(
                    live_p2_df,
                    circuit=scen_circuit,
                    total_race_laps=int(scen_meta["total_laps"]),
                    weather_condition=scen_weather,
                ),
            )

        render_pit_wall_radio_console(
            key_prefix="p2",
            history_state_key="p2_radio_history",
            ego_driver=scen_ego,
            current_lap=_p2_start_lap,
            max_laps=int(scen_meta["total_laps"]),
            track_status=scen_status,
            target_tyre=scen_target_tyre,
            build_context_fn=_build_p2_radio_ctx,
        )

    _p2_run_every = 0.25 if st.session_state.p2_is_playing else None

    @st.fragment(run_every=_p2_run_every)
    def render_page2_live_scenario_fragment():
        lap_wrapped = False
        if st.session_state.p2_is_playing:
            base_s = max(60.0, float(scen_meta["base_lap_sec"]))
            d_frac = (0.25 * float(p2_speed_mult)) / base_s
            next_p2_sub = float(st.session_state.p2_sub_lap_progress) + d_frac
            total_p2_laps = int(scen_meta["total_laps"])
            if next_p2_sub >= 1.0:
                laps_inc = int(next_p2_sub // 1.0)
                curr_p2_lap = int(st.session_state.p2_current_lap)
                steps_to_run = min(laps_inc, max(0, total_p2_laps - curr_p2_lap))
                if steps_to_run > 0:
                    updated_rows = st.session_state.custom_scenario_rows
                    for _ in range(steps_to_run):
                        updated_rows = _advance_custom_scenario_rows(updated_rows)
                    st.session_state.custom_scenario_rows = updated_rows
                    st.session_state._pending_scenario_rows = updated_rows
                    next_p2_lap = curr_p2_lap + steps_to_run
                    st.session_state.p2_current_lap = next_p2_lap
                    st.session_state._pending_scen_lap = next_p2_lap
                    st.session_state.p2_sub_lap_progress = round(next_p2_sub % 1.0, 4)
                    lap_wrapped = True
                else:
                    # Reached final checkered-flag lap: hold at 99% and pause playback
                    st.session_state.p2_sub_lap_progress = 0.99
                    st.session_state.p2_is_playing = False
            else:
                st.session_state.p2_sub_lap_progress = round(next_p2_sub, 4)
            st.session_state._last_p2_slider_pct = int(round(st.session_state.p2_sub_lap_progress * 100)) % 100

        active_p2_lap = int(st.session_state.p2_current_lap)
        p2_sub = float(st.session_state.p2_sub_lap_progress)

        # Always reflect current simulated scenario rows and sync DuckDB whenever auto-play advances a lap
        if lap_wrapped or active_p2_lap != int(scen_lap):
            df_custom_base = _build_custom_scenario_df(
                st.session_state.custom_scenario_rows,
                active_p2_lap,
                sync_db=lap_wrapped,
            )
        else:
            df_custom_base = df_custom.copy()

        scen_decision = get_cached_driver_decision(
            driver=scen_ego,
            df_base_standings=df_custom_base,
            circuit=scen_circuit,
            track_status=scen_status,
            track_temp=scen_temp,
            pit_box_num=scen_pit_box,
            target_tyre=scen_target_tyre,
            total_laps=int(scen_meta["total_laps"]),
            weather_condition=scen_weather,
        )

        st.markdown("---")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.info(f"**🎙️ SCENARIO DRIVER RADIO ({scen_ego})**\n\n{scen_decision['driver_radio_msg']}")
        with d_c2:
            st.success(f"**🧠 SCENARIO PIT WALL CALL**\n\n**{scen_decision['team_decision']}**")

        # Side-by-Side Analytics: Tyre Degradation Curves & 10k Monte Carlo Pit Rejoin Window
        st.markdown("---")
        p2_col_deg, p2_col_mc = st.columns([1.05, 1.05], gap="medium")
        with p2_col_deg:
            st.subheader(f"📈 L1 Tensor Tyre Degradation — {scen_circuit} ({scen_temp:.1f}°C)")
            p2_laps_axis = np.arange(1, int(scen_meta["total_laps"]) + 1)
            fig_p2_deg = go.Figure()
            p2_deg_curves = get_cached_degradation_curves(round(float(scen_temp), 1), int(scen_meta["total_laps"]))
            for comp, (color, dash_style) in {
                "SOFT": ("#FF3333", "solid"),
                "MEDIUM": ("#FFD700", "dash"),
                "HARD": ("#00E5FF", "dot"),
            }.items():
                fig_p2_deg.add_trace(
                    go.Scatter(
                        x=p2_laps_axis,
                        y=p2_deg_curves[comp],
                        mode="lines",
                        name=f"{comp} Compound",
                        line=dict(color=color, width=3, dash=dash_style),
                    )
                )
            fig_p2_deg.add_trace(
                go.Scatter(
                    x=[scen_decision["tyre_life"]],
                    y=[scen_decision["deg"]],
                    mode="markers+text",
                    name=f"{scen_ego} Now ({scen_decision['tyre_life']}L)",
                    text=[f"  {scen_ego} (+{scen_decision['deg']:.2f}s)"],
                    textposition="middle right",
                    marker=dict(size=13, color="#00FF66", symbol="diamond", line=dict(width=2, color="white")),
                )
            )
            fig_p2_deg.add_hline(y=2.15, line_dash="dot", line_color="#FF8800", annotation_text="Pit Window (+2.15s/L)")
            fig_p2_deg.update_layout(
                template="plotly_dark",
                height=370,
                margin=dict(l=20, r=20, t=25, b=20),
                xaxis_title="Tyre Age (Laps)",
                yaxis_title="Relative Degradation Delta (+s/lap)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                uirevision="vre_p2_deg_constant",
            )
            st.plotly_chart(fig_p2_deg, use_container_width=True)

        with p2_col_mc:
            st.subheader(
                f"🎯 10k Monte Carlo Pit Rejoin — {scen_circuit} (Lap {active_p2_lap}/{scen_meta['total_laps']} · {scen_status})"
            )
            ego_row_p2 = df_custom_base[df_custom_base["driver"] == scen_ego].iloc[0]
            p2_pit_sim = get_cached_pit_simulator(scen_circuit, scen_pit_box)
            p2_samples = p2_pit_sim._sample_pit_loss_vectorized(scen_status, 10_000) + float(ego_row_p2["gap_to_leader"])
            fig_p2_mc = go.Figure()
            fig_p2_mc.add_trace(
                go.Histogram(x=p2_samples, nbinsx=45, marker_color="#00E5FF", opacity=0.75)
            )
            for car in df_custom_base.head(10).itertuples(index=False):
                if car.driver != scen_ego:
                    fig_p2_mc.add_vline(
                        x=float(car.gap_to_leader),
                        line_dash="dash",
                        line_color="#FF4B4B",
                        annotation_text=f"{car.driver} (+{float(car.gap_to_leader):.1f}s)",
                        annotation_position="top",
                    )
            fig_p2_mc.update_layout(
                template="plotly_dark",
                height=370,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Track Gap to Leader Upon Pit Exit (Seconds)",
                yaxis_title="Monte Carlo Outcomes (out of 10,000)",
                showlegend=False,
                uirevision="vre_p2_mc_constant",
            )
            st.plotly_chart(fig_p2_mc, use_container_width=True)

        # Championship Standings Projection for Custom Scenario Grid (memoized across 4 Hz ticks)
        st.markdown("---")
        p2_ego_team = DRIVER_METADATA_MAP.get(scen_ego, {"team": "Red Bull Racing"})["team"]
        p2_champ_view = get_cached_championship_view(
            season=st.session_state.selected_season,
            circuit=scen_circuit,
            df_standings=df_custom_base,
            ego_driver=scen_ego,
            ego_team_name=p2_ego_team,
        )
        p2_champ = p2_champ_view["champ_data"]
        st.subheader(
            f"🏆 Live Projected Championship Standings (WDC & WCC) — Round {p2_champ['round']}: {p2_champ['race_name']}"
        )
        p2_tab_wdc, p2_tab_wcc = st.tabs([
            "🏎️ Drivers' Championship (WDC) Projection",
            "🛠️ Constructors' Championship (WCC) Projection",
        ])
        with p2_tab_wdc:
            st.markdown(p2_champ_view["wdc_html"], unsafe_allow_html=True)
        with p2_tab_wcc:
            st.markdown(p2_champ_view["wcc_html"], unsafe_allow_html=True)

    render_page2_live_scenario_fragment()
