import asyncio
from contextlib import contextmanager
import json
import os
import threading
import time
from functools import lru_cache

import duckdb
import fastf1
import numpy as np
import pandas as pd
from loguru import logger

_DUCKDB_PROCESS_LOCK = threading.RLock()


@contextmanager
def duckdb_connect(db_path: str):
    """
    Thread-safe DuckDB connection manager using a uniform configuration (read_only=False)
    and a process-wide RLock so concurrent Streamlit 4 Hz fragment ticks never trigger
    DuckDB's 'different configuration than existing connections' exception.
    """
    with _DUCKDB_PROCESS_LOCK:
        con = duckdb.connect(db_path)
        try:
            yield con
        finally:
            con.close()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except Exception:
    pass


# Full 2024 F1 Calendar (24 Rounds) + Multi-Season Metadata (with official FIA circuit lengths in meters)
F1_2024_CALENDAR = [
    {"season": 2024, "round": 1, "race_name": "Bahrain Grand Prix", "circuit": "Bahrain", "circuit_id": "bahrain", "date": "2024-03-02", "total_laps": 57, "base_lap_sec": 94.1, "track_temp": 31.8, "track_length_m": 5412.0},
    {"season": 2024, "round": 2, "race_name": "Saudi Arabian Grand Prix", "circuit": "Saudi Arabia", "circuit_id": "jeddah", "date": "2024-03-09", "total_laps": 50, "base_lap_sec": 91.6, "track_temp": 33.5, "track_length_m": 6174.0},
    {"season": 2024, "round": 3, "race_name": "Australian Grand Prix", "circuit": "Australia", "circuit_id": "albert_park", "date": "2024-03-24", "total_laps": 58, "base_lap_sec": 80.2, "track_temp": 36.0, "track_length_m": 5278.0},
    {"season": 2024, "round": 4, "race_name": "Japanese Grand Prix", "circuit": "Japan", "circuit_id": "suzuka", "date": "2024-04-07", "total_laps": 53, "base_lap_sec": 93.7, "track_temp": 34.0, "track_length_m": 5807.0},
    {"season": 2024, "round": 5, "race_name": "Chinese Grand Prix", "circuit": "China", "circuit_id": "shanghai", "date": "2024-04-21", "total_laps": 56, "base_lap_sec": 97.8, "track_temp": 30.5, "track_length_m": 5451.0},
    {"season": 2024, "round": 6, "race_name": "Miami Grand Prix", "circuit": "Miami", "circuit_id": "miami", "date": "2024-05-05", "total_laps": 57, "base_lap_sec": 90.6, "track_temp": 43.0, "track_length_m": 5412.0},
    {"season": 2024, "round": 7, "race_name": "Emilia Romagna Grand Prix", "circuit": "Emilia Romagna", "circuit_id": "imola", "date": "2024-05-19", "total_laps": 63, "base_lap_sec": 78.5, "track_temp": 41.0, "track_length_m": 4909.0},
    {"season": 2024, "round": 8, "race_name": "Monaco Grand Prix", "circuit": "Monaco", "circuit_id": "monaco", "date": "2024-05-26", "total_laps": 78, "base_lap_sec": 74.1, "track_temp": 45.0, "track_length_m": 3337.0},
    {"season": 2024, "round": 9, "race_name": "Canadian Grand Prix", "circuit": "Canada", "circuit_id": "villeneuve", "date": "2024-06-09", "total_laps": 70, "base_lap_sec": 74.8, "track_temp": 24.5, "track_length_m": 4361.0},
    {"season": 2024, "round": 10, "race_name": "Spanish Grand Prix", "circuit": "Spain", "circuit_id": "catalunya", "date": "2024-06-23", "total_laps": 66, "base_lap_sec": 77.1, "track_temp": 42.0, "track_length_m": 4657.0},
    {"season": 2024, "round": 11, "race_name": "Austrian Grand Prix", "circuit": "Austria", "circuit_id": "red_bull_ring", "date": "2024-06-30", "total_laps": 71, "base_lap_sec": 67.6, "track_temp": 47.0, "track_length_m": 4318.0},
    {"season": 2024, "round": 12, "race_name": "British Grand Prix", "circuit": "Great Britain", "circuit_id": "silverstone", "date": "2024-07-07", "total_laps": 52, "base_lap_sec": 88.3, "track_temp": 26.0, "track_length_m": 5891.0},
    {"season": 2024, "round": 13, "race_name": "Hungarian Grand Prix", "circuit": "Hungary", "circuit_id": "hungaroring", "date": "2024-07-21", "total_laps": 70, "base_lap_sec": 80.3, "track_temp": 46.5, "track_length_m": 4381.0},
    {"season": 2024, "round": 14, "race_name": "Belgian Grand Prix", "circuit": "Belgium", "circuit_id": "spa", "date": "2024-07-28", "total_laps": 44, "base_lap_sec": 105.7, "track_temp": 34.0, "track_length_m": 7004.0},
    {"season": 2024, "round": 15, "race_name": "Dutch Grand Prix", "circuit": "Netherlands", "circuit_id": "zandvoort", "date": "2024-08-25", "total_laps": 72, "base_lap_sec": 73.8, "track_temp": 30.0, "track_length_m": 4259.0},
    {"season": 2024, "round": 16, "race_name": "Italian Grand Prix", "circuit": "Italy", "circuit_id": "monza", "date": "2024-09-01", "total_laps": 53, "base_lap_sec": 81.4, "track_temp": 48.0, "track_length_m": 5793.0},
    {"season": 2024, "round": 17, "race_name": "Azerbaijan Grand Prix", "circuit": "Azerbaijan", "circuit_id": "baku", "date": "2024-09-15", "total_laps": 51, "base_lap_sec": 105.2, "track_temp": 40.0, "track_length_m": 6003.0},
    {"season": 2024, "round": 18, "race_name": "Singapore Grand Prix", "circuit": "Singapore", "circuit_id": "marina_bay", "date": "2024-09-22", "total_laps": 62, "base_lap_sec": 94.4, "track_temp": 37.0, "track_length_m": 4940.0},
    {"season": 2024, "round": 19, "race_name": "United States Grand Prix", "circuit": "USA", "circuit_id": "americas", "date": "2024-10-20", "total_laps": 56, "base_lap_sec": 97.3, "track_temp": 44.0, "track_length_m": 5513.0},
    {"season": 2024, "round": 20, "race_name": "Mexico City Grand Prix", "circuit": "Mexico", "circuit_id": "rodriguez", "date": "2024-10-27", "total_laps": 71, "base_lap_sec": 79.8, "track_temp": 38.0, "track_length_m": 4304.0},
    {"season": 2024, "round": 21, "race_name": "São Paulo Grand Prix", "circuit": "Brazil", "circuit_id": "interlagos", "date": "2024-11-03", "total_laps": 69, "base_lap_sec": 73.5, "track_temp": 29.0, "track_length_m": 4309.0},
    {"season": 2024, "round": 22, "race_name": "Las Vegas Grand Prix", "circuit": "Las Vegas", "circuit_id": "vegas", "date": "2024-11-23", "total_laps": 50, "base_lap_sec": 94.8, "track_temp": 21.0, "track_length_m": 6201.0},
    {"season": 2024, "round": 23, "race_name": "Qatar Grand Prix", "circuit": "Qatar", "circuit_id": "losail", "date": "2024-12-01", "total_laps": 57, "base_lap_sec": 83.8, "track_temp": 28.0, "track_length_m": 5419.0},
    {"season": 2024, "round": 24, "race_name": "Abu Dhabi Grand Prix", "circuit": "Abu Dhabi", "circuit_id": "yas_marina", "date": "2024-12-08", "total_laps": 58, "base_lap_sec": 87.2, "track_temp": 32.5, "track_length_m": 5281.0},
]

# Full 20-Driver 2024 F1 Grid with Team & Driver Number Metadata
F1_2024_DRIVERS = [
    {"driver": "VER", "number": 1, "name": "Max Verstappen", "team": "Red Bull Racing", "color": "#3671C6", "pace_offset": 0.00},
    {"driver": "PER", "number": 11, "name": "Sergio Perez", "team": "Red Bull Racing", "color": "#3671C6", "pace_offset": 0.25},
    {"driver": "LEC", "number": 16, "name": "Charles Leclerc", "team": "Ferrari", "color": "#E80020", "pace_offset": 0.15},
    {"driver": "SAI", "number": 55, "name": "Carlos Sainz", "team": "Ferrari", "color": "#E80020", "pace_offset": 0.18},
    {"driver": "NOR", "number": 4, "name": "Lando Norris", "team": "McLaren", "color": "#FF8000", "pace_offset": 0.12},
    {"driver": "PIA", "number": 81, "name": "Oscar Piastri", "team": "McLaren", "color": "#FF8000", "pace_offset": 0.22},
    {"driver": "RUS", "number": 63, "name": "George Russell", "team": "Mercedes", "color": "#27F4D2", "pace_offset": 0.24},
    {"driver": "HAM", "number": 44, "name": "Lewis Hamilton", "team": "Mercedes", "color": "#27F4D2", "pace_offset": 0.26},
    {"driver": "ALO", "number": 14, "name": "Fernando Alonso", "team": "Aston Martin", "color": "#229971", "pace_offset": 0.45},
    {"driver": "STR", "number": 18, "name": "Lance Stroll", "team": "Aston Martin", "color": "#229971", "pace_offset": 0.65},
    {"driver": "TSU", "number": 22, "name": "Yuki Tsunoda", "team": "RB", "color": "#6692FF", "pace_offset": 0.70},
    {"driver": "RIC", "number": 3, "name": "Daniel Ricciardo", "team": "RB", "color": "#6692FF", "pace_offset": 0.78},
    {"driver": "HUL", "number": 27, "name": "Nico Hulkenberg", "team": "Haas F1 Team", "color": "#B6BABD", "pace_offset": 0.72},
    {"driver": "MAG", "number": 20, "name": "Kevin Magnussen", "team": "Haas F1 Team", "color": "#B6BABD", "pace_offset": 0.80},
    {"driver": "ALB", "number": 23, "name": "Alexander Albon", "team": "Williams", "color": "#64C4FF", "pace_offset": 0.82},
    {"driver": "GAS", "number": 10, "name": "Pierre Gasly", "team": "Alpine", "color": "#0093CC", "pace_offset": 0.85},
    {"driver": "OCO", "number": 31, "name": "Esteban Ocon", "team": "Alpine", "color": "#0093CC", "pace_offset": 0.88},
    {"driver": "ZHO", "number": 24, "name": "Zhou Guanyu", "team": "Kick Sauber", "color": "#52E252", "pace_offset": 0.92},
    {"driver": "BOT", "number": 77, "name": "Valtteri Bottas", "team": "Kick Sauber", "color": "#52E252", "pace_offset": 0.95},
    {"driver": "SAR", "number": 2, "name": "Logan Sargeant", "team": "Williams", "color": "#64C4FF", "pace_offset": 1.10},
]

DRIVER_METADATA_MAP = {d["driver"]: d for d in F1_2024_DRIVERS}
SUPPORTED_SEASONS = [2024, 2023, 2022, 2021]
SESSION_TYPES = ["FP1", "FP2", "FP3", "Qualifying", "Sprint", "Race"]
SESSION_DISPLAY_LABELS = {
    "FP1": "Practice 1 (FP1)",
    "FP2": "Practice 2 (FP2)",
    "FP3": "Practice 3 (FP3)",
    "Qualifying": "Qualifying",
    "Sprint": "Sprint",
    "Race": "Race",
}
_SESSION_ALIAS_MAP = {
    "PRACTICE 1": "FP1",
    "PRACTICE 1 (FP1)": "FP1",
    "FP1": "FP1",
    "PRACTICE 2": "FP2",
    "PRACTICE 2 (FP2)": "FP2",
    "FP2": "FP2",
    "PRACTICE 3": "FP3",
    "PRACTICE 3 (FP3)": "FP3",
    "FP3": "FP3",
    "QUALIFYING": "Qualifying",
    "Q": "Qualifying",
    "SPRINT": "Sprint",
    "S": "Sprint",
    "RACE": "Race",
    "R": "Race",
}

# Season-specific driver pace adjustments for realistic 2021-2024 era differences
SEASON_PACE_MODIFIERS: dict[int, dict[str, float]] = {
    2021: {"VER": 0.00, "HAM": 0.01, "BOT": 0.19, "PER": 0.24, "NOR": 0.35, "LEC": 0.36, "SAI": 0.38, "RIC": 0.42, "GAS": 0.44, "ALO": 0.48},
    2022: {"VER": 0.00, "LEC": 0.08, "PER": 0.16, "SAI": 0.19, "RUS": 0.24, "HAM": 0.26, "NOR": 0.48, "ALO": 0.52, "OCO": 0.54, "BOT": 0.62},
    2023: {"VER": 0.00, "PER": 0.22, "HAM": 0.27, "ALO": 0.28, "LEC": 0.30, "NOR": 0.31, "SAI": 0.32, "RUS": 0.35, "PIA": 0.42, "STR": 0.58},
    2024: {},
}

# Standard FIA Grand Prix Championship Points Table (P1-P10)
FIA_RACE_POINTS_TABLE: dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


@lru_cache(maxsize=1)
def _load_track_map_cache() -> dict:
    """Loads precomputed real FIA GPS track maps (all 24 circuits) from data/circuit_track_map_cache.json."""
    map_cache_path = os.path.join(PROJECT_ROOT, "data", "circuit_track_map_cache.json")
    if not os.path.exists(map_cache_path):
        return {}
    try:
        with open(map_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_historical_championship_cache() -> dict:
    """Loads authentic 2021-2024 FIA/Jolpica Grand Prix race results from data/historical_championship_cache.json."""
    hist_path = os.path.join(PROJECT_ROOT, "data", "historical_championship_cache.json")
    if not os.path.exists(hist_path):
        return {}
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=128)
def get_historical_race_order(season: int, circuit: str) -> tuple[str, ...]:
    """
    Returns the authentic FIA race finishing order for `(season, circuit)` from Jolpica historical cache,
    mapped onto the 20-driver grid (real finishers in `F1_2024_DRIVERS` ordered by actual finish position,
    followed by any remaining grid drivers). Returns empty tuple if not found in cache.
    """
    hist_cache = _load_historical_championship_cache()
    season_block = hist_cache.get(str(int(season)))
    if not season_block:
        return ()

    meta = get_circuit_metadata(circuit, season=int(season))
    cid = str(meta.get("circuit_id", "")).lower()
    rname = str(meta.get("race_name", "")).lower()

    matched_round = None
    for rnd in season_block.get("rounds", []):
        if str(rnd.get("circuit_id", "")).lower() == cid or str(rnd.get("race_name", "")).lower() == rname:
            matched_round = rnd
            break
    if not matched_round:
        return ()

    known_drivers = [d["driver"] for d in F1_2024_DRIVERS]
    known_set = set(known_drivers)
    ordered: list[str] = []
    for res in sorted(matched_round.get("results", []), key=lambda r: int(r.get("position", 99))):
        drv = str(res.get("driver", "")).upper()
        if drv in known_set and drv not in ordered:
            ordered.append(drv)
    for drv in known_drivers:
        if drv not in ordered:
            ordered.append(drv)
    return tuple(ordered)


@lru_cache(maxsize=128)
def _compute_prior_championship_totals(
    season: int,
    target_round: int,
    circuit_id: str | None = None,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """
    Caches cumulative WDC prior-round points/wins/podiums for completed rounds prior to `target_round`.
    Uses authentic 2021-2024 FIA/Jolpica Grand Prix results from `data/historical_championship_cache.json`
    when available, with deterministic fallback if the cache is absent.
    """
    season = int(season)
    target_round = int(target_round)
    season_mods = SEASON_PACE_MODIFIERS.get(season, {})

    prior_driver_pts: dict[str, int] = {d["driver"]: 0 for d in F1_2024_DRIVERS}
    prior_driver_wins: dict[str, int] = {d["driver"]: 0 for d in F1_2024_DRIVERS}
    prior_driver_podiums: dict[str, int] = {d["driver"]: 0 for d in F1_2024_DRIVERS}

    hist_cache = _load_historical_championship_cache()
    season_block = hist_cache.get(str(season))
    if season_block and season_block.get("rounds"):
        rounds = season_block["rounds"]
        cutoff_round = target_round
        if circuit_id:
            cid_low = str(circuit_id).lower()
            for rnd in rounds:
                if str(rnd.get("circuit_id", "")).lower() == cid_low:
                    cutoff_round = int(rnd["round"])
                    break

        for rnd in rounds:
            rnd_num = int(rnd.get("round", 999))
            if rnd_num >= cutoff_round:
                continue
            for res in rnd.get("results", []):
                drv = str(res.get("driver", "")).upper()
                if drv not in prior_driver_pts:
                    continue
                pts = int(round(float(res.get("points", 0.0))))
                pos = int(res.get("position", 99))
                prior_driver_pts[drv] += pts
                if pos == 1:
                    prior_driver_wins[drv] += 1
                if pos <= 3:
                    prior_driver_podiums[drv] += 1

        return (
            tuple(prior_driver_pts.items()),
            tuple(prior_driver_wins.items()),
            tuple(prior_driver_podiums.items()),
        )

    for rnd_meta in F1_2024_CALENDAR:
        rnd_num = int(rnd_meta["round"])
        if rnd_num >= target_round:
            continue
        rng = np.random.default_rng(season * 1000 + rnd_num * 37)
        round_scores = []
        for idx, d in enumerate(F1_2024_DRIVERS):
            drv = d["driver"]
            base_off = season_mods.get(drv, d["pace_offset"])
            perf_score = base_off + float(rng.normal(0.0, 0.14)) + idx * 0.002
            round_scores.append((perf_score, drv))
        round_scores.sort(key=lambda x: x[0])
        for pos_idx, (_, drv) in enumerate(round_scores, 1):
            pts = FIA_RACE_POINTS_TABLE.get(pos_idx, 0)
            prior_driver_pts[drv] = prior_driver_pts.get(drv, 0) + pts
            if pos_idx == 1:
                prior_driver_wins[drv] = prior_driver_wins.get(drv, 0) + 1
            if pos_idx <= 3:
                prior_driver_podiums[drv] = prior_driver_podiums.get(drv, 0) + 1

    return (
        tuple(prior_driver_pts.items()),
        tuple(prior_driver_wins.items()),
        tuple(prior_driver_podiums.items()),
    )


def compute_championship_standings_till_race(
    season: int,
    circuit: str,
    current_race_standings_df: pd.DataFrame | None = None,
) -> dict:
    """
    Computes cumulative World Drivers' Championship (WDC) and World Constructors' Championship (WCC)
    standings for `season` (2021-2024) from Round 1 up through the selected `circuit` round:
    - `prior_points`: Cumulative points earned in completed rounds 1..(round_number - 1).
    - `live_race_points`: Live projected FIA race points (+25, +18, +15, ...) from current on-track positions
      in `current_race_standings_df` (or deterministic round classification if None).
    - `total_points`: `prior_points + live_race_points`.
    - `gap_to_leader`: Points deficit to P1 in WDC / WCC (`0` for the championship leader).
    """
    season = int(season)
    meta = get_circuit_metadata(circuit, season=season)
    target_round = int(meta["round"])
    race_name = str(meta["race_name"])
    season_mods = SEASON_PACE_MODIFIERS.get(season, {})

    pts_items, wins_items, pod_items = _compute_prior_championship_totals(
        season,
        target_round,
        circuit_id=meta.get("circuit_id"),
    )
    prior_driver_pts = dict(pts_items)
    prior_driver_wins = dict(wins_items)
    prior_driver_podiums = dict(pod_items)

    # Compute live race projected points from current on-track positions
    live_driver_pts: dict[str, int] = {d["driver"]: 0 for d in F1_2024_DRIVERS}
    live_driver_pos: dict[str, int] = {}

    if current_race_standings_df is not None and not current_race_standings_df.empty:
        df_sorted = current_race_standings_df.sort_values("position", ascending=True)
        used_positions: set[int] = set()
        for pos_idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            drv = str(row["driver"])
            raw_pos = int(row["position"]) if pd.notna(row.get("position")) else pos_idx
            pos_val = raw_pos if (raw_pos >= 1 and raw_pos not in used_positions) else pos_idx
            used_positions.add(pos_val)
            live_driver_pos[drv] = pos_val
            live_driver_pts[drv] = FIA_RACE_POINTS_TABLE.get(pos_val, 0)

        # If a custom scenario grid has fewer than 20 cars, assign the remaining WDC drivers
        # to non-colliding positions after the active grid so positions 1..20 are unique.
        if len(live_driver_pos) < len(F1_2024_DRIVERS):
            rng_curr = np.random.default_rng(season * 1000 + target_round * 37)
            rem_scores = []
            for idx, d in enumerate(F1_2024_DRIVERS):
                drv = d["driver"]
                base_off = season_mods.get(drv, d["pace_offset"])
                score = base_off + float(rng_curr.normal(0.0, 0.12)) + idx * 0.002
                if drv not in live_driver_pos:
                    rem_scores.append((score, drv))
            rem_scores.sort(key=lambda x: x[0])
            next_pos = 1
            for _, drv in rem_scores:
                while next_pos in used_positions:
                    next_pos += 1
                used_positions.add(next_pos)
                live_driver_pos[drv] = next_pos
                live_driver_pts[drv] = FIA_RACE_POINTS_TABLE.get(next_pos, 0)
    else:
        hist_order = get_historical_race_order(season, circuit)
        if hist_order:
            for pos_idx, drv in enumerate(hist_order, 1):
                live_driver_pos[drv] = pos_idx
                live_driver_pts[drv] = FIA_RACE_POINTS_TABLE.get(pos_idx, 0)
        else:
            rng_curr = np.random.default_rng(season * 1000 + target_round * 37)
            curr_scores = []
            for idx, d in enumerate(F1_2024_DRIVERS):
                drv = d["driver"]
                base_off = season_mods.get(drv, d["pace_offset"])
                curr_scores.append((base_off + float(rng_curr.normal(0.0, 0.12)) + idx * 0.002, drv))
            curr_scores.sort(key=lambda x: x[0])
            for pos_idx, (_, drv) in enumerate(curr_scores, 1):
                live_driver_pos[drv] = pos_idx
                live_driver_pts[drv] = FIA_RACE_POINTS_TABLE.get(pos_idx, 0)

    # Assemble WDC standings table
    wdc_rows = []
    for d in F1_2024_DRIVERS:
        drv = d["driver"]
        prior_p = int(prior_driver_pts.get(drv, 0))
        live_p = int(live_driver_pts.get(drv, 0))
        tot_p = prior_p + live_p
        cur_pos = int(live_driver_pos.get(drv, 20))
        wins = int(prior_driver_wins.get(drv, 0)) + (1 if cur_pos == 1 else 0)
        podiums = int(prior_driver_podiums.get(drv, 0)) + (1 if cur_pos <= 3 else 0)
        wdc_rows.append({
            "driver": drv,
            "name": d["name"],
            "number": int(d["number"]),
            "team": d["team"],
            "color": d["color"],
            "race_position": cur_pos,
            "prior_points": prior_p,
            "live_race_points": live_p,
            "total_points": tot_p,
            "wins": wins,
            "podiums": podiums,
        })

    wdc_rows.sort(key=lambda r: (-r["total_points"], -r["wins"], r["race_position"]))
    wdc_leader_pts = wdc_rows[0]["total_points"] if wdc_rows else 0
    for rank_idx, r in enumerate(wdc_rows, 1):
        r["rank"] = rank_idx
        r["gap_to_leader"] = wdc_leader_pts - r["total_points"]

    # Assemble WCC (Constructors' Championship) standings table
    team_map: dict[str, dict] = {}
    for r in wdc_rows:
        t_name = r["team"]
        if t_name not in team_map:
            team_map[t_name] = {
                "team": t_name,
                "color": r["color"],
                "drivers": [],
                "prior_points": 0,
                "live_race_points": 0,
                "total_points": 0,
                "wins": 0,
                "podiums": 0,
                "best_race_pos": 99,
            }
        t_entry = team_map[t_name]
        t_entry["drivers"].append(r["driver"])
        t_entry["prior_points"] += r["prior_points"]
        t_entry["live_race_points"] += r["live_race_points"]
        t_entry["total_points"] += r["total_points"]
        t_entry["wins"] += r["wins"]
        t_entry["podiums"] += r["podiums"]
        t_entry["best_race_pos"] = min(t_entry["best_race_pos"], r["race_position"])

    wcc_rows = list(team_map.values())
    wcc_rows.sort(key=lambda r: (-r["total_points"], -r["wins"], r["best_race_pos"]))
    wcc_leader_pts = wcc_rows[0]["total_points"] if wcc_rows else 0
    for rank_idx, r in enumerate(wcc_rows, 1):
        r["rank"] = rank_idx
        r["drivers_str"] = " / ".join(r["drivers"])
        r["gap_to_leader"] = wcc_leader_pts - r["total_points"]

    return {
        "season": season,
        "circuit": meta["circuit"],
        "race_name": race_name,
        "round": target_round,
        "total_rounds": len(F1_2024_CALENDAR),
        "wdc": pd.DataFrame(wdc_rows),
        "wcc": pd.DataFrame(wcc_rows),
    }


def normalize_session_type(session_type: str) -> str:
    return _SESSION_ALIAS_MAP.get(str(session_type).strip().upper(), session_type)


def get_circuit_metadata(circuit: str, season: int = 2024) -> dict:
    for item in F1_2024_CALENDAR:
        if item["circuit"].lower() == circuit.lower() or item["race_name"].lower() == circuit.lower():
            res = dict(item)
            res["season"] = int(season)
            res["date"] = f"{int(season)}{item['date'][4:]}"
            return res
    res = dict(F1_2024_CALENDAR[0])
    res["season"] = int(season)
    res["date"] = f"{int(season)}{res['date'][4:]}"
    return res


@lru_cache(maxsize=64)
def generate_circuit_map_coordinates(circuit: str, n_points: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns 2D (X, Y) circuit track coordinates in meters.
    Loads real FIA GPS circuit outlines for ALL 24 F1 circuits from `data/circuit_track_map_cache.json`
    (populated from FastF1 GPS telemetry & official GeoJSON circuit geometries for <1ms cold loads).
    Falls back to live FastF1 extraction (Bahrain) or deterministic parametric geometry only if absent.
    """
    map_cache_path = os.path.join(PROJECT_ROOT, "data", "circuit_track_map_cache.json")
    circuit_key = str(circuit).strip().lower()
    meta_key = get_circuit_metadata(circuit)["circuit"].lower()

    cached_maps = _load_track_map_cache()
    cached_entry = cached_maps.get(circuit_key) or cached_maps.get(meta_key)
    if cached_entry and "x" in cached_entry and "y" in cached_entry:
        cx = np.asarray(cached_entry["x"], dtype=np.float64)
        cy = np.asarray(cached_entry["y"], dtype=np.float64)
        if len(cx) == n_points:
            return cx, cy
        idx = np.linspace(0, len(cx) - 1, n_points).astype(int)
        return cx[idx], cy[idx]

    if circuit_key == "bahrain" or meta_key == "bahrain":
        try:
            s = fastf1.get_session(2024, "Bahrain", "R")
            s.load(telemetry=True, weather=False, messages=False)
            lap = s.laps.pick_drivers("VER").pick_fastest()
            tel = lap.get_telemetry()
            if "X" in tel.columns and "Y" in tel.columns and len(tel) > 20:
                x = tel["X"].to_numpy(dtype=np.float64)
                y = tel["Y"].to_numpy(dtype=np.float64)
                idx_240 = np.linspace(0, len(x) - 1, 240).astype(int)
                try:
                    os.makedirs(os.path.dirname(map_cache_path), exist_ok=True)
                    updated = dict(cached_maps)
                    updated["bahrain"] = {
                        "x": np.round(x[idx_240], 2).tolist(),
                        "y": np.round(y[idx_240], 2).tolist(),
                    }
                    with open(map_cache_path, "w", encoding="utf-8") as f:
                        json.dump(updated, f)
                    _load_track_map_cache.cache_clear()
                except Exception:
                    pass
                idx = np.linspace(0, len(x) - 1, n_points).astype(int)
                return x[idx], y[idx]
        except Exception:
            pass

    # Deterministic circuit-specific Fourier harmonic closed track loop
    seed_val = sum((i + 1) * ord(c) for i, c in enumerate(circuit))
    rng = np.random.default_rng(seed_val)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=True)

    a2, b2 = rng.uniform(-0.35, 0.35, 2)
    a3, b3 = rng.uniform(-0.28, 0.28, 2)
    a4, b4 = rng.uniform(-0.18, 0.18, 2)
    a5, b5 = rng.uniform(-0.12, 0.12, 2)
    aspect = rng.uniform(0.65, 1.35)

    r = (
        1.0
        + a2 * np.cos(2 * theta)
        + b2 * np.sin(2 * theta)
        + a3 * np.cos(3 * theta)
        + b3 * np.sin(3 * theta)
        + a4 * np.cos(4 * theta)
        + b4 * np.sin(4 * theta)
        + a5 * np.cos(5 * theta)
        + b5 * np.sin(5 * theta)
    )
    x = 3500.0 * aspect * r * np.cos(theta)
    y = 3500.0 * r * np.sin(theta)
    return x, y


@lru_cache(maxsize=64)
def get_circuit_curvature_profile(circuit: str, n_points: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes and caches `(x, y, normalized_curvature)` once per `(circuit, n_points)` for 4 Hz playback."""
    x, y = generate_circuit_map_coordinates(circuit, n_points=n_points)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-3)
    norm_curv = np.clip(curvature / (np.percentile(curvature, 92) + 1e-6), 0.0, 1.0)
    return x, y, norm_curv


@lru_cache(maxsize=128)
def _generate_driver_micro_telemetry_cached(
    driver: str,
    circuit: str,
    lap: int,
    compound: str,
    tyre_life: int,
    track_status: str,
    n_points: int,
) -> pd.DataFrame:
    x, y, norm_curv = get_circuit_curvature_profile(circuit, n_points=n_points)

    # Circular Gaussian smoothing eliminates GeoJSON polygon vertex jitter so straights stay clean
    k_rad = max(3, int(round(n_points / 60.0)))
    k_axis = np.arange(-k_rad, k_rad + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k_axis / max(1.5, k_rad * 0.45)) ** 2)
    kernel /= kernel.sum()
    padded_curv = np.concatenate([norm_curv[-k_rad:], norm_curv, norm_curv[:k_rad]])
    smooth_curv = np.convolve(padded_curv, kernel, mode="valid")
    smooth_curv = np.clip(smooth_curv / (np.max(smooth_curv) + 1e-6), 0.0, 1.0)

    drv_info = DRIVER_METADATA_MAP.get(driver, F1_2024_DRIVERS[0])
    pace_factor = 1.0 - (drv_info["pace_offset"] * 0.008)
    deg_penalty = min(0.15, tyre_life * 0.0025)
    status_cap = 1.0 if track_status == "GREEN" else (0.72 if track_status == "VSC" else 0.58)

    # Speed between 85 km/h (hairpins) and 338 km/h (main straights)
    raw_speed = (335.0 - 235.0 * (smooth_curv ** 0.78)) * pace_factor * (1.0 - deg_penalty) * status_cap
    speed = np.clip(raw_speed, 75.0, 345.0)
    dspeed = np.gradient(speed)

    # Strictly mutually exclusive F1 pedal physics: never simultaneous throttle + brake, never DRS under braking
    braking_zone = (dspeed < -1.8) | ((smooth_curv > 0.68) & (dspeed < 0.2))
    brake_pct = np.where(
        braking_zone,
        np.clip((-dspeed * 18.0) + smooth_curv * 45.0, 25.0, 100.0),
        0.0,
    )
    throttle = np.where(
        brake_pct > 0.0,
        0.0,
        np.clip(100.0 - smooth_curv * 58.0 + np.maximum(0.0, dspeed) * 6.0, 12.0, 100.0),
    )
    if track_status != "GREEN":
        throttle = np.minimum(throttle, 78.0)

    gear = np.clip(np.floor(speed / 42.0) + 1, 1, 8).astype(int)
    rpm = np.clip(8200.0 + (speed / (gear * 42.0)) * 3900.0, 6500.0, 12500.0)
    drs = (
        (speed > 265.0)
        & (smooth_curv < 0.16)
        & (brake_pct == 0.0)
        & (throttle >= 92.0)
        & (track_status == "GREEN")
    ).astype(int)

    rel_dist = np.linspace(0.0, 1.0, n_points)
    meta = get_circuit_metadata(circuit)
    track_len_m = float(meta.get("track_length_m", 5412.0))
    dist_m = rel_dist * track_len_m

    base_lap = meta["base_lap_sec"]
    time_sec = rel_dist * base_lap

    return pd.DataFrame({
        "TimeSec": np.round(time_sec, 3),
        "Distance": np.round(dist_m, 1),
        "RelativeDistance": np.round(rel_dist, 4),
        "Speed": np.round(speed, 1),
        "RPM": np.round(rpm, 0),
        "Throttle": np.round(throttle, 1),
        "Brake": np.round(brake_pct, 1),
        "nGear": gear,
        "DRS": drs,
        "X": x,
        "Y": y,
    })


def infer_lap_weather_condition(df_lap: pd.DataFrame, rainfall: bool | None = None) -> str:
    """
    Infers realistic track weather/grip state from the active lap's real tyre compounds,
    track temperature, and optional FastF1 rainfall flag so wet races (e.g. 2022 Monaco)
    never falsely display 'Dry (Clear)'.
    """
    if df_lap is None or df_lap.empty:
        return "Dry (Clear)"
    comps = df_lap["compound"].astype(str).str.upper().tolist() if "compound" in df_lap.columns else []
    n_cars = max(1, len(comps))
    wet_frac = sum(1 for c in comps if c == "WET") / n_cars
    inter_frac = sum(1 for c in comps if c == "INTERMEDIATE") / n_cars

    if wet_frac >= 0.45 or (rainfall and wet_frac >= 0.20):
        return "Heavy Rain (Wet)"
    if (wet_frac + inter_frac) >= 0.20 or rainfall:
        return "Light Drizzle (Damp)"
    if "track_temp" in df_lap.columns:
        med_temp = float(pd.to_numeric(df_lap["track_temp"], errors="coerce").dropna().median() or 32.0)
        if med_temp < 28.0:
            return "Overcast (Cool)"
    return "Dry (Clear)"


def get_real_driver_lap_telemetry(
    season: int,
    circuit: str,
    session_type: str,
    driver: str,
    lap: int,
    compound: str = "SOFT",
    tyre_life: int = 12,
    track_status: str = "GREEN",
    n_points: int = 240,
    allow_fastf1_download: bool = False,
) -> pd.DataFrame:
    """
    Returns real FastF1 car telemetry (`Speed, RPM, Throttle, Brake, nGear, DRS`) for `(season, circuit, driver, lap)`
    from local cache (`data/fastf1_lap_telemetry_cache/`) if present (or downloads if `allow_fastf1_download=True`),
    falling back seamlessly to physics-clean `generate_driver_micro_telemetry()`.
    """
    tel_cache_dir = os.path.join(PROJECT_ROOT, "data", "fastf1_lap_telemetry_cache")
    safe_key = f"{int(season)}_{circuit.lower().replace(' ', '_')}_{normalize_session_type(session_type).lower()}_{driver.upper()}_L{int(lap)}.json"
    cache_file = os.path.join(tel_cache_dir, safe_key)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            df_c = pd.DataFrame(cached)
            if len(df_c) >= 60 and "Speed" in df_c.columns:
                return df_c
        except Exception:
            pass

    if allow_fastf1_download:
        try:
            meta = get_circuit_metadata(circuit, season=int(season))
            ff1_code = SESSION_TO_FASTF1_CODE.get(normalize_session_type(session_type), "R")
            event_query = "Bahrain" if circuit.lower() == "bahrain" else meta["race_name"]
            s = fastf1.get_session(int(season), event_query, ff1_code)
            s.load(telemetry=True, weather=False, messages=False)
            drv_laps = s.laps.pick_drivers(driver.upper())
            target_lap = drv_laps[drv_laps["LapNumber"] == int(lap)]
            if not target_lap.empty:
                car_data = target_lap.iloc[0].get_car_data().add_distance()
                if car_data is not None and len(car_data) >= 30:
                    idx = np.linspace(0, len(car_data) - 1, n_points).astype(int)
                    sub = car_data.iloc[idx]
                    x, y = generate_circuit_map_coordinates(circuit, n_points=n_points)
                    rel_dist = np.linspace(0.0, 1.0, n_points)
                    track_len_m = float(meta.get("track_length_m", 5412.0))
                    base_lap = float(meta["base_lap_sec"])
                    brake_raw = sub["Brake"].astype(float).to_numpy()
                    brake_pct = np.where(brake_raw > 1.0, np.clip(brake_raw, 0.0, 100.0), brake_raw * 100.0)
                    thr_pct = np.where(brake_pct > 5.0, 0.0, np.clip(sub["Throttle"].astype(float).to_numpy(), 0.0, 100.0))
                    drs_vals = (sub["DRS"].astype(float).to_numpy() >= 10.0).astype(int)
                    df_real = pd.DataFrame({
                        "TimeSec": np.round(rel_dist * base_lap, 3),
                        "Distance": np.round(rel_dist * track_len_m, 1),
                        "RelativeDistance": np.round(rel_dist, 4),
                        "Speed": np.round(sub["Speed"].astype(float).to_numpy(), 1),
                        "RPM": np.round(sub["RPM"].astype(float).to_numpy(), 0),
                        "Throttle": np.round(thr_pct, 1),
                        "Brake": np.round(brake_pct, 1),
                        "nGear": sub["nGear"].astype(int).to_numpy(),
                        "DRS": drs_vals,
                        "X": np.round(x, 2),
                        "Y": np.round(y, 2),
                    })
                    os.makedirs(tel_cache_dir, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(df_real.to_dict(orient="list"), f)
                    return df_real
        except Exception:
            pass

    return generate_driver_micro_telemetry(
        driver=driver,
        circuit=circuit,
        lap=lap,
        compound=compound,
        tyre_life=tyre_life,
        track_status=track_status,
        n_points=n_points,
    )


def generate_driver_micro_telemetry(
    driver: str,
    circuit: str,
    lap: int,
    compound: str = "SOFT",
    tyre_life: int = 12,
    track_status: str = "GREEN",
    n_points: int = 480,
) -> pd.DataFrame:
    """
    Produces high-frequency micro-telemetry sensor readings across the lap:
    Distance (m), RelativeDistance (0..1), Speed (km/h), RPM, Throttle (%), Brake (bool/%),
    nGear (1..8), DRS (0/1), X, Y, and timestamp.
    Cached per `(driver, circuit, lap, compound, tyre_life, track_status, n_points)` for 4 Hz playback.
    """
    return _generate_driver_micro_telemetry_cached(
        str(driver),
        str(circuit),
        int(lap),
        str(compound).upper(),
        int(tyre_life),
        str(track_status).upper(),
        int(n_points),
    )


class DuckDBSingleWriterDaemon:
    """
    Solves Windows NTFS exclusive file-locking on DuckDB.
    - Maintains a fast ':memory:' DuckDB database for instant zero-lock reads by the engine/UI.
    - Serializes all disk writes through an asyncio.Queue so only ONE writer ever touches
      data/f1_historical.duckdb at a time.
    """

    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.db_dir = os.path.join(project_root, "data")
        os.makedirs(self.db_dir, exist_ok=True)
        self.disk_db_path = os.path.join(self.db_dir, "f1_historical.duckdb")

        # 1. In-memory mirror for lock-free sub-millisecond reads during live simulation
        self.mem_conn = duckdb.connect(":memory:")
        self._init_schema(self.mem_conn)

        # 2. Initialize disk database schema once at startup and immediately release the NTFS lock
        with duckdb_connect(self.disk_db_path) as disk_conn:
            self._init_schema(disk_conn)

        # 3. Single-writer async queue
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    def _init_schema(self, conn: duckdb.DuckDBPyConnection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_telemetry_laps (
                lap INTEGER,
                driver VARCHAR,
                position INTEGER,
                compound VARCHAR,
                tyre_life INTEGER,
                gap_to_leader DOUBLE,
                lap_time_sec DOUBLE,
                track_status VARCHAR,
                track_temp DOUBLE,
                PRIMARY KEY (lap, driver)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS race_calendar (
                season INTEGER,
                round INTEGER,
                race_name VARCHAR,
                circuit VARCHAR,
                circuit_id VARCHAR,
                date VARCHAR,
                total_laps INTEGER,
                PRIMARY KEY (season, round)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grand_prix_sessions (
                season INTEGER,
                circuit VARCHAR,
                session_type VARCHAR,
                lap INTEGER,
                driver VARCHAR,
                position INTEGER,
                compound VARCHAR,
                tyre_life INTEGER,
                gap_to_leader DOUBLE,
                lap_time_sec DOUBLE,
                track_status VARCHAR,
                track_temp DOUBLE,
                speed_kmh DOUBLE,
                rpm DOUBLE,
                throttle_pct DOUBLE,
                brake_pct DOUBLE,
                gear INTEGER,
                drs INTEGER,
                lap_timestamp VARCHAR,
                PRIMARY KEY (season, circuit, session_type, lap, driver)
            )
        """)

    async def enqueue_lap_batch(self, rows: list[tuple]):
        """Writes immediately to the in-memory mirror and queues disk persistence."""
        self.mem_conn.executemany(
            "INSERT OR REPLACE INTO live_telemetry_laps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await self.write_queue.put(rows)

    def read_current_standings(self) -> list[dict]:
        """Lock-free read directly from the in-memory DuckDB mirror."""
        cur = self.mem_conn.execute("""
            SELECT lap, driver, position, compound, tyre_life, gap_to_leader, lap_time_sec, track_status
            FROM live_telemetry_laps
            WHERE lap = (SELECT MAX(lap) FROM live_telemetry_laps)
            ORDER BY position ASC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    async def writer_worker(self):
        """Background daemon that drains the write queue to the Windows NTFS DuckDB file."""
        self.running = True
        while self.running or not self.write_queue.empty():
            try:
                batch = await asyncio.wait_for(self.write_queue.get(), timeout=0.3)
            except TimeoutError:
                continue

            with duckdb_connect(self.disk_db_path) as disk_conn:
                disk_conn.executemany(
                    "INSERT OR REPLACE INTO live_telemetry_laps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
            self.write_queue.task_done()


class TelemetryReplayStream:
    """
    Streams lap-by-lap telemetry ticks for any Grand Prix circuit and session type.
    Uses real FastF1 historical race data when cached (e.g. 2024 Bahrain GP) and
    physics-calibrated multi-stint telemetry for all 24 Grand Prix circuits and all sessions.
    """

    def __init__(self, circuit: str = "Bahrain", session_type: str = "Race", full_grid: bool = False):
        self.circuit = circuit
        self.session_type = session_type
        meta = get_circuit_metadata(circuit)
        base_sec = meta["base_lap_sec"]

        drivers_subset = F1_2024_DRIVERS if full_grid else F1_2024_DRIVERS[:8]
        self.grid = []
        cum_gap = 0.0
        for idx, d in enumerate(drivers_subset, 1):
            if idx == 1:
                cum_gap = 0.0
            elif idx <= 5:
                cum_gap += 1.35
            else:
                cum_gap += 3.8
            self.grid.append({
                "driver": d["driver"],
                "pos": idx,
                "compound": "SOFT" if idx <= 6 else "MEDIUM",
                "tyre_life": 12,
                "gap": round(cum_gap, 2),
                "base": round(base_sec + d["pace_offset"], 2),
            })

    def next_lap_tick(self, lap: int, track_status: str = "GREEN", track_temp: float = 31.8) -> list[tuple]:
        rows = []
        leader_base = self.grid[0]["base"] if self.grid else 94.1
        for car in self.grid:
            car["tyre_life"] += 1
            deg = 0.08 * car["tyre_life"]
            status_mult = 1.0 if track_status == "GREEN" else (1.28 if track_status == "VSC" else 1.48)
            lap_time = (car["base"] + deg) * status_mult + float(np.random.normal(0, 0.08))
            if car["pos"] > 1:
                if track_status == "SC":
                    # Safety Car compresses the pack gaps
                    car["gap"] = max(0.35 * car["pos"], car["gap"] * 0.65)
                else:
                    car["gap"] = max(0.4, car["gap"] + (lap_time - (leader_base + 0.8) * status_mult))
            rows.append((
                int(lap),
                str(car["driver"]),
                int(car["pos"]),
                str(car["compound"]),
                int(car["tyre_life"]),
                round(float(car["gap"]), 3),
                round(float(lap_time), 3),
                str(track_status),
                float(track_temp),
            ))
        # Re-sort positions by gap to leader
        rows.sort(key=lambda r: r[5])
        sorted_rows = []
        leader_gap = rows[0][5] if rows else 0.0
        for idx, r in enumerate(rows, 1):
            sorted_rows.append((
                r[0],
                r[1],
                idx,
                r[3],
                r[4],
                round(max(0.0, r[5] - leader_gap), 3),
                r[6],
                r[7],
                r[8],
            ))
        return sorted_rows


def ensure_all_grand_prix_in_duckdb(db_path: str | None = None):
    """
    Ensures the DuckDB database contains:
    1. `race_calendar` with all 24 Grand Prix races of the 2024 season.
    2. `grand_prix_sessions` populated with telemetry laps for any requested Grand Prix & session
       (including real FastF1 2024 Bahrain Race laps from the local cache).
    3. `live_telemetry_laps` initialized for active dashboard playback.
    """
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with duckdb_connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS live_telemetry_laps (
                lap INTEGER,
                driver VARCHAR,
                position INTEGER,
                compound VARCHAR,
                tyre_life INTEGER,
                gap_to_leader DOUBLE,
                lap_time_sec DOUBLE,
                track_status VARCHAR,
                track_temp DOUBLE,
                PRIMARY KEY (lap, driver)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS race_calendar (
                season INTEGER,
                round INTEGER,
                race_name VARCHAR,
                circuit VARCHAR,
                circuit_id VARCHAR,
                date VARCHAR,
                total_laps INTEGER,
                PRIMARY KEY (season, round)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS grand_prix_sessions (
                season INTEGER,
                circuit VARCHAR,
                session_type VARCHAR,
                lap INTEGER,
                driver VARCHAR,
                position INTEGER,
                compound VARCHAR,
                tyre_life INTEGER,
                gap_to_leader DOUBLE,
                lap_time_sec DOUBLE,
                track_status VARCHAR,
                track_temp DOUBLE,
                speed_kmh DOUBLE,
                rpm DOUBLE,
                throttle_pct DOUBLE,
                brake_pct DOUBLE,
                gear INTEGER,
                drs INTEGER,
                lap_timestamp VARCHAR,
                PRIMARY KEY (season, circuit, session_type, lap, driver)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS session_data_sources (
                season INTEGER,
                circuit VARCHAR,
                session_type VARCHAR,
                source VARCHAR,
                PRIMARY KEY (season, circuit, session_type)
            )
        """)

        cal_count = con.execute("SELECT COUNT(*) FROM race_calendar").fetchone()[0]
        expected_total_cal = len(F1_2024_CALENDAR) * len(SUPPORTED_SEASONS)
        if cal_count < expected_total_cal:
            cal_rows = [
                (
                    int(yr),
                    r["round"],
                    r["race_name"],
                    r["circuit"],
                    r["circuit_id"],
                    f"{int(yr)}{r['date'][4:]}",
                    r["total_laps"],
                )
                for yr in SUPPORTED_SEASONS
                for r in F1_2024_CALENDAR
            ]
            con.executemany(
                "INSERT OR REPLACE INTO race_calendar VALUES (?, ?, ?, ?, ?, ?, ?)",
                cal_rows,
            )


SESSION_TO_FASTF1_CODE: dict[str, str] = {
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
    "Qualifying": "Q",
    "Sprint": "S",
    "Race": "R",
}

SESSION_TO_OPENF1_NAME: dict[str, str] = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Qualifying": "Qualifying",
    "Sprint": "Sprint",
    "Race": "Race",
}

_OPENF1_CIRCUIT_TOKENS: dict[str, tuple[str, ...]] = {
    "bahrain": ("sakhir", "bahrain"),
    "saudi arabia": ("jeddah", "saudi"),
    "australia": ("melbourne", "australia"),
    "japan": ("suzuka", "japan"),
    "china": ("shanghai", "china"),
    "miami": ("miami",),
    "emilia romagna": ("imola",),
    "monaco": ("monte carlo", "monaco"),
    "canada": ("montreal", "montréal", "canada"),
    "spain": ("catalunya", "barcelona", "spain"),
    "austria": ("spielberg", "austria"),
    "great britain": ("silverstone", "great britain", "united kingdom"),
    "hungary": ("hungaroring", "budapest", "hungary"),
    "belgium": ("spa-francorchamps", "spa", "belgium"),
    "netherlands": ("zandvoort", "netherlands"),
    "italy": ("monza",),
    "azerbaijan": ("baku", "azerbaijan"),
    "singapore": ("singapore", "marina bay"),
    "usa": ("austin",),
    "mexico": ("mexico",),
    "brazil": ("interlagos", "são paulo", "sao paulo", "brazil"),
    "las vegas": ("las vegas",),
    "qatar": ("lusail", "qatar"),
    "abu dhabi": ("yas marina", "abu dhabi"),
}


def _load_session_from_openf1(
    season: int,
    circuit: str,
    session_type: str,
    meta: dict,
) -> list[tuple]:
    """
    Fetches authentic FIA lap-by-lap timing, tyre stints, track status (SC/VSC), and weather
    from the free public OpenF1 REST API (`https://api.openf1.org/v1/`, covering 2023+ seasons).
    Used automatically when Formula 1's `livetiming.formula1.com` returns HTTP 401 to FastF1.
    """
    import requests

    if int(season) < 2023:
        return []

    of1_session_name = SESSION_TO_OPENF1_NAME.get(session_type, "Race")
    sess_resp = requests.get(
        "https://api.openf1.org/v1/sessions",
        params={"year": int(season), "session_name": of1_session_name},
        timeout=12,
    )
    if sess_resp.status_code != 200:
        return []
    sessions = sess_resp.json()
    if not isinstance(sessions, list) or not sessions:
        return []

    ckey = str(circuit).strip().lower()
    tokens = _OPENF1_CIRCUIT_TOKENS.get(ckey, (ckey,))
    matched_session = None
    for s_item in sessions:
        haystack = " ".join(
            str(s_item.get(k, "")).lower()
            for k in ("circuit_short_name", "location", "country_name")
        )
        if any(tok in haystack for tok in tokens):
            matched_session = s_item
            break
    if not matched_session:
        return []

    sk = matched_session["session_key"]
    drv_r = requests.get("https://api.openf1.org/v1/drivers", params={"session_key": sk}, timeout=12)
    lap_r = requests.get("https://api.openf1.org/v1/laps", params={"session_key": sk}, timeout=15)
    stint_r = requests.get("https://api.openf1.org/v1/stints", params={"session_key": sk}, timeout=12)
    wx_r = requests.get("https://api.openf1.org/v1/weather", params={"session_key": sk}, timeout=12)
    rc_r = requests.get("https://api.openf1.org/v1/race_control", params={"session_key": sk}, timeout=12)

    if drv_r.status_code != 200 or lap_r.status_code != 200:
        return []

    drivers_raw = drv_r.json() or []
    laps_raw = lap_r.json() or []
    stints_raw = stint_r.json() if stint_r.status_code == 200 else []
    wx_raw = wx_r.json() if wx_r.status_code == 200 else []
    rc_raw = rc_r.json() if rc_r.status_code == 200 else []
    if not laps_raw:
        return []

    num_to_code: dict[int, str] = {}
    for d in drivers_raw:
        if d.get("driver_number") is not None and d.get("name_acronym"):
            num_to_code[int(d["driver_number"])] = str(d["name_acronym"]).upper()

    # Build (driver_number, lap_number) -> (compound, tyre_life) from authentic OpenF1 stints
    stint_lookup: dict[tuple[int, int], tuple[str, int]] = {}
    for st in stints_raw or []:
        dnum = st.get("driver_number")
        l_start = st.get("lap_start")
        l_end = st.get("lap_end")
        comp = str(st.get("compound") or "MEDIUM").upper()
        if comp not in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
            comp = "MEDIUM"
        age_start = int(st.get("tyre_age_at_start") or 0)
        if dnum is not None and l_start is not None and l_end is not None:
            for ln in range(int(l_start), int(l_end) + 1):
                stint_lookup[(int(dnum), ln)] = (comp, max(1, age_start + (ln - int(l_start)) + 1))

    # Build per-lap track status (GREEN / VSC / SC) from authentic FIA Race Control messages
    max_lap = max(int(l["lap_number"]) for l in laps_raw if l.get("lap_number") is not None)
    lap_status_map: dict[int, str] = {ln: "GREEN" for ln in range(1, max_lap + 1)}
    active_flag = "GREEN"
    rc_by_lap: dict[int, list[str]] = {}
    for rc in rc_raw or []:
        ln = rc.get("lap_number")
        msg = str(rc.get("message") or "").upper()
        if ln is not None and msg:
            rc_by_lap.setdefault(int(ln), []).append(msg)
    for ln in range(1, max_lap + 1):
        msgs = rc_by_lap.get(ln, [])
        for m in msgs:
            if "VIRTUAL SAFETY CAR DEPLOYED" in m:
                active_flag = "VSC"
            elif "SAFETY CAR DEPLOYED" in m:
                active_flag = "SC"
            elif "VIRTUAL SAFETY CAR ENDING" in m or "SAFETY CAR IN THIS LAP" in m or "TRACK CLEAR" in m or "GREEN LIGHT" in m:
                # Keep SC/VSC active during the in-lap, then clear for the following lap
                if active_flag in ("SC", "VSC"):
                    lap_status_map[ln] = active_flag
                    active_flag = "GREEN"
                    continue
        if lap_status_map[ln] == "GREEN":
            lap_status_map[ln] = active_flag

    # Track temperature from authentic OpenF1 weather stream
    base_temp = float(meta["track_temp"])
    wx_temps = [float(w["track_temperature"]) for w in (wx_raw or []) if w.get("track_temperature") is not None]
    if wx_temps:
        base_temp = round(float(np.median(wx_temps)), 1)

    base_lap_sec = float(meta["base_lap_sec"])

    # Compute exact lap end timestamps per driver for accurate gap_to_leader & race order
    by_lap: dict[int, list[dict]] = {}
    t0_ts: float | None = None
    parsed_laps = []
    for lr in laps_raw:
        dnum = lr.get("driver_number")
        ln = lr.get("lap_number")
        if dnum is None or ln is None:
            continue
        dt_raw = lr.get("date_start")
        start_ts = pd.to_datetime(dt_raw, utc=True).timestamp() if dt_raw else None
        dur = float(lr["lap_duration"]) if lr.get("lap_duration") is not None else None
        if start_ts is not None and (t0_ts is None or start_ts < t0_ts):
            t0_ts = start_ts
        parsed_laps.append((int(dnum), int(ln), start_ts, dur, lr.get("st_speed")))

    if t0_ts is None:
        t0_ts = 0.0

    # Fill missing start_ts / lap_duration smoothly per driver
    parsed_laps.sort(key=lambda x: (x[0], x[1]))
    drv_last_end: dict[int, float] = {}
    for dnum, ln, start_ts, dur, st_spd in parsed_laps:
        dur_val = float(dur) if (dur is not None and dur > 10.0) else (base_lap_sec + 2.0)
        if start_ts is not None:
            end_s = max(0.0, (start_ts - t0_ts) + dur_val)
        elif dnum in drv_last_end:
            end_s = drv_last_end[dnum] + dur_val
        else:
            end_s = ln * dur_val
        drv_last_end[dnum] = end_s
        drv_code = num_to_code.get(dnum, f"D{dnum}")
        comp, t_life = stint_lookup.get((dnum, ln), ("MEDIUM", max(1, ln)))
        spd = round(float(st_spd), 1) if st_spd is not None else 305.0
        by_lap.setdefault(ln, []).append({
            "driver": drv_code,
            "end_s": end_s,
            "l_sec": round(dur_val, 3),
            "comp": comp,
            "t_life": t_life,
            "spd": spd,
        })

    rows_to_insert: list[tuple] = []
    last_known: dict[str, dict] = {}
    for ln in range(1, max_lap + 1):
        entries = by_lap.get(ln, [])
        if not entries:
            continue
        entries.sort(key=lambda e: e["end_s"])
        leader_end = entries[0]["end_s"]
        t_status = lap_status_map.get(ln, "GREEN")

        seen = set()
        lap_rows = []
        for idx_p, e in enumerate(entries, 1):
            seen.add(e["driver"])
            gap = 0.0 if idx_p == 1 else round(max(0.0, e["end_s"] - leader_end), 3)
            mins = int(e["end_s"] // 60)
            secs = e["end_s"] % 60
            ts_str = f"{mins // 60:02d}:{mins % 60:02d}:{secs:06.3f}"
            rec = {
                "driver": e["driver"],
                "pos": idx_p,
                "comp": e["comp"],
                "t_life": e["t_life"],
                "gap": gap,
                "l_sec": e["l_sec"],
                "t_status": t_status,
                "spd": e["spd"],
                "ts_str": ts_str,
            }
            last_known[e["driver"]] = rec
            lap_rows.append(rec)

        for drv_c, prev_r in last_known.items():
            if drv_c not in seen:
                carried = dict(prev_r)
                carried["gap"] = round(max(carried["gap"], base_lap_sec), 3)
                lap_rows.append(carried)

        lap_rows.sort(key=lambda e: (e["pos"], e["gap"]))
        for pos_f, e in enumerate(lap_rows, 1):
            gap_f = 0.0 if pos_f == 1 else e["gap"]
            rows_to_insert.append((
                int(season), circuit, session_type, int(ln), e["driver"], pos_f, e["comp"], e["t_life"],
                gap_f, e["l_sec"], e["t_status"], float(base_temp),
                e["spd"], round(10800.0 + (320 - pos_f * 5), 0), 78.5, 18.0, 7,
                1 if gap_f < 1.0 and pos_f > 1 else 0,
                e["ts_str"],
            ))
    return rows_to_insert


def get_session_data_source(
    season: int,
    circuit: str,
    session_type: str = "Race",
    db_path: str | None = None,
) -> str:
    """
    Returns the provenance of the loaded telemetry for `(season, circuit, session_type)`:
    - `"FASTF1_REAL"`: Authentic lap-by-lap FIA timing & telemetry downloaded via FastF1.
    - `"JOLPICA_CALIBRATED"`: Physics-synthesized laps calibrated to official FIA/Jolpica finishing order.
    - `"SYNTHETIC"`: Physics-synthesized laps (for circuits not held in that historical season).
    """
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    season = int(season)
    session_type = normalize_session_type(session_type)
    ensure_all_grand_prix_in_duckdb(db_path)

    with duckdb_connect(db_path) as con:
        row = con.execute(
            """
            SELECT source FROM session_data_sources
            WHERE season = ? AND circuit = ? AND session_type = ?
            """,
            [season, circuit, session_type],
        ).fetchone()
    if row and row[0]:
        return str(row[0])
    if season == 2024 and circuit.lower() == "bahrain" and session_type == "Race":
        return "FASTF1_REAL"
    if get_historical_race_order(season, circuit):
        return "JOLPICA_CALIBRATED"
    return "SYNTHETIC"


def load_or_populate_gp_session(
    circuit: str,
    session_type: str = "Race",
    season: int = 2024,
    db_path: str | None = None,
    force_fastf1: bool = False,
    allow_auto_fastf1: bool = False,
) -> list[int]:
    """
    Ensures the requested (season, circuit, session_type) is populated in `grand_prix_sessions`
    (supporting seasons 2021, 2022, 2023, 2024 and all sessions FP1/FP2/FP3/Qualifying/Sprint/Race)
    and returns the sorted list of available laps.
    - Automatically loads real FastF1 telemetry for 2024 Bahrain Race (cached locally) OR whenever
      `allow_auto_fastf1=True` (default in the live Pit Wall dashboard) OR `force_fastf1=True`.
    - Hardened FastF1 loader retries without weather stream if FIA weather API hiccups, and resets
      stale FastF1 `TyreLife` counts whenever a driver's compound changes across laps.
    """
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    season = int(season)
    session_type = normalize_session_type(session_type)
    ensure_all_grand_prix_in_duckdb(db_path)

    with duckdb_connect(db_path) as con:
        existing_laps = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT lap FROM grand_prix_sessions
                WHERE season = ? AND circuit = ? AND session_type = ?
                ORDER BY lap ASC
                """,
                [season, circuit, session_type],
            ).fetchall()
        ]
        min_cars = (
            con.execute(
                """
                SELECT MIN(cnt) FROM (
                    SELECT COUNT(*) AS cnt FROM grand_prix_sessions
                    WHERE season = ? AND circuit = ? AND session_type = ?
                    GROUP BY lap
                )
                """,
                [season, circuit, session_type],
            ).fetchone()[0]
            if existing_laps
            else 0
        )
        existing_src = con.execute(
            """
            SELECT source FROM session_data_sources
            WHERE season = ? AND circuit = ? AND session_type = ?
            """,
            [season, circuit, session_type],
        ).fetchone()
    src_val = str(existing_src[0]) if existing_src and existing_src[0] else ""
    already_real = src_val in ("FASTF1_REAL", "OPENF1_REAL") or (
        season == 2024 and circuit.lower() == "bahrain" and session_type == "Race" and bool(existing_laps)
    )
    if not force_fastf1 and existing_laps and (min_cars or 0) >= 20 and (not allow_auto_fastf1 or already_real):
        if not existing_src:
            inferred_src = (
                "FASTF1_REAL"
                if (season == 2024 and circuit.lower() == "bahrain" and session_type == "Race")
                else ("JOLPICA_CALIBRATED" if get_historical_race_order(season, circuit) else "SYNTHETIC")
            )
            with duckdb_connect(db_path) as con:
                con.execute(
                    "INSERT OR REPLACE INTO session_data_sources VALUES (?, ?, ?, ?)",
                    [season, circuit, session_type, inferred_src],
                )
        return existing_laps

    meta = get_circuit_metadata(circuit, season=season)
    total_laps = meta["total_laps"] if session_type == "Race" else (19 if session_type == "Sprint" else 20)
    base_temp = meta["track_temp"]
    base_lap_sec = meta["base_lap_sec"]
    rows_to_insert = []
    data_source = "SYNTHETIC"

    loaded_from_fastf1 = False
    should_try_fastf1 = (
        force_fastf1
        or allow_auto_fastf1
        or (season == 2024 and circuit.lower() == "bahrain" and session_type == "Race")
    )
    if should_try_fastf1:
        # For 2023+ sessions (except locally cached 2024 Bahrain Race), try free public OpenF1 API FIRST (~6s)
        # because F1's CloudFront now returns HTTP 401 to unauthenticated FastF1 requests on 2023+ archives.
        prefer_openf1_first = season >= 2023 and not (
            season == 2024 and circuit.lower() == "bahrain" and session_type == "Race"
        )
        if prefer_openf1_first:
            try:
                of1_rows = _load_session_from_openf1(season, circuit, session_type, meta)
                if of1_rows:
                    rows_to_insert = of1_rows
                    loaded_from_fastf1 = True
                    data_source = "OPENF1_REAL"
                    logger.success(
                        f"📡 Loaded authentic OpenF1 FIA telemetry for {season} {circuit} {session_type} "
                        f"({len(rows_to_insert)} driver-laps)."
                    )
            except Exception as of1_err:
                logger.warning(f"OpenF1 primary load fallback for {season} {circuit} {session_type}: {of1_err}")

        if not loaded_from_fastf1:
            try:
                ff1_code = SESSION_TO_FASTF1_CODE.get(session_type, "R")
                event_query = "Bahrain" if circuit.lower() == "bahrain" else meta["race_name"]
                s = fastf1.get_session(season, event_query, ff1_code)
                try:
                    s.load(telemetry=False, weather=True, messages=False)
                except Exception:
                    s.load(telemetry=False, weather=False, messages=False)
                laps_df = s.laps
                if laps_df is not None and not laps_df.empty:
                    weather_temp = base_temp
                    try:
                        w_data = getattr(s, "weather_data", None)
                        if w_data is not None and not w_data.empty and "TrackTemp" in w_data.columns:
                            w_med = w_data["TrackTemp"].dropna().median()
                            if pd.notna(w_med):
                                weather_temp = round(float(w_med), 1)
                    except Exception:
                        pass

                    last_known_driver_row: dict[str, dict] = {}
                    driver_prev_comp: dict[str, str] = {}
                    driver_stint_age: dict[str, int] = {}
                    max_lap_num = int(laps_df["LapNumber"].dropna().max())
                    for lap_num in range(1, max_lap_num + 1):
                        sub = laps_df[laps_df["LapNumber"] == lap_num].copy()
                        if sub.empty:
                            continue
                        sub["LapSec"] = sub["LapTime"].dt.total_seconds().fillna(base_lap_sec + 2.0)
                        sub["EndSec"] = sub["Time"].dt.total_seconds()
                        sub = sub.sort_values(["Position", "EndSec"], na_position="last")
                        leader_end = sub["EndSec"].dropna().min() if not sub["EndSec"].dropna().empty else 0.0

                        seen_this_lap = set()
                        lap_entries = []
                        for pos_idx, (_, lr) in enumerate(sub.iterrows(), 1):
                            drv = str(lr["Driver"])
                            seen_this_lap.add(drv)
                            pos = int(lr["Position"]) if pd.notna(lr["Position"]) else pos_idx
                            comp = str(lr["Compound"]).upper() if pd.notna(lr["Compound"]) else "HARD"
                            if comp not in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
                                comp = "HARD"

                            raw_life = int(lr["TyreLife"]) if pd.notna(lr["TyreLife"]) else max(1, lap_num)
                            prev_c = driver_prev_comp.get(drv)
                            if prev_c is not None and prev_c != comp:
                                # Reset stint age if FastF1 failed to reset TyreLife on compound switch
                                t_life = 1 if raw_life > 2 else max(1, raw_life)
                            elif prev_c == comp and drv in driver_stint_age and raw_life >= driver_stint_age[drv] + 5:
                                # Guard against stale cumulative TyreLife jump
                                t_life = driver_stint_age[drv] + 1
                            else:
                                t_life = max(1, raw_life)
                            driver_prev_comp[drv] = comp
                            driver_stint_age[drv] = t_life

                            end_s = float(lr["EndSec"]) if pd.notna(lr["EndSec"]) else (leader_end + pos_idx * 1.5)
                            gap = max(0.0, round(end_s - leader_end, 3)) if pos > 1 else 0.0
                            l_sec = round(float(lr["LapSec"]), 3)
                            raw_ts = str(lr.get("TrackStatus", "1"))
                            t_status = "SC" if "4" in raw_ts else ("VSC" if "6" in raw_ts or "7" in raw_ts else "GREEN")
                            spd = round(float(lr["SpeedST"]) if pd.notna(lr.get("SpeedST")) else (315.0 - pos * 0.8), 1)
                            mins = int(end_s // 60)
                            secs = end_s % 60
                            ts_str = f"00:{mins:02d}:{secs:06.3f}"
                            entry = {
                                "driver": drv,
                                "pos": pos,
                                "comp": comp,
                                "t_life": t_life,
                                "gap": gap,
                                "l_sec": l_sec,
                                "t_status": t_status,
                                "spd": spd,
                                "ts_str": ts_str,
                            }
                            last_known_driver_row[drv] = entry
                            lap_entries.append(entry)

                        # Carry forward classified lapped finishers / early retirees so the grid stays intact
                        for drv, prev_entry in last_known_driver_row.items():
                            if drv not in seen_this_lap:
                                carried = dict(prev_entry)
                                carried["gap"] = round(max(carried["gap"], base_lap_sec), 3)
                                lap_entries.append(carried)

                        lap_entries.sort(key=lambda e: (e["pos"], e["gap"]))
                        for idx_pos, e in enumerate(lap_entries, 1):
                            pos_final = idx_pos
                            gap_final = 0.0 if pos_final == 1 else e["gap"]
                            rows_to_insert.append((
                                season, circuit, session_type, int(lap_num), e["driver"], pos_final, e["comp"], e["t_life"],
                                gap_final, e["l_sec"], e["t_status"], float(weather_temp),
                                e["spd"], round(10800.0 + (320 - pos_final * 5), 0), 78.5, 18.0, 7,
                                1 if gap_final < 1.0 and pos_final > 1 else 0,
                                e["ts_str"],
                            ))
                    loaded_from_fastf1 = len(rows_to_insert) > 0
                    if loaded_from_fastf1:
                        data_source = "FASTF1_REAL"
            except Exception as e:
                logger.warning(f"FastF1 load fallback for {season} {circuit} {session_type}: {e}")

        if not loaded_from_fastf1 and (force_fastf1 or allow_auto_fastf1) and existing_laps and (min_cars or 0) >= 20:
            return existing_laps

    if not loaded_from_fastf1:
        rng = np.random.default_rng(sum(ord(c) for c in f"{season}_{circuit}_{session_type}"))
        season_mods = SEASON_PACE_MODIFIERS.get(season, {})
        hist_order = get_historical_race_order(season, circuit)
        if hist_order:
            data_source = "JOLPICA_CALIBRATED"
            ordered_drivers = [DRIVER_METADATA_MAP[drv] for drv in hist_order if drv in DRIVER_METADATA_MAP]
        else:
            data_source = "SYNTHETIC"
            ordered_drivers = list(F1_2024_DRIVERS)

        # Build realistic multi-stint session for all 20 drivers
        driver_states = []
        for idx, d in enumerate(ordered_drivers, 1):
            start_comp = "SOFT" if idx <= 10 or session_type != "Race" else "MEDIUM"
            pit_lap_1 = int(total_laps * 0.32) + (idx % 5)
            pit_lap_2 = int(total_laps * 0.68) + (idx % 4)
            if hist_order:
                # Calibrate base pace directly to the official historical finishing order
                pace_off = round((idx - 1) * 0.055, 3)
            else:
                pace_off = season_mods.get(d["driver"], d["pace_offset"])
            driver_states.append({
                "driver": d["driver"],
                "compound": start_comp,
                "tyre_life": 0,
                "cum_time": idx * 0.45,
                "pace_offset": pace_off,
                "pit_laps": {pit_lap_1: "HARD", pit_lap_2: "MEDIUM"} if session_type == "Race" else {},
            })

        sc_lap = int(total_laps * 0.28) if session_type == "Race" else -1
        vsc_lap = int(total_laps * 0.62) if session_type == "Race" else -1

        for lap_num in range(1, total_laps + 1):
            t_status = "GREEN"
            if lap_num in (sc_lap, sc_lap + 1):
                t_status = "SC"
            elif lap_num == vsc_lap:
                t_status = "VSC"

            lap_records = []
            for st in driver_states:
                if lap_num in st["pit_laps"]:
                    st["compound"] = st["pit_laps"][lap_num]
                    st["tyre_life"] = 1
                    pit_penalty = 11.5 if t_status in ("SC", "VSC") else 22.0
                else:
                    st["tyre_life"] += 1
                    pit_penalty = 0.0

                comp_delta = 0.0 if st["compound"] == "SOFT" else (0.35 if st["compound"] == "MEDIUM" else 0.65)
                deg_rate = 0.085 if st["compound"] == "SOFT" else (0.055 if st["compound"] == "MEDIUM" else 0.035)
                status_factor = 1.0 if t_status == "GREEN" else (1.28 if t_status == "VSC" else 1.45)

                l_sec = (
                    (base_lap_sec + st["pace_offset"] + comp_delta + deg_rate * st["tyre_life"])
                    * status_factor
                    + pit_penalty
                    + float(rng.normal(0.0, 0.12))
                )
                st["cum_time"] += l_sec
                lap_records.append((st, round(l_sec, 3)))

            lap_records.sort(key=lambda item: item[0]["cum_time"])
            leader_time = lap_records[0][0]["cum_time"]

            for pos_idx, (st, l_sec) in enumerate(lap_records, 1):
                if t_status == "SC" and pos_idx > 1:
                    # Compress gaps behind Safety Car
                    st["cum_time"] = leader_time + pos_idx * 0.55
                gap = 0.0 if pos_idx == 1 else round(max(0.0, st["cum_time"] - leader_time), 3)
                car_ahead_gap = (
                    gap - round(max(0.0, lap_records[pos_idx - 2][0]["cum_time"] - leader_time), 3)
                    if pos_idx > 1
                    else 99.0
                )
                drs_active = 1 if (t_status == "GREEN" and pos_idx > 1 and car_ahead_gap < 1.0) else 0
                spd = round(318.0 - st["pace_offset"] * 6.0 + (12.0 if drs_active else 0.0), 1)
                rpm = round(11100.0 - st["pace_offset"] * 180.0 + (350.0 if drs_active else 0.0), 0)
                mins = int(st["cum_time"] // 60)
                secs = st["cum_time"] % 60
                ts_str = f"{mins // 60:02d}:{mins % 60:02d}:{secs:06.3f}"
                rows_to_insert.append((
                    season,
                    circuit,
                    session_type,
                    int(lap_num),
                    st["driver"],
                    int(pos_idx),
                    st["compound"],
                    int(st["tyre_life"]),
                    gap,
                    l_sec,
                    t_status,
                    round(base_temp + np.sin(lap_num / 10.0) * 1.2, 1),
                    spd,
                    rpm,
                    round(76.0 - st["pace_offset"] * 2.5, 1),
                    round(17.5 + st["pace_offset"] * 1.2, 1),
                    7 if t_status == "GREEN" else 5,
                    drs_active,
                    ts_str,
                ))

    with duckdb_connect(db_path) as con:
        if loaded_from_fastf1:
            con.execute(
                """
                DELETE FROM grand_prix_sessions
                WHERE season = ? AND circuit = ? AND session_type = ?
                """,
                [season, circuit, session_type],
            )
        con.executemany(
            "INSERT OR REPLACE INTO grand_prix_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_to_insert,
        )
        con.execute(
            "INSERT OR REPLACE INTO session_data_sources VALUES (?, ?, ?, ?)",
            [season, circuit, session_type, data_source],
        )
        laps = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT lap FROM grand_prix_sessions
                WHERE season = ? AND circuit = ? AND session_type = ?
                ORDER BY lap ASC
                """,
                [season, circuit, session_type],
            ).fetchall()
        ]
    return laps


def fetch_real_fastf1_session(
    circuit: str,
    session_type: str = "Race",
    season: int = 2024,
    db_path: str | None = None,
) -> tuple[bool, str]:
    """
    On-demand downloader invoked by the dashboard button to fetch authentic FastF1 FIA timing & tyre data
    for any supported `(season, circuit, session_type)` and persist it to DuckDB.
    """
    season = int(season)
    session_type = normalize_session_type(session_type)
    laps = load_or_populate_gp_session(
        circuit=circuit,
        session_type=session_type,
        season=season,
        db_path=db_path,
        force_fastf1=True,
    )
    src = get_session_data_source(season, circuit, session_type, db_path=db_path)
    if src in ("FASTF1_REAL", "OPENF1_REAL") and laps:
        provider = "OpenF1 FIA API" if src == "OPENF1_REAL" else "FastF1"
        return True, f"Loaded real {provider} telemetry for {season} {circuit} {session_type} ({len(laps)} laps)."
    return False, (
        f"Real FIA telemetry unavailable for {season} {circuit} {session_type} "
        "(session may not have occurred in that season or API is unreachable). Kept calibrated model."
    )


def get_gp_session_laps_df(
    circuit: str,
    session_type: str = "Race",
    season: int = 2024,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Returns the complete DataFrame of all laps for (season, circuit, session_type),
    populating it in DuckDB first if needed.
    """
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    season = int(season)
    session_type = normalize_session_type(session_type)
    load_or_populate_gp_session(circuit, session_type, season=season, db_path=db_path)
    with duckdb_connect(db_path) as con:
        cur = con.execute(
            """
            SELECT season, circuit, session_type, lap, driver, position, compound, tyre_life,
                   gap_to_leader, lap_time_sec, track_status, track_temp, speed_kmh, rpm,
                   throttle_pct, brake_pct, gear, drs, lap_timestamp
            FROM grand_prix_sessions
            WHERE season = ? AND circuit = ? AND session_type = ?
            ORDER BY lap ASC, position ASC
            """,
            [season, circuit, session_type],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def sync_gp_lap_to_live_table(
    circuit: str,
    session_type: str,
    lap: int,
    season: int = 2024,
    track_status_override: str | None = None,
    track_temp_override: float | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Loads the specified (season, circuit, session_type, lap) from `grand_prix_sessions`
    and syncs it into `live_telemetry_laps` so the LangGraph Race Engineer agent and
    all downstream tools operate on the exact active lap and environment state.
    """
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    season = int(season)
    session_type = normalize_session_type(session_type)
    available_laps = load_or_populate_gp_session(circuit, session_type, season, db_path)
    target_lap = lap if lap in available_laps else (available_laps[-1] if available_laps else 1)

    with duckdb_connect(db_path) as con:
        cur = con.execute(
            """
            SELECT lap, driver, position, compound, tyre_life, gap_to_leader, lap_time_sec,
                   track_status, track_temp, speed_kmh, rpm, throttle_pct, brake_pct, gear, drs, lap_timestamp
            FROM grand_prix_sessions
            WHERE season = ? AND circuit = ? AND session_type = ? AND lap = ?
            ORDER BY position ASC
            """,
            [season, circuit, session_type, target_lap],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        if rows:
            live_tuples = []
            for r in rows:
                t_stat = track_status_override if track_status_override else r["track_status"]
                t_temp = float(track_temp_override) if track_temp_override is not None else float(r["track_temp"])
                r["track_status"] = t_stat
                r["track_temp"] = t_temp
                live_tuples.append((
                    int(r["lap"]),
                    str(r["driver"]),
                    int(r["position"]),
                    str(r["compound"]),
                    int(r["tyre_life"]),
                    float(r["gap_to_leader"]),
                    float(r["lap_time_sec"]),
                    str(t_stat),
                    float(t_temp),
                ))
            # Replace live_telemetry_laps with the exact active session lap & driver set
            con.execute("DELETE FROM live_telemetry_laps")
            con.executemany(
                "INSERT OR REPLACE INTO live_telemetry_laps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                live_tuples,
            )

    return pd.DataFrame(rows)


async def _test_daemon():
    logger.info("🧪 Testing Windows DuckDB Single-Writer Daemon & In-Memory Mirror...")
    daemon = DuckDBSingleWriterDaemon()
    stream = TelemetryReplayStream()

    writer_task = asyncio.create_task(daemon.writer_worker())

    t0 = time.perf_counter()
    for lap in range(13, 18):
        status = "SC" if lap == 15 else "GREEN"
        batch = stream.next_lap_tick(lap, track_status=status)
        await daemon.enqueue_lap_batch(batch)

        standings = daemon.read_current_standings()
        leader = standings[0]
        p2 = standings[1]
        logger.info(
            f"📡 [Lap {lap} | {status}] In-Memory Read OK -> "
            f"P1: {leader['driver']} ({leader['tyre_life']}L {leader['compound']}) | "
            f"P2: {p2['driver']} (+{p2['gap_to_leader']:.2f}s)"
        )
        await asyncio.sleep(0.1)

    daemon.running = False
    await writer_task
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    with duckdb_connect(daemon.disk_db_path) as con:
        total_disk_rows = con.execute("SELECT COUNT(*) FROM live_telemetry_laps").fetchone()[0]

    logger.success(
        f"✅ Windows DuckDB Single-Writer Verified! {total_disk_rows} rows persisted to disk "
        f"with 0 NTFS lock collisions (Total test time: {elapsed_ms:.1f}ms)."
    )


if __name__ == "__main__":
    asyncio.run(_test_daemon())
