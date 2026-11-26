from dataclasses import dataclass
import os
import sys
import threading

import duckdb
import pandas as pd
from langchain_core.tools import tool
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion.telemetry_daemon import duckdb_connect, get_circuit_metadata
from src.ml_surrogates.pit_simulator import (
    COMPOUND_PACE_OFFSET_SEC,
    TYRE_WARMUP_PENALTIES_SEC,
    PitStopSimulator,
)
from src.ml_surrogates.strategy_optimizer import RaceStrategyOptimizer
from src.ml_surrogates.tensor_cache import TyreL1TensorCache
from src.rag_engine.multi_hop import MultiHopFIARulebook


@dataclass(frozen=True)
class RaceContext:
    """
    Immutable per-session / per-request execution context so concurrent Streamlit sessions
    never mutate or clobber shared `@st.cache_resource` singletons.
    """
    driver: str
    circuit: str = "Bahrain"
    total_race_laps: int = 57
    team_pit_box_position: int = 1
    weather_condition: str = "Dry (Clear)"
    live_data: dict | None = None
    used_dry_compounds: tuple[str, ...] = ()
    teammate_gap_ahead_sec: float | None = None
    teammate_pitting: bool = False


def get_driver_used_compounds_up_to_lap(
    season: int,
    circuit: str,
    session_type: str,
    driver: str,
    up_to_lap: int,
    db_path: str | None = None,
) -> tuple[str, ...]:
    """Queries DuckDB `grand_prix_sessions` for distinct dry compounds used by `driver` through `up_to_lap`."""
    if db_path is None:
        db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
    if not os.path.exists(db_path):
        return ()
    try:
        with duckdb_connect(db_path) as con:
            rows = con.execute(
                """
                SELECT DISTINCT UPPER(compound)
                FROM grand_prix_sessions
                WHERE season = ? AND circuit = ? AND session_type = ?
                  AND driver = ? AND lap <= ?
                """,
                [int(season), str(circuit), str(session_type), str(driver), int(up_to_lap)],
            ).fetchall()
        dry = tuple(sorted({str(r[0]) for r in rows if r and str(r[0]) in ("SOFT", "MEDIUM", "HARD")}))
        return dry
    except Exception:
        return ()


def build_live_data_from_standings_df(
    df_standings: pd.DataFrame,
    circuit: str,
    total_race_laps: int,
    weather_condition: str = "Dry (Clear)",
    used_dry_compounds: tuple[str, ...] = (),
    teammate_gap_ahead_sec: float | None = None,
    teammate_pitting: bool = False,
) -> dict:
    """Converts a session-local standings DataFrame into the canonical `live_data` dict."""
    if df_standings is None or df_standings.empty:
        return {"error": "No standings available."}

    rows = []
    for r in df_standings.sort_values("position", ascending=True).itertuples(index=False):
        rows.append({
            "lap": int(r.lap),
            "driver": str(r.driver),
            "position": int(r.position),
            "compound": str(r.compound),
            "tyre_life": int(r.tyre_life),
            "gap_to_leader": float(r.gap_to_leader),
            "lap_time_sec": float(r.lap_time_sec),
            "track_status": str(getattr(r, "track_status", "GREEN")),
            "track_temp": float(getattr(r, "track_temp", 32.0)),
        })
    return {
        "lap": rows[0]["lap"],
        "total_race_laps": int(total_race_laps),
        "circuit": str(circuit),
        "track_status": rows[0]["track_status"],
        "track_temp_c": rows[0]["track_temp"],
        "weather_condition": str(weather_condition),
        "used_dry_compounds": tuple(used_dry_compounds),
        "teammate_gap_ahead_sec": teammate_gap_ahead_sec,
        "teammate_pitting": bool(teammate_pitting),
        "standings": rows,
    }


class RaceEngineerToolkit:
    """
    Singleton hardware-accelerated backend holding the read-only L1 Tyre Tensor Cache,
    cached per-(circuit, pit_box) Monte Carlo Pit Simulators, DuckDB Reader, and Multi-Hop FIA Rulebook.
    """

    _instance = None

    def __init__(self, circuit: str = "Bahrain", total_race_laps: int = 57):
        logger.info("🧰 Initializing Race Engineer Toolbelt Subsystems...")
        self.circuit = circuit
        self.total_race_laps = total_race_laps
        self.db_path = os.path.join(PROJECT_ROOT, "data", "f1_historical.duckdb")
        self.tyre_cache = TyreL1TensorCache(total_race_laps=total_race_laps)
        self.strategy_optimizer = RaceStrategyOptimizer(self.tyre_cache)
        self._pit_sim_lock = threading.RLock()
        self._pit_sim_pool: dict[tuple[str, int], PitStopSimulator] = {}
        self.pit_sim = self.get_pit_sim(circuit=circuit, team_pit_box_position=1)
        self._multihop_rag: MultiHopFIARulebook | None = None

    @property
    def multihop_rag(self) -> MultiHopFIARulebook:
        if self._multihop_rag is None:
            self._multihop_rag = MultiHopFIARulebook()
        return self._multihop_rag

    def get_pit_sim(self, circuit: str, team_pit_box_position: int = 1) -> PitStopSimulator:
        """Thread-safe pool lookup for PitStopSimulator without mutating global singleton state."""
        key = (str(circuit), int(team_pit_box_position))
        with self._pit_sim_lock:
            sim = self._pit_sim_pool.get(key)
            if sim is None:
                sim = PitStopSimulator(
                    circuit=key[0],
                    team_pit_box_position=key[1],
                    seed=42,
                )
                self._pit_sim_pool[key] = sim
            return sim

    def set_circuit(self, circuit: str, total_race_laps: int = 57, team_pit_box_position: int = 1):
        self.circuit = circuit
        self.pit_sim = self.get_pit_sim(circuit=circuit, team_pit_box_position=team_pit_box_position)
        self.total_race_laps = total_race_laps

    @classmethod
    def get(cls) -> "RaceEngineerToolkit":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


@tool
def get_live_race_standings() -> dict:
    """
    Fetches the latest live Grand Prix lap, track status (GREEN/VSC/SC), track temperature,
    and full driver standings (positions, tyre compounds, tyre ages, and gaps to leader)
    from DuckDB.
    """
    tk = RaceEngineerToolkit.get()
    if not os.path.exists(tk.db_path):
        return {"error": "No live telemetry database found."}

    with duckdb_connect(tk.db_path) as con:
        cur = con.execute("""
            SELECT lap, driver, position, compound, tyre_life, gap_to_leader, lap_time_sec, track_status, track_temp
            FROM live_telemetry_laps
            WHERE lap = (SELECT MAX(lap) FROM live_telemetry_laps)
            ORDER BY position ASC
        """)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    if not rows:
        return {"error": "Telemetry table is empty."}

    return {
        "lap": rows[0]["lap"],
        "total_race_laps": tk.total_race_laps,
        "circuit": tk.circuit,
        "track_status": rows[0]["track_status"],
        "track_temp_c": rows[0]["track_temp"],
        "standings": rows,
    }


@tool
def predict_tyre_degradation(
    driver: str,
    compound: str,
    tyre_life: int,
    track_temp: float = 32.0,
    race_lap: int | None = None,
    weather_condition: str | None = None,
    circuit: str | None = None,
) -> dict:
    """
    Queries the 56.86 KB L1 Physics Tensor Cache to calculate current lap tyre degradation (s/lap)
    and projects cumulative time loss over the next 5 laps for a given compound, tyre age,
    circuit asphalt abrasion, race lap (fuel/track evolution), and weather regime.
    """
    tk = RaceEngineerToolkit.get()
    active_circuit = circuit or tk.circuit
    current_deg = tk.tyre_cache.lookup_degradation(
        compound,
        track_temp,
        tyre_life,
        race_lap=race_lap,
        weather_condition=weather_condition,
        circuit=active_circuit,
    )
    next_5_laps = [
        round(
            tk.tyre_cache.lookup_degradation(
                compound,
                track_temp,
                tyre_life + k,
                race_lap=(race_lap + k) if race_lap is not None else None,
                weather_condition=weather_condition,
                circuit=active_circuit,
            ),
            3,
        )
        for k in range(1, 6)
    ]
    weather_penalty = TyreL1TensorCache.get_weather_compound_penalty(compound, weather_condition)
    abrasion = TyreL1TensorCache.get_circuit_abrasion_factor(active_circuit)
    return {
        "driver": driver,
        "compound": compound.upper(),
        "circuit": active_circuit,
        "circuit_abrasion_factor": round(abrasion, 2),
        "current_tyre_life": tyre_life,
        "race_lap": race_lap,
        "weather_condition": weather_condition or "Dry (Clear)",
        "weather_mismatch_penalty_sec": round(weather_penalty, 2),
        "current_deg_sec_per_lap": round(current_deg, 3),
        "projected_next_5_laps_deg": next_5_laps,
        "cumulative_5lap_loss_sec": round(sum(next_5_laps), 2),
    }


def compute_remaining_stint_pit_breakeven(
    ego_compound: str,
    ego_tyre_life: int,
    track_temp: float,
    laps_remaining: int,
    pit_loss_sec: float,
    target_new_tyre: str | None = None,
    warmup_penalty_sec: float | None = None,
    race_lap: int | None = None,
    weather_condition: str | None = None,
    circuit: str | None = None,
    p_clean_air: float | None = None,
    positions_dropped: int = 0,
    race_condition: str = "GREEN",
) -> dict:
    """
    Computes the cumulative remaining-stint tyre gain from pitting now onto `target_new_tyre`
    versus staying out on current tyres for the remaining `laps_remaining` laps, incorporating:
    1. Circuit asphalt abrasion scaling (`circuit_pit_data.json` relative to Bahrain = 1.00).
    2. Cold-track compound thermal activation window penalties (e.g., HARD below 30°C asphalt).
    3. Dirty-air traffic-rejoin cost weighted by `(1 - p_clean_air)`, positions dropped, and
       circuit overtaking difficulty (e.g., Hungaroring / Monaco traffic traps).
    4. Alternative compound & extend-stint counter-recommendation when the requested pit call fails.
    """
    tk = RaceEngineerToolkit.get()
    rem_laps = max(0, int(laps_remaining))
    ego_comp = str(ego_compound).upper()
    fresh_comp = (
        target_new_tyre.upper()
        if target_new_tyre
        else TyreL1TensorCache.recommended_compound_for_weather(
            weather_condition=weather_condition,
            laps_remaining=rem_laps,
            current_compound=ego_comp,
            track_temp=track_temp,
        )
    )
    ego_age = max(0, int(ego_tyre_life))
    warmup_loss = (
        float(warmup_penalty_sec)
        if warmup_penalty_sec is not None
        else TYRE_WARMUP_PENALTIES_SEC.get(fresh_comp, 1.2)
    )
    peak_windows = {"SOFT": 12.0, "MEDIUM": 18.0, "HARD": 25.0, "INTERMEDIATE": 15.0, "WET": 20.0}
    hard_ref_offset = COMPOUND_PACE_OFFSET_SEC.get("HARD", 0.50)
    old_fresh_bonus = max(0.0, hard_ref_offset - COMPOUND_PACE_OFFSET_SEC.get(ego_comp, 0.25))
    new_fresh_bonus = max(0.0, hard_ref_offset - COMPOUND_PACE_OFFSET_SEC.get(fresh_comp, 0.25))
    old_peak_win = peak_windows.get(ego_comp, 18.0)
    new_peak_win = peak_windows.get(fresh_comp, 18.0)

    def _eval_compound_raw(cand_comp: str) -> tuple[float, float]:
        cand_fresh_bonus = max(0.0, hard_ref_offset - COMPOUND_PACE_OFFSET_SEC.get(cand_comp, 0.25))
        cand_peak_win = peak_windows.get(cand_comp, 18.0)
        cum_gain = 0.0
        cold_loss_total = 0.0
        for k in range(1, rem_laps + 1):
            fut_lap = (race_lap + k) if race_lap is not None else None
            old_deg_k = tk.tyre_cache.lookup_degradation(
                ego_comp, track_temp, ego_age + k, race_lap=fut_lap, weather_condition=weather_condition, circuit=circuit
            )
            new_deg_k = tk.tyre_cache.lookup_degradation(
                cand_comp, track_temp, k, race_lap=fut_lap, weather_condition=weather_condition, circuit=circuit
            )
            old_cold_k = TyreL1TensorCache.get_cold_track_activation_penalty(
                ego_comp, track_temp, stint_lap=ego_age + k, weather_condition=weather_condition
            )
            new_cold_k = TyreL1TensorCache.get_cold_track_activation_penalty(
                cand_comp, track_temp, stint_lap=k, weather_condition=weather_condition
            )
            cold_loss_total += new_cold_k
            deg_gain = (old_deg_k + old_cold_k) - (new_deg_k + new_cold_k)
            old_bonus_k = old_fresh_bonus * max(0.0, 1.0 - (ego_age + k) / old_peak_win)
            new_bonus_k = cand_fresh_bonus * max(0.0, 1.0 - k / cand_peak_win)
            cum_gain += deg_gain + (new_bonus_k - old_bonus_k)
        return cum_gain, cold_loss_total

    raw_cum_gain, cold_activation_loss = _eval_compound_raw(fresh_comp)

    # Dirty-air traffic-rejoin cost (only when MC clean-air probability is supplied)
    traffic_penalty = 0.0
    if p_clean_air is not None and rem_laps > 0:
        dirty_prob = max(0.0, min(1.0, 1.0 - float(p_clean_air)))
        ov_diff = TyreL1TensorCache.get_circuit_overtaking_difficulty(circuit)
        pos_drop = min(4, max(0, int(positions_dropped)))
        cond_scale = 0.25 if str(race_condition).upper() in ("SC", "VSC") else 1.0
        traffic_penalty = round(
            min(12.0, (dirty_prob * 2.0 + pos_drop * dirty_prob * 1.05) * ov_diff * cond_scale),
            2,
        )

    recoverable_gain = round(
        max(0.0, raw_cum_gain - (warmup_loss if rem_laps > 0 else 0.0) - traffic_penalty),
        2,
    )
    pit_loss_rounded = round(float(pit_loss_sec), 2)
    net_payback = round(recoverable_gain - pit_loss_rounded, 2)

    # Check if the requested target compound itself violates the active weather regime
    # (e.g. asking for INTERMEDIATEs on a DRY track when currently on WETs -> skip Inters, go straight to slicks!)
    ego_weather_pen = TyreL1TensorCache.get_weather_compound_penalty(ego_comp, weather_condition)
    target_weather_pen = TyreL1TensorCache.get_weather_compound_penalty(fresh_comp, weather_condition)
    wrong_weather_target = bool(target_weather_pen > 0.0 and rem_laps > 0)
    optimal_weather_comp = TyreL1TensorCache.recommended_compound_for_weather(
        weather_condition=weather_condition,
        laps_remaining=rem_laps,
        current_compound=ego_comp,
        track_temp=track_temp,
    )

    rejection_reasons: list[str] = []
    if wrong_weather_target:
        wx_label = weather_condition or "Dry (Clear)"
        rejection_reasons.append(
            f"Wrong compound for {wx_label}: {fresh_comp} loses +{target_weather_pen:.1f}s/L and wastes a {pit_loss_rounded:.1f}s stop"
        )
    if cold_activation_loss >= 2.5:
        rejection_reasons.append(
            f"Cold {fresh_comp} activation loss (-{cold_activation_loss:.1f}s at {track_temp:.1f}°C)"
        )
    if traffic_penalty >= 3.0 and p_clean_air is not None:
        rejection_reasons.append(
            f"Heavy traffic trap ({p_clean_air*100:.1f}% clean air, -{traffic_penalty:.1f}s dirty-air cost)"
        )

    # Generate counter-proposal if the requested stop is wrong-weather, negative payback, or thermally compromised
    alternative_recommendation = ""
    if wrong_weather_target:
        opt_raw, _ = _eval_compound_raw(optimal_weather_comp)
        opt_warm = TYRE_WARMUP_PENALTIES_SEC.get(optimal_weather_comp, 1.2)
        opt_gain = round(max(0.0, opt_raw - opt_warm - traffic_penalty), 2)
        opt_net = round(opt_gain - pit_loss_rounded, 2)
        if ego_weather_pen > 0.0 and opt_net > 0.0:
            alternative_recommendation = (
                f"SKIP {fresh_comp}s — box straight for {optimal_weather_comp}s "
                f"(0.0s/L weather penalty, net {opt_net:+.1f}s, avoids double-stopping!)"
            )
        else:
            alternative_recommendation = (
                f"STAY OUT on {ego_comp}s — do not pit onto {fresh_comp}s under {weather_condition or 'Dry'} conditions"
            )
    elif rem_laps > 8 and (net_payback <= 0.0 or cold_activation_loss >= 4.0):
        regime = TyreL1TensorCache.normalize_weather_regime(weather_condition)
        if regime == "DRY":
            best_alt_comp = None
            best_alt_net = net_payback
            for cand in ("SOFT", "MEDIUM", "HARD"):
                if cand == fresh_comp:
                    continue
                alt_raw, _ = _eval_compound_raw(cand)
                alt_warm = TYRE_WARMUP_PENALTIES_SEC.get(cand, 1.2)
                alt_gain = max(0.0, alt_raw - alt_warm - traffic_penalty)
                alt_net = round(alt_gain - pit_loss_rounded, 2)
                if alt_net > best_alt_net + 2.5:
                    best_alt_net = alt_net
                    best_alt_comp = cand

            if best_alt_comp and best_alt_net > 0.0:
                alternative_recommendation = (
                    f"Switch target to {best_alt_comp} (net {best_alt_net:+.1f}s vs {fresh_comp} {net_payback:+.1f}s)"
                )
            elif traffic_penalty >= 3.0 or cold_activation_loss >= 2.5:
                extend_laps = min(8, max(5, rem_laps - 18)) if rem_laps > 22 else max(3, rem_laps // 3)
                late_target = "SOFT" if (rem_laps - extend_laps) <= 24 else "MEDIUM"
                alternative_recommendation = (
                    f"Extend {ego_comp}s {extend_laps} more laps in clean air, then pit onto {late_target}s once traffic clears"
                )

    return {
        "fresh_compound": fresh_comp,
        "laps_remaining": rem_laps,
        "pit_loss_sec": pit_loss_rounded,
        "warmup_penalty_sec": round(warmup_loss if rem_laps > 0 else 0.0, 2),
        "circuit_abrasion_factor": round(TyreL1TensorCache.get_circuit_abrasion_factor(circuit), 2),
        "cold_track_penalty_sec": round(cold_activation_loss, 2),
        "traffic_rejoin_penalty_sec": round(traffic_penalty, 2),
        "wrong_weather_target": wrong_weather_target,
        "optimal_weather_compound": optimal_weather_comp,
        "rejection_reasons": rejection_reasons,
        "alternative_recommendation": alternative_recommendation,
        "recoverable_tyre_gain_sec": recoverable_gain,
        "net_stint_pit_payback_sec": net_payback,
        "pit_breakeven_reached": bool(net_payback > 0.0 and not wrong_weather_target),
    }


def _analyze_rival_catch_battle_impl(
    ego_driver: str,
    target_driver: str | None = None,
    target_new_tyre: str | None = None,
    live_data: dict | None = None,
    rejoin_sim: dict | None = None,
    pit_sim: PitStopSimulator | None = None,
    total_race_laps: int | None = None,
) -> dict:
    tk = RaceEngineerToolkit.get()
    active_pit_sim = pit_sim if pit_sim is not None else tk.pit_sim
    live = live_data if live_data is not None else get_live_race_standings.invoke({})
    standings = live.get("standings", [])
    if not standings:
        return {"error": "No standings available."}

    current_lap = int(live.get("lap", 17))
    default_total = total_race_laps if total_race_laps is not None else tk.total_race_laps
    total_laps = max(current_lap, int(live.get("total_race_laps", default_total)))
    laps_remaining = max(0, total_laps - current_lap)
    track_temp = float(live.get("track_temp_c", 32.0))
    track_status = str(live.get("track_status", "GREEN")).upper()
    weather_cond = live.get("weather_condition")

    ego_row = next((r for r in standings if r["driver"] == ego_driver), standings[0])
    ego_pos = int(ego_row["position"])
    ego_gap = float(ego_row["gap_to_leader"])
    ego_lap_time = float(ego_row["lap_time_sec"])
    ego_comp = str(ego_row["compound"]).upper()
    ego_age = int(ego_row["tyre_life"])
    circuit_name = str(live.get("circuit", active_pit_sim.circuit))
    ego_deg = tk.tyre_cache.lookup_degradation(
        ego_comp, track_temp, ego_age, race_lap=current_lap, weather_condition=weather_cond, circuit=circuit_name
    )

    # Pick target rival: explicit target_driver if present (and not ego_driver itself), else car ahead (if P>1), else car behind (P2)
    rival_row = None
    target_note = ""
    target_in_grid = True
    req_target = target_driver.upper() if target_driver else None

    if req_target:
        if req_target == ego_driver.upper():
            target_note = f"(You are {ego_driver.upper()} in P{ego_pos}; comparing against nearest rival) "
        else:
            rival_row = next((r for r in standings if r["driver"] == req_target), None)
            if rival_row is None:
                target_in_grid = False
                target_note = f"NOTE: {req_target} is not in the active {len(standings)}-car grid; comparing against nearest rival instead — "

    if rival_row is None:
        if ego_pos > 1:
            rival_row = next((r for r in standings if int(r["position"]) == ego_pos - 1 and r["driver"] != ego_driver), None)
        if rival_row is None and len(standings) >= 2:
            rival_row = next((r for r in standings if r["driver"] != ego_driver), None)

    if rival_row is None:
        return {"error": "No rival car found on track."}

    rival_drv = rival_row["driver"]
    rival_pos = int(rival_row["position"])
    rival_gap = float(rival_row["gap_to_leader"])
    rival_lap_time = float(rival_row["lap_time_sec"])
    rival_comp = str(rival_row["compound"]).upper()
    rival_age = int(rival_row["tyre_life"])
    rival_deg = tk.tyre_cache.lookup_degradation(
        rival_comp, track_temp, rival_age, race_lap=current_lap, weather_condition=weather_cond, circuit=circuit_name
    )

    # Positive gap_between means rival is AHEAD of ego; negative means rival is BEHIND ego
    signed_gap = rival_gap - ego_gap  # if rival_gap < ego_gap, rival is ahead (signed_gap < 0)
    abs_gap = round(abs(ego_gap - rival_gap), 3)
    rival_is_ahead = rival_pos < ego_pos

    # Compute Monte Carlo pit rejoin (reusing precomputed result if matching driver & track condition)
    if (
        rejoin_sim is None
        or rejoin_sim.get("driver") != ego_row["driver"]
        or rejoin_sim.get("condition") != track_status
    ):
        base_lap_sec = float(get_circuit_metadata(circuit_name)["base_lap_sec"])
        traffic = [
            {
                "driver": c["driver"],
                "gap_to_leader": float(c["gap_to_leader"]),
                "is_lapped": float(c["gap_to_leader"]) >= base_lap_sec,
            }
            for c in standings
            if c["driver"] != ego_row["driver"]
        ]
        rejoin_sim = active_pit_sim.simulate_rejoin_mc(
            driver=ego_row["driver"],
            current_gap_to_leader=ego_gap,
            traffic=traffic,
            race_condition=track_status,
            n_sims=10_000,
            base_lap_sec=base_lap_sec,
        )
    pit_loss = float(rejoin_sim["median_pit_loss"])
    rejoin_pos = int(rejoin_sim["median_pos"])

    breakeven = compute_remaining_stint_pit_breakeven(
        ego_compound=ego_comp,
        ego_tyre_life=ego_age,
        track_temp=track_temp,
        laps_remaining=laps_remaining,
        pit_loss_sec=pit_loss,
        target_new_tyre=target_new_tyre,
        race_lap=current_lap,
        weather_condition=weather_cond,
        circuit=circuit_name,
        p_clean_air=float(rejoin_sim["p_clean_air"]),
        positions_dropped=max(0, rejoin_pos - ego_pos),
        race_condition=track_status,
    )
    fresh_comp = breakeven["fresh_compound"]
    recoverable_tyre_gain_sec = breakeven["recoverable_tyre_gain_sec"]
    net_stint_pit_payback_sec = breakeven["net_stint_pit_payback_sec"]

    # Compute multi-lap pace projection including L1 Tensor tyre degradation evolution
    # Average net pace gain per lap over the next 5 laps:
    # (rival_lap_time - ego_lap_time) + future tyre degradation difference
    ego_5l_deg = sum(
        tk.tyre_cache.lookup_degradation(ego_comp, track_temp, ego_age + k, circuit=circuit_name)
        for k in range(1, 6)
    ) / 5.0
    rival_5l_deg = sum(
        tk.tyre_cache.lookup_degradation(rival_comp, track_temp, rival_age + k, circuit=circuit_name)
        for k in range(1, 6)
    ) / 5.0
    deg_advantage_per_lap = round(rival_5l_deg - ego_5l_deg, 3)
    raw_lap_delta = round(rival_lap_time - ego_lap_time, 3)  # >0 means ego is faster than rival

    # Detect pit-in / pit-out outlier laps under GREEN flag (e.g. rival slowed entering pit lane or on out-lap)
    # so we never mistake a pit-in lap for genuine slow racing pace!
    pit_lap_note = ""
    rival_pit_outlier = (track_status == "GREEN") and (rival_age <= 1 or raw_lap_delta > 2.8)
    ego_pit_outlier = (track_status == "GREEN") and (ego_age <= 1 or raw_lap_delta < -2.8)
    if rival_pit_outlier or ego_pit_outlier:
        comp_pace_adv = (
            COMPOUND_PACE_OFFSET_SEC.get(rival_comp, 0.25)
            - COMPOUND_PACE_OFFSET_SEC.get(ego_comp, 0.25)
        )
        effective_pace_advantage = round(deg_advantage_per_lap + comp_pace_adv, 3)
        if rival_pit_outlier:
            pit_lap_note = (
                f"⚠️ RIVAL PIT LAP DETECTED ({rival_drv} lap {rival_lap_time:.2f}s reflects pit-in/out — "
                f"projecting true green-flag pace from tyre physics): "
            )
        else:
            pit_lap_note = (
                f"⚠️ EGO PIT LAP DETECTED (your lap {ego_lap_time:.2f}s reflects pit-in/out — "
                f"projecting true green-flag pace from tyre physics): "
            )
    else:
        # Effective closing rate (s/lap): positive means ego is faster than rival
        effective_pace_advantage = round(raw_lap_delta + 0.5 * deg_advantage_per_lap, 3)

    if abs(effective_pace_advantage) < 0.03:
        effective_pace_advantage = round(deg_advantage_per_lap if abs(deg_advantage_per_lap) >= 0.03 else 0.15, 3)

    if track_status in ("SC", "VSC"):
        # Under Safety Car / VSC, overtaking is prohibited and lap-time deltas are neutralized
        laps_to_catch = 999.0 if rival_is_ahead else 0.0
        can_catch = False if rival_is_ahead else True
        rel_pos_txt = (
            f"{rival_drv} (P{rival_pos}) is {abs_gap:.2f}s ahead"
            if rival_is_ahead
            else f"{rival_drv} (P{rival_pos}) is {abs_gap:.2f}s behind"
        )
        if net_stint_pit_payback_sec > 0.0 and laps_remaining > 0:
            verdict = (
                f"{track_status} ACTIVE — On-track overtaking & pace deltas are neutralized ({rel_pos_txt}). "
                f"Take the cheap {track_status} pit window for fresh {fresh_comp} tyres to gain a restart tyre offset "
                f"(+{recoverable_tyre_gain_sec:.1f}s over {laps_remaining}L vs ~{pit_loss:.1f}s {track_status} pit loss, net {net_stint_pit_payback_sec:+.1f}s)."
            )
        else:
            verdict = (
                f"{track_status} ACTIVE — On-track overtaking & pace deltas are neutralized ({rel_pos_txt}). "
                f"With only {laps_remaining} laps left, pitting costs ~{pit_loss:.1f}s vs +{recoverable_tyre_gain_sec:.1f}s tyre gain "
                f"(net {net_stint_pit_payback_sec:+.1f}s) — DO NOT BOX, stay out and hold P{ego_pos} for the restart."
            )
    elif rival_is_ahead:
        # Ego is chasing rival ahead
        closing_rate = effective_pace_advantage
        if closing_rate > 0.01:
            laps_to_catch = round(abs_gap / closing_rate, 1)
            can_catch = laps_to_catch <= laps_remaining
            catch_lap = min(total_laps, int(round(current_lap + laps_to_catch)))
            if can_catch:
                verdict = (
                    f"YES (ON-TRACK CATCH) — You are catching {rival_drv} (P{rival_pos}) by +{closing_rate:.2f}s/lap "
                    f"and will enter DRS/overtake range in ~{laps_to_catch:.1f} laps (around Lap {catch_lap}/{total_laps})."
                )
            elif net_stint_pit_payback_sec <= 0.0 or recoverable_tyre_gain_sec < (pit_loss + abs_gap):
                verdict = (
                    f"NO ON-TRACK CATCH — You gain +{closing_rate:.2f}s/lap on {rival_drv} (gap {abs_gap:.2f}s), "
                    f"requiring ~{laps_to_catch:.1f} laps with only {laps_remaining} laps left. "
                    f"With only {laps_remaining} laps left, pitting costs ~{pit_loss:.1f}s while fresh {fresh_comp} tyres "
                    f"only recover ~{recoverable_tyre_gain_sec:.1f}s (net loss {net_stint_pit_payback_sec:+.1f}s, dropping P{ego_pos}->P{rejoin_pos}) — "
                    f"DO NOT BOX — stay out, manage tyres, and hold P{ego_pos}."
                )
            else:
                verdict = (
                    f"NO ON-TRACK CATCH ON CURRENT TYRES — You gain +{closing_rate:.2f}s/lap on {rival_drv} (gap {abs_gap:.2f}s), "
                    f"requiring ~{laps_to_catch:.1f} laps ({laps_remaining} laps remaining). "
                    f"Recommend an undercut pit stop onto fresh {fresh_comp}s (+{recoverable_tyre_gain_sec:.1f}s stint tyre gain vs ~{pit_loss:.1f}s pit loss, net {net_stint_pit_payback_sec:+.1f}s)."
                )
        else:
            laps_to_catch = 999.0
            can_catch = False
            if net_stint_pit_payback_sec <= 0.0 or recoverable_tyre_gain_sec < (pit_loss + abs_gap):
                verdict = (
                    f"NO ON-TRACK CATCH — {rival_drv} (P{rival_pos}) is currently pulling away by {abs(closing_rate):.2f}s/lap "
                    f"(current gap: {abs_gap:.2f}s). With only {laps_remaining} laps left, pitting costs ~{pit_loss:.1f}s "
                    f"while fresh {fresh_comp} tyres only recover ~{recoverable_tyre_gain_sec:.1f}s "
                    f"(net loss {net_stint_pit_payback_sec:+.1f}s, dropping P{ego_pos}->P{rejoin_pos}) — "
                    f"DO NOT BOX — stay out, manage tyres, and hold P{ego_pos}."
                )
            else:
                verdict = (
                    f"ON-TRACK CATCH UNLIKELY — {rival_drv} (P{rival_pos}) is pulling away by {abs(closing_rate):.2f}s/lap "
                    f"(current gap: {abs_gap:.2f}s). Recommend pitting for fresh {fresh_comp} tyres to create a tyre offset "
                    f"(+{recoverable_tyre_gain_sec:.1f}s tyre gain over {laps_remaining} laps vs ~{pit_loss:.1f}s pit loss, net {net_stint_pit_payback_sec:+.1f}s)."
                )
    else:
        # Rival is behind ego_driver
        gap_change_per_lap = effective_pace_advantage  # >0 means ego pulls away; <0 means rival is catching ego
        if gap_change_per_lap >= 0:
            laps_to_catch = 0.0
            can_catch = True
            verdict = (
                f"{pit_lap_note}YOU ARE ALREADY AHEAD OF {rival_drv} (You are P{ego_pos}, {rival_drv} is P{rival_pos}, +{abs_gap:.2f}s behind you) "
                f"and extending your gap by +{gap_change_per_lap:.2f}s/lap."
            )
        else:
            rival_closing = abs(gap_change_per_lap)
            laps_until_caught = round(abs_gap / max(0.01, rival_closing), 1)
            can_catch = laps_until_caught > laps_remaining
            verdict = (
                f"{pit_lap_note}YOU ARE AHEAD OF {rival_drv} (P{rival_pos} is +{abs_gap:.2f}s behind you), "
                f"WARNING: {rival_drv} is closing at {rival_closing:.2f}s/lap and could catch you in ~{laps_until_caught:.1f} laps!"
            )
            laps_to_catch = laps_until_caught

    return {
        "ego_driver": ego_driver,
        "ego_position": ego_pos,
        "ego_compound": ego_comp,
        "ego_tyre_life": ego_age,
        "ego_lap_time_sec": round(ego_lap_time, 3),
        "ego_deg_sec_per_lap": round(ego_deg, 3),
        "requested_target_driver": req_target,
        "target_in_grid": target_in_grid,
        "target_driver": rival_drv,
        "target_position": rival_pos,
        "target_compound": rival_comp,
        "target_tyre_life": rival_age,
        "target_lap_time_sec": round(rival_lap_time, 3),
        "target_deg_sec_per_lap": round(rival_deg, 3),
        "rival_is_ahead": rival_is_ahead,
        "gap_between_sec": abs_gap,
        "signed_gap_sec": round(signed_gap, 3),
        "effective_pace_advantage_sec_per_lap": effective_pace_advantage,
        "laps_remaining": laps_remaining,
        "estimated_laps_to_intersect": laps_to_catch,
        "can_catch_or_hold": can_catch,
        "fresh_pit_compound": fresh_comp,
        "expected_pit_loss_sec": round(pit_loss, 2),
        "projected_rejoin_position": rejoin_pos,
        "recoverable_tyre_gain_sec": recoverable_tyre_gain_sec,
        "net_stint_pit_payback_sec": net_stint_pit_payback_sec,
        "pit_breakeven_reached": bool(net_stint_pit_payback_sec > 0.0),
        "tactical_verdict": f"{target_note}{verdict}",
        "rejoin_simulation": rejoin_sim,
    }


@tool
def analyze_rival_catch_battle(
    ego_driver: str,
    target_driver: str | None = None,
    target_new_tyre: str | None = None,
) -> dict:
    """
    Calculates exact on-track gap, lap-time pace delta, L1 Tensor tyre wear delta,
    laps remaining, remaining-stint pit breakeven physics, and projected catch/defend lap
    between ego_driver and a specific target_driver (or nearest ahead/behind rival).
    """
    return _analyze_rival_catch_battle_impl(
        ego_driver=ego_driver,
        target_driver=target_driver,
        target_new_tyre=target_new_tyre,
    )


def _simulate_pit_stop_mc_impl(
    driver: str,
    race_condition: str = "GREEN",
    target_new_tyre: str = "HARD",
    rival_gap_sec: float = 1.5,
    live_data: dict | None = None,
    rejoin_sim: dict | None = None,
    pit_sim: PitStopSimulator | None = None,
    total_race_laps: int | None = None,
) -> dict:
    tk = RaceEngineerToolkit.get()
    active_pit_sim = pit_sim if pit_sim is not None else tk.pit_sim
    live = live_data if live_data is not None else get_live_race_standings.invoke({})
    standings = live.get("standings", [])

    current_lap = int(live.get("lap", 17))
    default_total = total_race_laps if total_race_laps is not None else tk.total_race_laps
    total_laps = max(current_lap, int(live.get("total_race_laps", default_total)))
    laps_remaining = max(0, total_laps - current_lap)
    track_temp = float(live.get("track_temp_c", 32.0))
    weather_cond = live.get("weather_condition")
    cond_upper = race_condition.upper()

    circuit_name = str(live.get("circuit", active_pit_sim.circuit))
    base_lap_sec = float(get_circuit_metadata(circuit_name)["base_lap_sec"])

    ego_gap = 0.0
    ego_pos = 1
    old_tyre = "SOFT"
    ego_age = 15
    traffic = []
    for car in standings:
        if car["driver"] == driver:
            ego_gap = float(car["gap_to_leader"])
            ego_pos = int(car["position"])
            old_tyre = str(car["compound"]).upper()
            ego_age = int(car["tyre_life"])
        else:
            c_gap = float(car["gap_to_leader"])
            traffic.append({
                "driver": car["driver"],
                "gap_to_leader": c_gap,
                "is_lapped": c_gap >= base_lap_sec,
            })

    tm_gap_ahead = live.get("teammate_gap_ahead_sec")
    tm_pitting = bool(live.get("teammate_pitting", False))
    used_dry = live.get("used_dry_compounds") or ()

    if (
        rejoin_sim is not None
        and rejoin_sim.get("driver") == driver
        and rejoin_sim.get("condition") == cond_upper
        and not tm_pitting
    ):
        rejoin_res = rejoin_sim
    else:
        rejoin_res = active_pit_sim.simulate_rejoin_mc(
            driver=driver,
            current_gap_to_leader=ego_gap,
            traffic=traffic,
            race_condition=cond_upper,
            n_sims=10_000,
            base_lap_sec=base_lap_sec,
            teammate_gap_ahead_sec=tm_gap_ahead,
            teammate_pitting=tm_pitting,
        )
    target_comp = target_new_tyre.upper()
    old_tyre_deg = tk.tyre_cache.lookup_degradation(
        old_tyre, track_temp, ego_age, race_lap=current_lap, weather_condition=weather_cond, circuit=circuit_name
    )
    new_tyre_deg = tk.tyre_cache.lookup_degradation(
        target_comp, track_temp, 1, race_lap=current_lap + 1, weather_condition=weather_cond, circuit=circuit_name
    ) + TyreL1TensorCache.get_cold_track_activation_penalty(
        target_comp, track_temp, stint_lap=1, weather_condition=weather_cond
    )

    undercut_res = active_pit_sim.calculate_undercut_mc(
        rival_gap=rival_gap_sec,
        new_tyre=target_comp,
        old_tyre=old_tyre,
        race_condition=cond_upper,
        n_sims=10_000,
        old_tyre_deg=old_tyre_deg,
        new_tyre_deg=new_tyre_deg,
    )
    rejoin_pos = int(rejoin_res["median_pos"])
    breakeven = compute_remaining_stint_pit_breakeven(
        ego_compound=old_tyre,
        ego_tyre_life=ego_age,
        track_temp=track_temp,
        laps_remaining=laps_remaining,
        pit_loss_sec=float(rejoin_res["median_pit_loss"]),
        target_new_tyre=target_comp,
        race_lap=current_lap,
        weather_condition=weather_cond,
        circuit=circuit_name,
        p_clean_air=float(rejoin_res["p_clean_air"]),
        positions_dropped=max(0, rejoin_pos - ego_pos),
        race_condition=cond_upper,
    )
    breakeven["ego_position"] = ego_pos
    breakeven["rejoin_position"] = rejoin_pos

    strategy_ranking = tk.strategy_optimizer.evaluate_all_strategies(
        ego_compound=old_tyre,
        ego_tyre_life=ego_age,
        ego_pos=ego_pos,
        track_temp=track_temp,
        current_lap=current_lap,
        total_laps=total_laps,
        pit_loss_now_sec=float(rejoin_res["median_pit_loss"]),
        green_flag_pit_loss_sec=float(rejoin_res.get("green_flag_pit_loss", rejoin_res["median_pit_loss"])),
        rejoin_pos_now=rejoin_pos,
        p_clean_air_now=float(rejoin_res["p_clean_air"]),
        circuit=circuit_name,
        weather_condition=weather_cond,
        race_condition=cond_upper,
        requested_target_tyre=target_comp,
        used_dry_compounds=used_dry,
        double_stack_delay_sec=float(rejoin_res.get("double_stack_delay_sec", 0.0)),
    )
    return {
        "rejoin_simulation": rejoin_res,
        "undercut_simulation": undercut_res,
        "stint_breakeven": breakeven,
        "strategy_ranking": strategy_ranking,
    }


@tool
def simulate_pit_stop_mc(
    driver: str,
    race_condition: str = "GREEN",
    target_new_tyre: str = "HARD",
    rival_gap_sec: float = 1.5,
) -> dict:
    """
    Runs 10,000 Vectorized Monte Carlo Pit Rejoin and Undercut simulations (<1.5ms)
    using live traffic gaps from DuckDB. Returns median pit loss, rejoin position,
    clean air probability, undercut win probability, and remaining-stint pit breakeven physics.
    """
    return _simulate_pit_stop_mc_impl(
        driver=driver,
        race_condition=race_condition,
        target_new_tyre=target_new_tyre,
        rival_gap_sec=rival_gap_sec,
    )


@tool
def consult_fia_rulebook_multihop(legal_question: str) -> dict:
    """
    Executes a 2-Hop FIA Sporting Regulations legal search in Qdrant.
    Automatically detects cited articles (e.g., Article 54.3) and resolves their exact definitions.
    Degrades gracefully with `offline=True` if Qdrant is unreachable.
    """
    tk = RaceEngineerToolkit.get()
    try:
        return tk.multihop_rag.ask_multihop(legal_question, top_k=2, max_hops=2)
    except Exception as e:
        logger.warning(f"⚠️ [Legal RAG Tool] FIA rulebook offline ({e}). Returning graceful fallback.")
        return {
            "question": legal_question,
            "hop1_primary_rules": [],
            "cited_articles": [],
            "hop2_resolved_references": [],
            "elapsed_ms": 0.0,
            "offline": True,
            "error": f"FIA rulebook offline: {e}",
        }


ALL_ENGINEER_TOOLS = [
    get_live_race_standings,
    predict_tyre_degradation,
    analyze_rival_catch_battle,
    simulate_pit_stop_mc,
    consult_fia_rulebook_multihop,
]


if __name__ == "__main__":
    print("\n" + "=" * 65 + "\nTESTING RACE ENGINEER TOOLBELT (TASK 5.2)\n" + "=" * 65)
    t1 = get_live_race_standings.invoke({})
    logger.info(f"Tool 1 [Live Standings]: Lap {t1.get('lap')} | Track: {t1.get('track_status')} | Cars: {len(t1.get('standings', []))}")

    t2 = predict_tyre_degradation.invoke({"driver": "VER", "compound": "SOFT", "tyre_life": 17, "track_temp": 32.0})
    logger.info(f"Tool 2 [Tyre Physics]: {t2}")

    t2b = analyze_rival_catch_battle.invoke({"ego_driver": "VER", "target_driver": "HAM"})
    logger.info(f"Tool 2b [Rival Catch Battle]: {t2b}")

    t3 = simulate_pit_stop_mc.invoke({"driver": "VER", "race_condition": "SC", "target_new_tyre": "MEDIUM", "rival_gap_sec": 1.2})
    logger.info(f"Tool 3 [Monte Carlo Pit Sim]: Rejoin P{t3['rejoin_simulation']['median_pos']} | Clean Air: {t3['rejoin_simulation']['p_clean_air']*100:.1f}%")

    t4 = consult_fia_rulebook_multihop.invoke({"legal_question": "What is the penalty for overtaking under the Safety Car?"})
    logger.info(f"Tool 4 [Multi-Hop FIA RAG]: Hop1={len(t4['hop1_primary_rules'])} rules | Hop2 Resolved={[h['article'] for h in t4['hop2_resolved_references']]}")
