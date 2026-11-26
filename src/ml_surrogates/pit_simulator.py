import json
import os
import random
import time

import numpy as np
from loguru import logger

# Canonical single-lap cold-tyre warmup penalties (seconds lost on out-lap vs warmed rubber)
TYRE_WARMUP_PENALTIES_SEC: dict[str, float] = {
    "SOFT": 0.5,
    "MEDIUM": 1.2,
    "HARD": 2.2,
    "INTERMEDIATE": 0.8,
    "WET": 1.0,
}

# Intrinsic fresh-compound pace offset relative to fresh SOFT (seconds/lap)
COMPOUND_PACE_OFFSET_SEC: dict[str, float] = {
    "SOFT": 0.0,
    "MEDIUM": 0.25,
    "HARD": 0.50,
    "INTERMEDIATE": 1.0,
    "WET": 2.0,
}


class PitStopSimulator:
    def __init__(self, circuit: str, team_pit_box_position: int = 1, seed: int | None = None):
        """
        team_pit_box_position: 1 is closest to pit entry (e.g. Mercedes/Red Bull),
                               10 is furthest (e.g. Haas/Sauber).
        """
        self.circuit = circuit
        self.team_pit_box_position = team_pit_box_position
        self.rng = np.random.default_rng(seed)

        # Load the full 24-race database from our central config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        config_path = os.path.join(project_root, "config", "circuit_pit_data.json")

        with open(config_path, "r") as f:
            self.pit_db = json.load(f)

        if circuit not in self.pit_db:
            logger.warning(f"Circuit {circuit} not in DB. Defaulting to Bahrain metrics.")
            self.circuit_data = self.pit_db["Bahrain"]
        else:
            self.circuit_data = self.pit_db[circuit]

        logger.info(f"🏎️ Elite Pit Simulator Initialized for {circuit} (Pit Box #{self.team_pit_box_position})")

    def _get_execution_time(self, silent: bool = False) -> float:
        """Calculates stationary pit stop time including human variance and botched stops."""
        if self.rng.random() < 0.02:
            botch_penalty = float(self.rng.uniform(2.5, 6.0))
            if not silent:
                logger.error(f"🔧 DISASTER! Botched pit stop. Lost an extra {botch_penalty:.1f}s!")
            return 2.5 + botch_penalty

        stationary_time = max(1.8, float(self.rng.normal(loc=2.4, scale=0.2)))
        return stationary_time

    def _get_traffic_hold_delay(self, race_condition: str, silent: bool = False) -> float:
        """Calculates delay from being held in the box due to fast-lane traffic."""
        if race_condition in ["SC", "VSC"]:
            hold_prob = 0.05 * self.team_pit_box_position
            if self.rng.random() < hold_prob:
                hold_time = float(self.rng.uniform(1.5, 4.0))
                if not silent:
                    logger.warning(f"🛑 UNSAFE RELEASE AVOIDANCE: Held in box for {hold_time:.1f}s due to traffic!")
                return hold_time
        return 0.0

    def get_total_pit_loss(self, race_condition: str = "GREEN", silent: bool = False) -> float:
        """Calculates the true Net Green Loss using Kinematic Reference Frame math."""
        t_pit_base = self.circuit_data["t_pit"]
        l_norm_base = self.circuit_data["normal_loss"]
        t_track_base = t_pit_base - l_norm_base

        actual_stationary = self._get_execution_time(silent=silent)
        hold_delay = self._get_traffic_hold_delay(race_condition, silent=silent)

        variance_penalty = (actual_stationary - 2.4) + hold_delay
        t_pit_actual = t_pit_base + variance_penalty

        if race_condition == "VSC":
            t_track_bypass = t_track_base / 0.65
            net_loss = t_pit_actual - t_track_bypass
            if not silent:
                logger.info(f"🟡 VSC Pit Stop! Net track time lost: {net_loss:.2f}s (Cheap Stop)")
            return net_loss

        elif race_condition == "SC":
            t_track_bypass = t_track_base / 0.52
            net_loss = t_pit_actual - t_track_bypass
            if not silent:
                logger.info(f"🟡 SC Pit Stop! Net track time lost: {net_loss:.2f}s (Extremely Cheap Stop)")
            return net_loss

        net_loss = l_norm_base + variance_penalty
        if not silent:
            logger.info(f"🟢 GREEN Flag Pit Stop! Net track time lost: {net_loss:.2f}s")
        return net_loss

    # =========================================================================
    # TASK 4.2: SIMD Vectorized Monte Carlo Engine (< 1.0 ms for 10,000 runs)
    # =========================================================================
    def _sample_pit_loss_vectorized(self, race_condition: str, n_sims: int) -> np.ndarray:
        """Generates N pit-loss samples simultaneously using NumPy SIMD broadcasting."""
        t_pit_base = self.circuit_data["t_pit"]
        l_norm_base = self.circuit_data["normal_loss"]
        t_track_base = t_pit_base - l_norm_base

        # 1. Stationary stop distribution (Gaussian mean=2.4, std=0.2, clamped >= 1.8s)
        stationary = np.maximum(1.8, self.rng.normal(loc=2.4, scale=0.2, size=n_sims))

        # 2. 2% Botched stop mask (+2.5s to +6.0s penalty on 2.5s base)
        botch_mask = self.rng.random(n_sims) < 0.02
        botch_penalties = self.rng.uniform(2.5, 6.0, size=n_sims)
        stationary = np.where(botch_mask, 2.5 + botch_penalties, stationary)

        # 3. Traffic hold delay under SC/VSC
        if race_condition in ("SC", "VSC"):
            hold_prob = 0.05 * self.team_pit_box_position
            hold_mask = self.rng.random(n_sims) < hold_prob
            hold_delays = np.where(hold_mask, self.rng.uniform(1.5, 4.0, size=n_sims), 0.0)
        else:
            hold_delays = np.zeros(n_sims, dtype=np.float64)

        variance_penalty = (stationary - 2.4) + hold_delays
        t_pit_actual = t_pit_base + variance_penalty

        if race_condition == "VSC":
            return t_pit_actual - (t_track_base / 0.65)
        elif race_condition == "SC":
            return t_pit_actual - (t_track_base / 0.52)
        return l_norm_base + variance_penalty

    def simulate_rejoin_mc(
        self,
        driver: str,
        current_gap_to_leader: float,
        traffic: list,
        race_condition: str = "GREEN",
        n_sims: int = 10_000,
        silent: bool = False,
        base_lap_sec: float = 90.0,
        teammate_gap_ahead_sec: float | None = None,
        teammate_pitting: bool = False,
    ) -> dict:
        """
        Vectorized Monte Carlo Pit Rejoin Simulator.
        Computes 10,000 stochastic pit-stop outcomes in ~1ms using 2D broadcasting:
        - Classification (`rejoin_positions`) compares cumulative race time (excluding lapped cars
          from demoting a lead-lap car's race position).
        - Physical clean/dirty air (`p_clean_air`, `p_dirty_air`, `p_blue_flag_traffic`) evaluates
          circular on-track phase distance (`gap % base_lap_sec`) across ALL cars, including
          lapped backmarkers that can ruin an out-lap under blue flags.
        - Teammate Double-Stack Delay: when `teammate_pitting=True` and `0 <= teammate_gap_ahead_sec < 4.5s`,
          injects a realistic queuing delay behind the teammate in the shared pit box.
        """
        t0 = time.perf_counter()
        lap_sec = max(55.0, float(base_lap_sec))

        # Shape: (N,)
        pit_losses = self._sample_pit_loss_vectorized(race_condition, n_sims)
        double_stack_delay = 0.0
        if teammate_pitting and teammate_gap_ahead_sec is not None and 0.0 <= float(teammate_gap_ahead_sec) < 4.5:
            gap_tm = float(teammate_gap_ahead_sec)
            # Car 1 needs ~2.5s stationary + ~2.0s box exit/entry clearance (~4.5s safe buffer)
            queue_samples = np.maximum(0.8, (4.5 - gap_tm) + self.rng.normal(loc=1.2, scale=0.4, size=n_sims))
            pit_losses = pit_losses + queue_samples
            double_stack_delay = round(float(np.median(queue_samples)), 2)

        new_gaps = current_gap_to_leader + pit_losses  # Shape: (N,)
        green_flag_loss = float(self.circuit_data["normal_loss"])
        median_loss = float(np.median(pit_losses))
        saved_vs_green = float(max(0.0, green_flag_loss - median_loss))

        if not traffic:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "driver": driver,
                "condition": race_condition,
                "green_flag_pit_loss": green_flag_loss,
                "pit_loss_saved_vs_green": saved_vs_green,
                "double_stack_delay_sec": double_stack_delay,
                "median_pit_loss": median_loss,
                "p95_pit_loss": float(np.percentile(pit_losses, 95)),
                "expected_pos": 1.0,
                "median_pos": 1,
                "p_clean_air": 1.0,
                "p_dirty_air": 0.0,
                "p_blue_flag_traffic": 0.0,
                "elapsed_ms": elapsed_ms,
            }

        # Shape: (M,) where M is number of rival cars
        traffic_gaps = np.array([float(car["gap_to_leader"]) for car in traffic], dtype=np.float64)
        is_lapped_mask = np.array(
            [bool(car.get("is_lapped", False)) or (float(car["gap_to_leader"]) >= lap_sec) for car in traffic],
            dtype=bool,
        )
        ego_is_lapped = float(current_gap_to_leader) >= lap_sec

        # 1. Race classification: lapped backmarkers never count as ahead of a lead-lap car in race position
        ahead_mask = traffic_gaps[None, :] < new_gaps[:, None]  # Shape: (N, M)
        if not ego_is_lapped:
            ahead_mask = ahead_mask & (~is_lapped_mask[None, :])
        rejoin_positions = 1 + np.sum(ahead_mask, axis=1)  # Shape: (N,)

        # 2. Physical on-track clean/dirty air via circular track phase (`gap % base_lap_sec`)
        #    A car physically ahead of our rejoined car has `0 <= (new_phase - traffic_phase) % lap_sec < 1.5s`.
        new_phase = np.mod(new_gaps, lap_sec)  # Shape: (N,)
        traffic_phase = np.mod(traffic_gaps, lap_sec)  # Shape: (M,)
        phase_gap_ahead = np.mod(new_phase[:, None] - traffic_phase[None, :], lap_sec)  # Shape: (N, M)
        min_gap_ahead = np.min(phase_gap_ahead, axis=1)  # Shape: (N,)

        dirty_air_prob = float(np.mean(min_gap_ahead < 1.5))
        clean_air_prob = 1.0 - dirty_air_prob
        if np.any(is_lapped_mask):
            min_blue_flag_gap = np.min(phase_gap_ahead[:, is_lapped_mask], axis=1)
            blue_flag_prob = float(np.mean(min_blue_flag_gap < 1.5))
        else:
            blue_flag_prob = 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "driver": driver,
            "condition": race_condition,
            "green_flag_pit_loss": green_flag_loss,
            "pit_loss_saved_vs_green": saved_vs_green,
            "double_stack_delay_sec": double_stack_delay,
            "median_pit_loss": median_loss,
            "p95_pit_loss": float(np.percentile(pit_losses, 95)),
            "expected_pos": float(np.mean(rejoin_positions)),
            "median_pos": int(np.median(rejoin_positions)),
            "p_clean_air": clean_air_prob,
            "p_dirty_air": dirty_air_prob,
            "p_blue_flag_traffic": blue_flag_prob,
            "elapsed_ms": elapsed_ms,
        }

        if not silent:
            ds_tag = f" | Double-Stack +{double_stack_delay:.2f}s" if double_stack_delay > 0 else ""
            logger.success(
                f"⚡ [MC {n_sims:,} sims in {elapsed_ms:.2f}ms] {driver} ({race_condition}) -> "
                f"Rejoin P{result['median_pos']} (E[P]={result['expected_pos']:.2f}) | "
                f"Loss: {result['median_pit_loss']:.2f}s (P95: {result['p95_pit_loss']:.2f}s) | "
                f"Clean Air: {clean_air_prob * 100:.1f}%{ds_tag}"
            )
        return result

    def calculate_undercut_mc(
        self,
        rival_gap: float,
        new_tyre: str,
        old_tyre: str,
        raw_pace_delta: float | None = None,
        race_condition: str = "GREEN",
        n_sims: int = 10_000,
        old_tyre_deg: float | None = None,
        new_tyre_deg: float | None = None,
        silent: bool = False,
    ) -> dict:
        """
        Vectorized Monte Carlo Undercut Probability Calculator.
        Uses real L1 Tensor tyre degradation delta (`old_tyre_deg - new_tyre_deg`) plus
        intrinsic compound grip offset (`COMPOUND_PACE_OFFSET_SEC[old] - COMPOUND_PACE_OFFSET_SEC[new]`)
        minus the new compound's cold-tyre warmup penalty.
        """
        new_comp = new_tyre.upper()
        old_comp = old_tyre.upper()
        warmup_loss = TYRE_WARMUP_PENALTIES_SEC.get(new_comp, 1.5)
        compound_grip_delta = (
            COMPOUND_PACE_OFFSET_SEC.get(old_comp, 0.25)
            - COMPOUND_PACE_OFFSET_SEC.get(new_comp, 0.25)
        )

        if raw_pace_delta is not None:
            effective_pace_delta = float(raw_pace_delta)
        elif old_tyre_deg is not None and new_tyre_deg is not None:
            effective_pace_delta = (float(old_tyre_deg) - float(new_tyre_deg)) + compound_grip_delta
        else:
            # Fallback when caller omits L1 tensor degradations: mid-stint wear recovery + compound delta
            effective_pace_delta = 1.8 + compound_grip_delta

        pit_loss_ego = self._sample_pit_loss_vectorized(race_condition, n_sims)
        pit_loss_rival = self._sample_pit_loss_vectorized(race_condition, n_sims)

        net_undercut_gain = -pit_loss_ego + pit_loss_rival + (effective_pace_delta - warmup_loss)
        margins = net_undercut_gain - rival_gap

        win_prob = float(np.mean(margins > 0.0))
        median_margin = float(np.median(margins))

        if not silent:
            logger.info(
                f"⚔️ [MC Undercut {old_comp}->{new_comp}] Delta: {effective_pace_delta:+.2f}s (Warmup -{warmup_loss:.1f}s) | "
                f"Win Prob: {win_prob * 100:.1f}% | Median Margin: {median_margin:+.2f}s"
            )
        return {
            "win_probability": win_prob,
            "median_margin_sec": median_margin,
            "raw_pace_delta_sec": round(effective_pace_delta, 3),
            "warmup_loss_sec": round(warmup_loss, 2),
        }

    def calculate_overcut_mc(
        self,
        rival_gap: float,
        rival_new_tyre: str,
        ego_old_tyre: str,
        race_condition: str = "GREEN",
        n_sims: int = 10_000,
        ego_old_tyre_deg: float = 1.2,
        rival_new_tyre_deg: float = 0.2,
        rival_cold_penalty_sec: float = 0.0,
        rival_traffic_delay_sec: float = 0.0,
        overcut_laps: int = 2,
        silent: bool = False,
    ) -> dict:
        """
        Vectorized Monte Carlo Overcut Probability Calculator:
        Evaluates staying out `overcut_laps` in clean air on warm tyres vs a rival who pits first
        onto `rival_new_tyre` (incurring out-lap warmup, cold-track activation, and rejoin traffic).
        """
        r_comp = rival_new_tyre.upper()
        e_comp = ego_old_tyre.upper()
        rival_warmup = TYRE_WARMUP_PENALTIES_SEC.get(r_comp, 1.5)
        compound_delta = (
            COMPOUND_PACE_OFFSET_SEC.get(r_comp, 0.25)
            - COMPOUND_PACE_OFFSET_SEC.get(e_comp, 0.25)
        )
        # Positive overcut_pace_gain means ego on warm clean-air tyres beats rival's cold/traffic out-laps
        n_l = max(1, int(overcut_laps))
        rival_penalty_total = rival_warmup + float(rival_cold_penalty_sec) + float(rival_traffic_delay_sec)
        ego_wear_loss_total = n_l * max(0.0, float(ego_old_tyre_deg) - float(rival_new_tyre_deg))
        clean_air_bonus = 0.35 * n_l
        net_pace_swing = rival_penalty_total + clean_air_bonus + compound_delta * n_l - ego_wear_loss_total

        pit_loss_ego = self._sample_pit_loss_vectorized(race_condition, n_sims)
        pit_loss_rival = self._sample_pit_loss_vectorized(race_condition, n_sims)
        margins = (pit_loss_rival - pit_loss_ego) + net_pace_swing - float(rival_gap)

        win_prob = float(np.mean(margins > 0.0))
        median_margin = float(np.median(margins))
        if not silent:
            logger.info(
                f"🛡️ [MC Overcut {e_comp} vs rival {r_comp}] Swing: {net_pace_swing:+.2f}s | "
                f"Win Prob: {win_prob * 100:.1f}% | Median Margin: {median_margin:+.2f}s"
            )
        return {
            "win_probability": win_prob,
            "median_margin_sec": median_margin,
            "net_overcut_swing_sec": round(net_pace_swing, 2),
        }

    # Legacy single-shot methods preserved for backwards compatibility
    def simulate_rejoin(self, driver: str, current_gap_to_leader: float, traffic: list, race_condition="GREEN"):
        res = self.simulate_rejoin_mc(driver, current_gap_to_leader, traffic, race_condition, n_sims=1)
        return res["median_pos"]


if __name__ == "__main__":
    sim = PitStopSimulator("Bahrain", team_pit_box_position=9, seed=42)

    traffic_snapshot = [
        {"driver": "LEC", "gap_to_leader": 18.0, "is_lapped": False},
        {"driver": "NOR", "gap_to_leader": 21.5, "is_lapped": False},
        {"driver": "SAR", "gap_to_leader": 23.2, "is_lapped": True},
    ]

    print("\n" + "=" * 55 + "\nBENCHMARK: 10,000 Monte Carlo Pit Rejoin Simulations\n" + "=" * 55)
    sim.simulate_rejoin_mc("VER", 0.0, traffic_snapshot, race_condition="GREEN", n_sims=10_000)
    sim.simulate_rejoin_mc("VER", 0.0, traffic_snapshot, race_condition="VSC", n_sims=10_000)
    sim.simulate_rejoin_mc("VER", 0.0, traffic_snapshot, race_condition="SC", n_sims=10_000)

    print("\n" + "=" * 55 + "\nBENCHMARK: 10,000 Monte Carlo Undercut Evaluations\n" + "=" * 55)
    sim.calculate_undercut_mc(rival_gap=1.2, new_tyre="HARD", old_tyre="MEDIUM")
    sim.calculate_undercut_mc(rival_gap=1.2, new_tyre="SOFT", old_tyre="MEDIUM")