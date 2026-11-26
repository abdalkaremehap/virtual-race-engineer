from dataclasses import asdict, dataclass

from src.ml_surrogates.pit_simulator import (
    COMPOUND_PACE_OFFSET_SEC,
    TYRE_WARMUP_PENALTIES_SEC,
)
from src.ml_surrogates.tensor_cache import TyreL1TensorCache


DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
ALL_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
PEAK_GRIP_WINDOWS = {
    "SOFT": 12.0,
    "MEDIUM": 18.0,
    "HARD": 25.0,
    "INTERMEDIATE": 15.0,
    "WET": 20.0,
}


@dataclass
class StrategyOption:
    strategy_id: str
    label: str
    action_type: str  # "STAY_OUT" | "BOX_NOW" | "EXTEND_AND_BOX"
    pit_lap_offset: int  # 0 = box now, >0 = box in N laps, -1 = no stop
    target_compound: str
    expected_remaining_time_sec: float
    pit_loss_sec: float
    tyre_time_sec: float
    cold_penalty_sec: float
    traffic_penalty_sec: float
    double_stack_penalty_sec: float
    weather_penalty_sec: float
    rejoin_pos: int
    p_clean_air: float
    fia_legal: bool = True
    illegal_reason: str = ""
    delta_vs_optimal_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RaceStrategyOptimizer:
    """
    Vectorized Multi-Option F1 Pit-Wall Strategy Optimizer.
    Scores all feasible legal race strategies over the remaining race distance (`laps_remaining`)
    on total expected remaining time delta (tyre deg + thermal cliff + cold activation +
    weather mismatch + pit loss + dirty-air traffic rejoin + teammate double-stack delay),
    enforcing FIA's mandatory 2-compound dry rule.
    """

    def __init__(self, tyre_cache: TyreL1TensorCache):
        self.tyre_cache = tyre_cache

    def _compute_stint_segment_cost(
        self,
        compound: str,
        start_tyre_life: int,
        n_laps: int,
        start_race_lap: int,
        track_temp: float,
        weather_condition: str | None,
        circuit: str | None,
    ) -> tuple[float, float, float]:
        """
        Integrates per-lap tyre cost relative to an idealized 0-deg fresh-SOFT baseline over `n_laps`.
        Returns `(total_tyre_cost_sec, cold_penalty_total_sec, weather_penalty_total_sec)`.
        """
        comp = str(compound).upper()
        n = max(0, int(n_laps))
        if n == 0:
            return 0.0, 0.0, 0.0

        hard_ref = COMPOUND_PACE_OFFSET_SEC.get("HARD", 0.50)
        fresh_bonus = max(0.0, hard_ref - COMPOUND_PACE_OFFSET_SEC.get(comp, 0.25))
        peak_win = PEAK_GRIP_WINDOWS.get(comp, 18.0)
        wx_pen_per_lap = TyreL1TensorCache.get_weather_compound_penalty(comp, weather_condition)

        total_cost = 0.0
        cold_total = 0.0
        wx_total = 0.0
        for step in range(1, n + 1):
            age = int(start_tyre_life) + step
            r_lap = int(start_race_lap) + step
            # Base wear + cliff + circuit abrasion + race_lap fuel load (excluding weather penalty so we track it cleanly)
            deg_only = self.tyre_cache.lookup_degradation(
                comp,
                track_temp,
                age,
                race_lap=r_lap,
                weather_condition=None,
                circuit=circuit,
            )
            cold_k = TyreL1TensorCache.get_cold_track_activation_penalty(
                comp,
                track_temp,
                stint_lap=age,
                weather_condition=weather_condition,
            )
            # Peak fresh grip bonus reduces lap time during opening laps of a fresh stint
            bonus_k = fresh_bonus * max(0.0, 1.0 - age / peak_win)
            total_cost += deg_only + cold_k + wx_pen_per_lap - bonus_k
            cold_total += cold_k
            wx_total += wx_pen_per_lap

        return round(total_cost, 2), round(cold_total, 2), round(wx_total, 2)

    @staticmethod
    def _estimate_traffic_penalty(
        circuit: str | None,
        p_clean_air: float,
        positions_dropped: int,
        race_condition: str,
        delay_laps: int = 0,
    ) -> float:
        dirty_prob = max(0.0, min(1.0, 1.0 - float(p_clean_air)))
        ov_diff = TyreL1TensorCache.get_circuit_overtaking_difficulty(circuit)
        pos_drop = min(4, max(0, int(positions_dropped)))
        cond_scale = 0.25 if str(race_condition).upper() in ("SC", "VSC") else 1.0
        # Extending stint spreads pit-window traffic out over subsequent laps
        delay_relief = max(0.30, 1.0 - 0.085 * max(0, int(delay_laps)))
        raw = (dirty_prob * 2.0 + pos_drop * dirty_prob * 1.05) * ov_diff * cond_scale * delay_relief
        return round(min(12.0, raw), 2)

    def evaluate_all_strategies(
        self,
        *,
        ego_compound: str,
        ego_tyre_life: int,
        ego_pos: int,
        track_temp: float,
        current_lap: int,
        total_laps: int,
        pit_loss_now_sec: float,
        green_flag_pit_loss_sec: float,
        rejoin_pos_now: int,
        p_clean_air_now: float,
        circuit: str | None = None,
        weather_condition: str | None = "Dry (Clear)",
        race_condition: str = "GREEN",
        requested_target_tyre: str | None = None,
        used_dry_compounds: tuple[str, ...] | list[str] = (),
        double_stack_delay_sec: float = 0.0,
        drying_track_hint: bool = False,
    ) -> dict:
        """
        Scores STAY_OUT, BOX_NOW (across all compounds), and EXTEND_AND_BOX (+1L if double-stack,
        +4L, +8L, +12L -> optimal final compound), returning ranked options + Optimal vs Requested delta.
        """
        ego_comp = str(ego_compound).upper()
        ego_age = max(0, int(ego_tyre_life))
        cur_lap = max(1, int(current_lap))
        tot_laps = max(cur_lap, int(total_laps))
        rem_laps = max(0, tot_laps - cur_lap)
        regime = TyreL1TensorCache.normalize_weather_regime(weather_condition)
        cond_upper = str(race_condition).upper()

        used_dry_set = {str(c).upper() for c in used_dry_compounds if str(c).upper() in DRY_COMPOUNDS}
        if ego_comp in DRY_COMPOUNDS and used_dry_compounds:
            used_dry_set.add(ego_comp)
        enforce_two_compound = bool(regime == "DRY" and len(used_dry_compounds) > 0)

        pos_dropped_now = max(0, int(rejoin_pos_now) - int(ego_pos))
        traffic_now = (
            self._estimate_traffic_penalty(circuit, p_clean_air_now, pos_dropped_now, cond_upper, delay_laps=0)
            if rem_laps > 0
            else 0.0
        )

        options: list[StrategyOption] = []

        # 1. STAY_OUT to checkered flag
        stay_tyre, stay_cold, stay_wx = self._compute_stint_segment_cost(
            ego_comp, ego_age, rem_laps, cur_lap, track_temp, weather_condition, circuit
        )
        stay_legal = True
        stay_illegal_reason = ""
        if enforce_two_compound and len(used_dry_set) < 2 and rem_laps >= 1:
            stay_legal = False
            stay_illegal_reason = (
                f"FIA 2-Compound Rule violation: only used {sorted(used_dry_set)} in dry race"
            )
        options.append(
            StrategyOption(
                strategy_id="STAY_OUT",
                label=f"STAY OUT on {ego_comp}s to Flag",
                action_type="STAY_OUT",
                pit_lap_offset=-1,
                target_compound=ego_comp,
                expected_remaining_time_sec=stay_tyre,
                pit_loss_sec=0.0,
                tyre_time_sec=stay_tyre,
                cold_penalty_sec=stay_cold,
                traffic_penalty_sec=0.0,
                double_stack_penalty_sec=0.0,
                weather_penalty_sec=stay_wx,
                rejoin_pos=int(ego_pos),
                p_clean_air=1.0 if int(ego_pos) == 1 else float(p_clean_air_now),
                fia_legal=stay_legal,
                illegal_reason=stay_illegal_reason,
            )
        )

        # Determine which compounds to score for BOX_NOW
        candidate_compounds = list(DRY_COMPOUNDS if regime == "DRY" else ALL_COMPOUNDS)
        req_comp = str(requested_target_tyre).upper() if requested_target_tyre else None
        if req_comp and req_comp not in candidate_compounds and req_comp in ALL_COMPOUNDS:
            candidate_compounds.append(req_comp)

        # 2. BOX_NOW -> each candidate compound
        # Under full Safety Car, the field bunches to ~0s gaps before the restart: if pitting drops
        # ZERO positions (rejoin_pos <= ego_pos in clean air), it is a FREE SC STOP (0.0s effective cost).
        is_free_sc_stop = bool(
            cond_upper == "SC"
            and int(rejoin_pos_now) <= int(ego_pos)
            and float(p_clean_air_now) >= 0.65
            and double_stack_delay_sec < 1.5
        )
        effective_pit_loss_now = 0.0 if is_free_sc_stop else float(pit_loss_now_sec)

        if rem_laps > 0:
            for cand in candidate_compounds:
                warm_pen = TYRE_WARMUP_PENALTIES_SEC.get(cand, 1.2)
                tyre_c, cold_c, wx_c = self._compute_stint_segment_cost(
                    cand, 0, rem_laps, cur_lap, track_temp, weather_condition, circuit
                )
                extra_double_stop = 0.0
                if wx_c > 0.0:
                    # Fitting wrong weather rubber forces a second pit stop later
                    extra_double_stop = round(float(green_flag_pit_loss_sec) * 0.85, 2)
                elif drying_track_hint and ego_comp == "WET" and cand == "INTERMEDIATE":
                    extra_double_stop = TyreL1TensorCache.estimate_drying_crossover_penalty(
                        cand, float(green_flag_pit_loss_sec), drying_laps_to_slicks=3
                    )

                total_t = round(
                    effective_pit_loss_now + warm_pen + traffic_now + tyre_c + extra_double_stop,
                    2,
                )
                cand_legal = True
                cand_reason = ""
                if enforce_two_compound and cand in DRY_COMPOUNDS and len(used_dry_set | {cand}) < 2:
                    cand_legal = False
                    cand_reason = f"FIA 2-Compound Rule: {cand} does not add a 2nd dry compound"
                elif wx_c > 0.0:
                    cand_legal = False
                    cand_reason = f"Wrong weather tyre for {weather_condition}: {cand} loses +{wx_c / max(1, rem_laps):.1f}s/L"

                options.append(
                    StrategyOption(
                        strategy_id=f"BOX_NOW_{cand}",
                        label=f"BOX NOW -> {cand}",
                        action_type="BOX_NOW",
                        pit_lap_offset=0,
                        target_compound=cand,
                        expected_remaining_time_sec=total_t,
                        pit_loss_sec=round(float(pit_loss_now_sec), 2),
                        tyre_time_sec=round(tyre_c + warm_pen, 2),
                        cold_penalty_sec=cold_c,
                        traffic_penalty_sec=traffic_now,
                        double_stack_penalty_sec=round(float(double_stack_delay_sec), 2),
                        weather_penalty_sec=wx_c,
                        rejoin_pos=int(rejoin_pos_now),
                        p_clean_air=float(p_clean_air_now),
                        fia_legal=cand_legal,
                        illegal_reason=cand_reason,
                    )
                )

        # 3. EXTEND_AND_BOX (+delay laps -> best legal compound)
        delay_candidates: list[int] = []
        if double_stack_delay_sec >= 2.0 and rem_laps >= 4:
            delay_candidates.append(1)  # Pit 1 lap later to avoid teammate double-stack queue!
        if rem_laps >= 12:
            for d in (4, 8, 12):
                if rem_laps - d >= 5:
                    delay_candidates.append(d)

        extend_compounds = DRY_COMPOUNDS if regime == "DRY" else ("INTERMEDIATE", "WET")
        for delay in delay_candidates:
            # Running current tyres for `delay` more laps
            pre_tyre, pre_cold, pre_wx = self._compute_stint_segment_cost(
                ego_comp, ego_age, delay, cur_lap, track_temp, weather_condition, circuit
            )
            # Pit 1 lap later during SC still gets SC discount; longer extensions pay green-flag pit loss
            pit_loss_delayed = (
                max(0.0, float(pit_loss_now_sec) - float(double_stack_delay_sec))
                if (delay == 1 and cond_upper in ("SC", "VSC"))
                else float(green_flag_pit_loss_sec)
            )
            traffic_delayed = self._estimate_traffic_penalty(
                circuit,
                p_clean_air_now,
                pos_dropped_now,
                "SC" if (delay == 1 and cond_upper == "SC") else "GREEN",
                delay_laps=delay,
            )
            rem_after = rem_laps - delay

            best_ext_opt: StrategyOption | None = None
            for cand in extend_compounds:
                warm_pen = TYRE_WARMUP_PENALTIES_SEC.get(cand, 1.2)
                post_tyre, post_cold, post_wx = self._compute_stint_segment_cost(
                    cand, 0, rem_after, cur_lap + delay, track_temp, weather_condition, circuit
                )
                ext_legal = True
                ext_reason = ""
                if enforce_two_compound and cand in DRY_COMPOUNDS and len(used_dry_set | {cand}) < 2:
                    ext_legal = False
                    ext_reason = f"FIA 2-Compound Rule: {cand} does not add a 2nd dry compound"
                elif post_wx > 0.0:
                    ext_legal = False
                    ext_reason = f"Wrong weather tyre {cand}"

                total_ext = round(
                    pre_tyre + pit_loss_delayed + warm_pen + traffic_delayed + post_tyre,
                    2,
                )
                opt = StrategyOption(
                    strategy_id=f"EXTEND_{delay}L_{cand}",
                    label=f"EXTEND {ego_comp} +{delay}L -> {cand}",
                    action_type="EXTEND_AND_BOX",
                    pit_lap_offset=delay,
                    target_compound=cand,
                    expected_remaining_time_sec=total_ext,
                    pit_loss_sec=round(pit_loss_delayed, 2),
                    tyre_time_sec=round(pre_tyre + post_tyre + warm_pen, 2),
                    cold_penalty_sec=round(pre_cold + post_cold, 2),
                    traffic_penalty_sec=traffic_delayed,
                    double_stack_penalty_sec=0.0,
                    weather_penalty_sec=round(pre_wx + post_wx, 2),
                    rejoin_pos=max(int(ego_pos), int(rejoin_pos_now) - (1 if delay >= 6 else 0)),
                    p_clean_air=min(1.0, float(p_clean_air_now) + 0.08 * delay),
                    fia_legal=ext_legal,
                    illegal_reason=ext_reason,
                )
                if ext_legal and (best_ext_opt is None or opt.expected_remaining_time_sec < best_ext_opt.expected_remaining_time_sec):
                    best_ext_opt = opt

            if best_ext_opt is not None:
                options.append(best_ext_opt)

        # Rank: all FIA-legal strategies first by lowest expected remaining time, then illegal options
        options.sort(key=lambda o: (not o.fia_legal, o.expected_remaining_time_sec))
        optimal = options[0]

        # Pit-stop risk hysteresis: under GREEN flag, if STAY_OUT is FIA-legal and BOX_NOW only beats it
        # by < 1.5s, prefer STAY_OUT to avoid gambling a 20s+ pit stop for a razor-thin margin.
        stay_opt = next((o for o in options if o.strategy_id == "STAY_OUT"), None)
        if (
            cond_upper == "GREEN"
            and stay_opt is not None
            and stay_opt.fia_legal
            and optimal.action_type == "BOX_NOW"
            and (stay_opt.expected_remaining_time_sec - optimal.expected_remaining_time_sec) < 1.5
        ):
            optimal = stay_opt
            options.remove(stay_opt)
            options.insert(0, stay_opt)

        for o in options:
            o.delta_vs_optimal_sec = round(o.expected_remaining_time_sec - optimal.expected_remaining_time_sec, 2)

        requested_opt = None
        if req_comp:
            requested_opt = next((o for o in options if o.strategy_id == f"BOX_NOW_{req_comp}"), None)
        if requested_opt is None:
            requested_opt = optimal

        return {
            "optimal": optimal.to_dict(),
            "requested": requested_opt.to_dict(),
            "top_options": [o.to_dict() for o in options[:5]],
            "all_options_count": len(options),
            "optimal_beats_requested_by_sec": round(
                max(0.0, requested_opt.expected_remaining_time_sec - optimal.expected_remaining_time_sec),
                2,
            ),
            "fia_two_compound_enforced": enforce_two_compound,
            "used_dry_compounds": sorted(used_dry_set),
        }
