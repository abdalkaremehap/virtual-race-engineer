"""
20-Scenario Historical F1 Strategy Red-Team Benchmark Suite.
Adversarially grades the Virtual Race Engineer, Multi-Option Strategy Optimizer,
PitStopSimulator, L1 Tyre Tensor Cache, and Telemetry Physics across real 2021-2024
F1 strategy blunders and core motorsport edge cases.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.race_engineer import AutonomousRaceEngineer
from src.agent.tools import (
    RaceContext,
    RaceEngineerToolkit,
    build_live_data_from_standings_df,
)
from src.data_ingestion.telemetry_daemon import (
    F1_2024_CALENDAR,
    generate_driver_micro_telemetry,
    infer_lap_weather_condition,
    sync_gp_lap_to_live_table,
)
from src.ml_surrogates.pit_simulator import PitStopSimulator
from src.ml_surrogates.strategy_optimizer import RaceStrategyOptimizer
from src.ml_surrogates.tensor_cache import TyreL1TensorCache


def _make_grid_df(
    lap: int,
    rows: list[tuple[str, int, str, int, float, float]],
    track_status: str = "GREEN",
    track_temp: float = 32.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lap": lap,
                "driver": drv,
                "position": pos,
                "compound": comp,
                "tyre_life": age,
                "gap_to_leader": gap,
                "lap_time_sec": lt,
                "track_status": track_status,
                "track_temp": track_temp,
            }
            for drv, pos, comp, age, gap, lt in rows
        ]
    )


def _ask(
    driver: str,
    circuit: str,
    total_laps: int,
    df: pd.DataFrame,
    query: str,
    weather: str = "Dry (Clear)",
    used_dry: tuple[str, ...] = (),
    tm_gap: float | None = None,
    tm_pitting: bool = False,
) -> dict:
    eng = AutonomousRaceEngineer(ego_driver=driver)
    ctx = RaceContext(
        driver=driver,
        circuit=circuit,
        total_race_laps=total_laps,
        team_pit_box_position=2,
        weather_condition=weather,
        used_dry_compounds=used_dry,
        teammate_gap_ahead_sec=tm_gap,
        teammate_pitting=tm_pitting,
        live_data=build_live_data_from_standings_df(
            df,
            circuit=circuit,
            total_race_laps=total_laps,
            weather_condition=weather,
            used_dry_compounds=used_dry,
            teammate_gap_ahead_sec=tm_gap,
            teammate_pitting=tm_pitting,
        ),
    )
    return eng.ask_engineer(query, context=ctx)


# 1. 2022 British GP (Silverstone L39, SC) — Ferrari left LEC out on 14L Hards
def test_01_2022_silverstone_l39_sc_box_for_softs():
    df = _make_grid_df(
        39,
        [
            ("SAI", 1, "HARD", 19, 0.0, 98.36),
            ("LEC", 2, "HARD", 14, 0.29, 102.89),
            ("HAM", 3, "HARD", 6, 1.40, 103.10),
            ("PER", 4, "MEDIUM", 35, 34.54, 104.20),
            ("ALO", 5, "HARD", 6, 58.28, 104.50),
        ],
        track_status="SC",
        track_temp=30.0,
    )
    res = _ask("LEC", "Great Britain", 52, df, "Safety Car is out! Can I catch SAI or should we box for Softs?")
    assert res["guardrail_passed"] is True
    assert "BOX BOX NOW" in res["engineer_radio_reply"]
    assert "SC ACTIVE" in res["battle_data"]["tactical_verdict"]
    opt = res["pit_sim_data"]["strategy_ranking"]["optimal"]
    assert opt["action_type"] == "BOX_NOW" and opt["target_compound"] == "SOFT"


# 2. 2022 Hungarian GP (Hungaroring L39, 26.7C) — Ferrari panicked onto cold Hards into P5 traffic
def test_02_2022_hungary_l39_reject_cold_hard_and_extend():
    df = _make_grid_df(
        39,
        [
            ("LEC", 1, "MEDIUM", 18, 0.0, 86.32),
            ("SAI", 2, "MEDIUM", 22, 4.98, 86.45),
            ("RUS", 3, "MEDIUM", 21, 5.70, 86.50),
            ("HAM", 4, "MEDIUM", 20, 10.40, 86.38),
            ("PER", 5, "MEDIUM", 21, 20.59, 86.80),
            ("VER", 6, "MEDIUM", 1, 22.88, 99.45),
        ],
        track_status="GREEN",
        track_temp=26.7,
    )
    res = _ask(
        "LEC",
        "Hungary",
        70,
        df,
        "do i pit onto HARD?",
        weather="Overcast (Cool)",
        used_dry=("MEDIUM",),
    )
    assert res["guardrail_passed"] is True
    assert "DO NOT BOX FOR HARDs" in res["engineer_radio_reply"]
    assert "STAY OUT" in res["engineer_radio_reply"]
    opt = res["pit_sim_data"]["strategy_ranking"]["optimal"]
    assert opt["target_compound"] != "HARD"
    assert res["pit_sim_data"]["strategy_ranking"]["optimal_beats_requested_by_sec"] > 5.0


# 3. 2022 Monaco GP (L18 Dry/Drying) — Must skip Intermediates and go straight to Dry Slicks
def test_03_2022_monaco_l18_skip_intermediates_on_dry_track():
    df = _make_grid_df(
        18,
        [
            ("SAI", 1, "WET", 18, 0.0, 105.0),
            ("PER", 2, "INTERMEDIATE", 2, 11.65, 85.2),
            ("LEC", 3, "WET", 18, 12.58, 106.0),
            ("VER", 4, "WET", 18, 20.96, 106.5),
            ("RUS", 5, "WET", 18, 23.76, 106.8),
            ("ALO", 6, "WET", 18, 25.60, 107.1),
            ("NOR", 7, "INTERMEDIATE", 2, 33.85, 86.0),
        ],
        track_status="GREEN",
        track_temp=29.1,
    )
    res = _ask("LEC", "Monaco", 64, df, "do i pit for Intermediates?", weather="Dry (Clear)")
    assert res["guardrail_passed"] is True
    assert "SKIP INTERMEDIATEs" in res["engineer_radio_reply"]
    assert res["pit_sim_data"]["stint_breakeven"]["wrong_weather_target"] is True


# 4. 2022 Monaco GP (L21 Double-Stack) — Teammate <3.5s ahead pitting injects queuing delay
def test_04_2022_monaco_l21_teammate_double_stack_delay():
    sim = PitStopSimulator("Monaco", team_pit_box_position=2, seed=42)
    traffic = [{"driver": "SAI", "gap_to_leader": 0.0, "is_lapped": False}]
    clean_stop = sim.simulate_rejoin_mc("LEC", 2.8, traffic, race_condition="GREEN", n_sims=10_000, silent=True)
    ds_stop = sim.simulate_rejoin_mc(
        "LEC",
        2.8,
        traffic,
        race_condition="GREEN",
        n_sims=10_000,
        silent=True,
        teammate_gap_ahead_sec=2.8,
        teammate_pitting=True,
    )
    assert clean_stop["double_stack_delay_sec"] == 0.0
    assert ds_stop["double_stack_delay_sec"] >= 2.0
    assert ds_stop["median_pit_loss"] > clean_stop["median_pit_loss"] + 2.0


# 5. 2022 Belgian GP (Spa L42/44) — Never pit from P5 with 2 laps left
def test_05_2022_spa_l42_no_late_fastest_lap_stop():
    df = _make_grid_df(
        42,
        [
            ("VER", 1, "MEDIUM", 12, 0.0, 109.5),
            ("PER", 2, "HARD", 15, 17.0, 110.2),
            ("SAI", 3, "HARD", 17, 27.0, 110.4),
            ("RUS", 4, "HARD", 13, 30.0, 110.6),
            ("LEC", 5, "MEDIUM", 17, 57.18, 111.5),
            ("ALO", 6, "HARD", 17, 71.55, 112.1),
        ],
        track_status="GREEN",
        track_temp=36.2,
    )
    res = _ask("LEC", "Belgium", 44, df, "2 laps left - should we box for Softs ahead of ALO?")
    assert res["guardrail_passed"] is True
    assert "STAY OUT & HOLD POSITION" in res["engineer_radio_reply"]
    assert res["pit_sim_data"]["strategy_ranking"]["optimal"]["action_type"] == "STAY_OUT"


# 6. 2024 British GP (Silverstone L19, Dry) — Reject premature Intermediates on dry track
def test_06_2024_silverstone_l19_reject_premature_inters():
    df = _make_grid_df(
        19,
        [
            ("RUS", 1, "MEDIUM", 19, 0.0, 90.2),
            ("HAM", 2, "MEDIUM", 19, 2.1, 90.3),
            ("LEC", 3, "MEDIUM", 19, 6.9, 90.8),
            ("NOR", 4, "MEDIUM", 19, 8.2, 90.5),
            ("VER", 5, "MEDIUM", 19, 10.1, 90.6),
        ],
        track_status="GREEN",
        track_temp=27.1,
    )
    res = _ask("LEC", "Great Britain", 52, df, "Should we box for Intermediates right now?", weather="Dry (Clear)")
    assert res["guardrail_passed"] is True
    assert "STAY OUT" in res["engineer_radio_reply"]
    assert "DO NOT BOX FOR INTERMEDIATEs" in res["engineer_radio_reply"]


# 7. 2021 Abu Dhabi GP (L53/58, SC) — Fresh Softs under SC beat 38-lap old Hards with 5L left
def test_07_2021_abu_dhabi_l53_sc_fresh_softs_crossover():
    df = _make_grid_df(
        53,
        [
            ("HAM", 1, "HARD", 39, 0.0, 88.5),
            ("VER", 2, "HARD", 17, 11.2, 88.1),
            ("PER", 3, "HARD", 25, 29.0, 88.9),
            ("SAI", 4, "HARD", 34, 42.0, 89.2),
        ],
        track_status="SC",
        track_temp=28.0,
    )
    res = _ask("HAM", "Abu Dhabi", 58, df, "Safety Car! Should we box for Softs right now?")
    assert res["guardrail_passed"] is True
    assert res["pit_sim_data"]["stint_breakeven"]["recoverable_tyre_gain_sec"] > 10.0
    assert "BOX BOX NOW" in res["engineer_radio_reply"]


# 8. 2021 Russian GP (Sochi L48/53, Heavy Rain) — Must box immediately from slicks in heavy rain
def test_08_2021_sochi_l48_heavy_rain_mandatory_crossover():
    df = _make_grid_df(
        48,
        [
            ("NOR", 1, "HARD", 22, 0.0, 108.5),
            ("HAM", 2, "INTERMEDIATE", 1, 2.0, 96.0),
            ("VER", 3, "INTERMEDIATE", 1, 18.0, 96.5),
        ],
        track_status="GREEN",
        track_temp=22.0,
    )
    res = _ask("NOR", "Russia", 53, df, "Heavy rain! Should we box right now?", weather="Heavy Rain (Wet)")
    assert res["guardrail_passed"] is True
    assert "BOX THIS LAP FOR WET" in res["engineer_radio_reply"]
    assert "WEATHER CROSSOVER" in res["engineer_radio_reply"]


# 9. 2024 Bahrain GP (L50/57) — RUS P5 vs LEC P4: Hold position with 7 laps left
def test_09_2024_bahrain_l50_rus_hold_position():
    RaceEngineerToolkit.get().set_circuit("Bahrain", 57, 1)
    df = sync_gp_lap_to_live_table("Bahrain", "Race", lap=50, season=2024)
    res = _ask("RUS", "Bahrain", 57, df, "if i can pass LEC")
    assert res["guardrail_passed"] is True
    assert "STAY OUT & HOLD POSITION" in res["engineer_radio_reply"]
    assert "BOX THIS LAP" not in res["engineer_radio_reply"]


# 10. FIA Mandatory 2-Compound Dry Rule Enforcement
def test_10_fia_mandatory_two_compound_dry_rule():
    cache = TyreL1TensorCache(total_race_laps=57)
    opt = RaceStrategyOptimizer(cache)
    res = opt.evaluate_all_strategies(
        ego_compound="MEDIUM",
        ego_tyre_life=25,
        ego_pos=1,
        track_temp=32.0,
        current_lap=54,
        total_laps=57,
        pit_loss_now_sec=23.0,
        green_flag_pit_loss_sec=23.0,
        rejoin_pos_now=5,
        p_clean_air_now=0.85,
        circuit="Bahrain",
        weather_condition="Dry (Clear)",
        used_dry_compounds=("MEDIUM",),
    )
    # Even with only 3 laps left, STAY_OUT is FIA-illegal because only 1 dry compound has been used!
    assert res["optimal"]["fia_legal"] is True
    assert res["optimal"]["action_type"] == "BOX_NOW"
    assert res["optimal"]["target_compound"] in ("SOFT", "HARD")


# 11. Tyre Cliff Crossover on 28-lap SOFTs
def test_11_tyre_cliff_crossover_triggers_box():
    df = _make_grid_df(
        28,
        [
            ("VER", 1, "SOFT", 28, 0.0, 97.5),
            ("LEC", 2, "HARD", 10, 18.0, 94.1),
            ("NOR", 3, "HARD", 11, 26.0, 94.3),
        ],
        track_status="GREEN",
        track_temp=34.0,
    )
    res = _ask("VER", "Bahrain", 57, df, "My softs are falling off a cliff, should we box for Hards?")
    assert res["guardrail_passed"] is True
    assert "BOX THIS LAP" in res["engineer_radio_reply"]


# 12. Clean-Air Overcut vs Cold-Track Undercut
def test_12_overcut_beats_undercut_in_cold_traffic():
    sim = PitStopSimulator("Hungary", team_pit_box_position=2, seed=42)
    oc = sim.calculate_overcut_mc(
        rival_gap=0.8,
        rival_new_tyre="HARD",
        ego_old_tyre="MEDIUM",
        race_condition="GREEN",
        n_sims=10_000,
        ego_old_tyre_deg=1.3,
        rival_new_tyre_deg=0.2,
        rival_cold_penalty_sec=2.5,
        rival_traffic_delay_sec=2.0,
        overcut_laps=2,
        silent=True,
    )
    assert oc["win_probability"] > 0.85
    assert oc["median_margin_sec"] > 2.0


# 13. Lapped Blue-Flag Traffic vs Lead-Lap Race Classification
def test_13_lapped_blue_flag_traffic_does_not_demote_position():
    sim = PitStopSimulator("Bahrain", team_pit_box_position=1, seed=42)
    traffic = [
        {"driver": "PER", "gap_to_leader": 30.0, "is_lapped": False},
        {"driver": "SAR", "gap_to_leader": 116.5, "is_lapped": True},  # 116.5 % 94.1 = 22.4s phase!
    ]
    res = sim.simulate_rejoin_mc("VER", 0.0, traffic, race_condition="GREEN", n_sims=10_000, silent=True, base_lap_sec=94.1)
    assert res["median_pos"] == 1
    assert res["p_blue_flag_traffic"] > 0.50


# 14. Final Lap Checkered Flag (0 laps remaining) — Never box even under SC
def test_14_final_lap_zero_remaining_never_box():
    df = _make_grid_df(
        57,
        [("VER", 1, "SOFT", 30, 0.0, 96.0), ("PER", 2, "HARD", 20, 5.0, 95.0)],
        track_status="SC",
        track_temp=32.0,
    )
    res = _ask("VER", "Bahrain", 57, df, "Safety Car on the final lap, should we box?")
    assert res["guardrail_passed"] is True
    assert "STAY OUT & HOLD POSITION" in res["engineer_radio_reply"]
    assert res["pit_sim_data"]["strategy_ranking"]["optimal"]["action_type"] == "STAY_OUT"


# 15. Multi-Compound Warmup vs Durability Ordering
def test_15_compound_warmup_and_durability_ordering():
    cache = TyreL1TensorCache(total_race_laps=57)
    soft_30 = cache.lookup_degradation("SOFT", 34.0, 30, circuit="Bahrain")
    med_30 = cache.lookup_degradation("MEDIUM", 34.0, 30, circuit="Bahrain")
    hard_30 = cache.lookup_degradation("HARD", 34.0, 30, circuit="Bahrain")
    assert soft_30 > med_30 >= hard_30


# 16. Zero-Contradiction Guardrail & Pit-In/Out Lap Outlier Filter (2021 Spain L42 + 2022 Silverstone L39)
def test_16_zero_contradiction_sc_rival_and_pit_call():
    df = _make_grid_df(
        39,
        [
            ("SAI", 1, "HARD", 19, 0.0, 98.36),
            ("LEC", 2, "HARD", 14, 0.29, 102.89),
            ("HAM", 3, "HARD", 6, 1.40, 103.10),
        ],
        track_status="SC",
        track_temp=30.0,
    )
    res = _ask("LEC", "Great Britain", 52, df, "Can I catch SAI or should we box for Softs?")
    reply = res["engineer_radio_reply"]
    assert "VERDICT: **NO" not in reply
    assert "BOX BOX NOW" in reply

    # 2021 Spain Lap 42: HAM slows to 85.686s entering pit lane (vs VER 82.118s) -> must NOT treat pit-in as slow pace!
    df_spain = _make_grid_df(
        42,
        [
            ("VER", 1, "MEDIUM", 18, 0.0, 82.118),
            ("HAM", 2, "MEDIUM", 14, 4.48, 85.686),
            ("BOT", 3, "MEDIUM", 19, 9.56, 82.80),
        ],
        track_status="GREEN",
        track_temp=32.7,
    )
    res_spain = _ask("VER", "Spain", 66, df_spain, "Can HAM catch me or should we box for Softs?")
    assert res_spain["guardrail_passed"] is True
    assert "RIVAL PIT LAP DETECTED" in res_spain["battle_data"]["tactical_verdict"]
    # Strategy Optimizer & Primary Call must never contradict each other (no STAY OUT + OPTIMAL: BOX NOW)
    opt_action = res_spain["pit_sim_data"]["strategy_ranking"]["optimal"]["action_type"]
    if opt_action == "STAY_OUT":
        assert "STAY OUT" in res_spain["engineer_radio_reply"]
    elif opt_action == "BOX_NOW":
        assert "BOX" in res_spain["engineer_radio_reply"]


# 17. Pedal Physics Sanity Invariant Across All 24 Circuits (No Simultaneous Brake + Throttle or DRS under Braking)
def test_17_pedal_physics_invariant_all_24_circuits():
    for cal in F1_2024_CALENDAR:
        tel = generate_driver_micro_telemetry("VER", cal["circuit"], lap=15, compound="SOFT", tyre_life=10)
        braking = tel["Brake"].to_numpy() > 0.0
        thr = tel["Throttle"].to_numpy()
        drs = tel["DRS"].to_numpy()
        assert np.all(thr[braking] == 0.0), f"Simultaneous brake & throttle detected at {cal['circuit']}!"
        assert np.all(drs[braking] == 0), f"DRS open while braking at {cal['circuit']}!"


# 18. Wet Weather Auto-Inference From Real Active Compounds
def test_18_wet_weather_auto_inference():
    df_wet = _make_grid_df(18, [("SAI", 1, "WET", 18, 0.0, 105.0), ("LEC", 2, "WET", 18, 1.0, 105.5), ("PER", 3, "INTERMEDIATE", 2, 11.0, 86.0)], track_temp=24.0)
    df_damp = _make_grid_df(25, [("NOR", 1, "INTERMEDIATE", 5, 0.0, 92.0), ("VER", 2, "INTERMEDIATE", 5, 2.0, 92.5), ("HAM", 3, "MEDIUM", 10, 8.0, 96.0)], track_temp=24.0)
    df_cool = _make_grid_df(39, [("LEC", 1, "MEDIUM", 18, 0.0, 86.3), ("SAI", 2, "MEDIUM", 22, 4.9, 86.5)], track_temp=26.7)
    df_dry = _make_grid_df(17, [("VER", 1, "SOFT", 17, 0.0, 94.1), ("PER", 2, "SOFT", 17, 3.0, 94.5)], track_temp=32.5)
    assert infer_lap_weather_condition(df_wet) == "Heavy Rain (Wet)"
    assert infer_lap_weather_condition(df_damp) == "Light Drizzle (Damp)"
    assert infer_lap_weather_condition(df_cool) == "Overcast (Cool)"
    assert infer_lap_weather_condition(df_dry) == "Dry (Clear)"


# 19. Circuit-Specific Pit Loss Accuracy Through Optimizer
def test_19_circuit_specific_pit_losses():
    for cname, expected_green in [("Singapore", 28.0), ("Emilia Romagna", 28.0), ("Canada", 18.5), ("Bahrain", 23.0)]:
        sim = PitStopSimulator(cname, team_pit_box_position=1, seed=42)
        mc = sim.simulate_rejoin_mc("VER", 0.0, [], race_condition="GREEN", n_sims=5_000, silent=True)
        assert mc["green_flag_pit_loss"] == expected_green


# 20. 2-Hop FIA Legal Citation Integrity (Online Qdrant or Clean Offline Fallback)
def test_20_fia_legal_citation_integrity():
    df = _make_grid_df(39, [("LEC", 1, "HARD", 14, 0.0, 102.0), ("SAI", 2, "HARD", 19, 1.0, 102.5)], track_status="SC", track_temp=30.0)
    res = _ask("LEC", "Great Britain", 52, df, "What is the exact FIA penalty for overtaking under the Safety Car?")
    assert res["guardrail_passed"] is True
    assert "FIA LEGAL" in res["engineer_radio_reply"]
    ld = res["legal_data"]
    assert ld.get("offline") is True or len(ld.get("hop1_primary_rules", [])) >= 1


# 21. Human F1 Radio Persona Alignment + Multi-Hop RAG Thread-Safe LRU Cache
def test_21_human_radio_call_clarification_and_rag_cache():
    from src.rag_engine.multi_hop import MultiHopFIARulebook

    # A) Silverstone 2022 SC -> Natural "Charles" + "Box" call aligned with engineer_radio_reply
    df_sc = _make_grid_df(
        39,
        [("LEC", 1, "HARD", 14, 0.0, 102.0), ("SAI", 2, "HARD", 19, 1.0, 102.5), ("HAM", 3, "HARD", 6, 3.0, 103.0)],
        track_status="SC",
        track_temp=30.0,
    )
    res_sc = _ask("LEC", "Great Britain", 52, df_sc, "Safety car is out! Should we box for Softs?")
    human_sc = res_sc["human_radio_call"]
    assert "Charles" in human_sc
    assert "box" in human_sc.lower()
    assert len(human_sc) < 480

    # B) Hungary 2022 Cold HARD trap -> Natural "Charles" + "Stay out" + "negative" aligned with engineer_radio_reply
    df_hun = _make_grid_df(
        39,
        [
            ("LEC", 1, "MEDIUM", 18, 0.0, 82.8),
            ("VER", 2, "MEDIUM", 1, 7.2, 81.9),
            ("RUS", 3, "MEDIUM", 2, 11.4, 82.4),
            ("SAI", 4, "MEDIUM", 22, 14.2, 83.1),
            ("HAM", 5, "MEDIUM", 20, 18.0, 83.0),
            ("PER", 6, "MEDIUM", 3, 21.5, 82.6),
            ("NOR", 7, "MEDIUM", 10, 24.0, 83.4),
        ],
        track_temp=26.7,
    )
    res_hun = _ask("LEC", "Hungary", 70, df_hun, "Should we box for Hards right now?", weather="Overcast (Cool)")
    human_hun = res_hun["human_radio_call"]
    assert "Charles" in human_hun
    assert "stay out" in human_hun.lower()
    assert len(human_hun) < 480

    # C) Monaco 2022 Dry track asking for Intermediates -> Natural "skip" call
    df_mon = _make_grid_df(
        18,
        [("LEC", 1, "WET", 18, 0.0, 99.0), ("SAI", 2, "WET", 18, 4.5, 99.5), ("PER", 3, "INTERMEDIATE", 2, 9.0, 88.0)],
        track_temp=28.0,
    )
    res_mon = _ask("LEC", "Monaco", 64, df_mon, "Should we box for Intermediates right now?", weather="Dry (Clear)")
    human_mon = res_mon["human_radio_call"]
    assert "Charles" in human_mon
    assert "skip" in human_mon.lower() or "negative" in human_mon.lower()

    # D) Multi-Hop RAG LRU Cache: repeat query returns instant defensive copy
    class _StubRulebook:
        def __init__(self):
            self.calls = 0
            self.last_error = None
            self.client = None
            self.collection_name = "stub"

        def query_cached(self, q, top_k=2):
            self.calls += 1
            return [{"page": 55, "score": 92.4, "text": "No car may overtake under the Safety Car per Article 55.8."}]

    stub_rb = _StubRulebook()
    mh = MultiHopFIARulebook(base_rulebook=stub_rb)
    mh._all_chunks_cache = [{"page": 56, "text": "55.8 Safety Car overtaking rules and exceptions."}]

    first = mh.ask_multihop("What is the Safety Car overtaking penalty?", top_k=2)
    assert first["cache_hit"] is False
    assert stub_rb.calls == 1
    first["hop1_primary_rules"].append({"page": 999, "score": 0.0, "text": "MUTATION"})

    second = mh.ask_multihop("  what is the safety car overtaking penalty?  ", top_k=2)
    assert second["cache_hit"] is True
    assert second["elapsed_ms"] <= 1.0
    assert stub_rb.calls == 1
    assert len(second["hop1_primary_rules"]) == 1  # Defensive copy prevented cache corruption!

    # E) Pit Wall Spoken Voice Normalizer & HTML Audio Bar
    from src.dashboard.radio_voice import format_text_for_f1_speech, render_f1_radio_voice_bar_html

    spoken = format_text_for_f1_speech("Box now Charles! Under SC we save +10.6s, rejoin P1, hold 2–3 laps (+1.45s/lap).")
    assert "Safety Car" in spoken
    assert "plus 10.6 seconds" in spoken
    assert "P 1" in spoken
    assert "2 to 3 laps" in spoken
    assert "1.45 seconds per lap" in spoken

    voice_html = render_f1_radio_voice_bar_html(
        speech_text=human_sc,
        message_id="p1_test_123",
        auto_play=True,
        driver_code="LEC",
        team_color="#DC0000",
    )
    assert "speechSynthesis" in voice_html
    assert "PLAY RADIO" in voice_html
    assert "playF1RadioBeep" in voice_html

    # F) Battle-primary question ("can PER catch me?") must answer about Checo, never repeat pit-only call
    df_b = sync_gp_lap_to_live_table("Bahrain", "Race", lap=17, season=2024)
    res_pit = _ask("VER", "Bahrain", 57, df_b, "do i pit now or we should wait")
    res_bat = _ask("VER", "Bahrain", 57, df_b, "can PER catch me ?")
    assert res_pit["human_radio_call"] != res_bat["human_radio_call"]
    assert "Checo" in res_bat["human_radio_call"]
    assert "wait 4 more laps" in res_pit["human_radio_call"].lower() or "extend" in res_pit["human_radio_call"].lower()

    # G) Pure FIA legal question ("can i overtake under the sc ?") must NOT trigger rival battle or pit sim
    res_leg = _ask("LEC", "Bahrain", 57, df_b, "can i overtake under the sc ?")
    assert res_leg["guardrail_passed"] is True
    assert "consult_fia_rulebook_multihop" in res_leg["required_tools"]
    assert "analyze_rival_catch_battle" not in res_leg["required_tools"]
    assert "simulate_pit_stop_mc" not in res_leg["required_tools"]
    assert "no overtaking under the safety car" in res_leg["human_radio_call"].lower()
    assert "Charles" in res_leg["human_radio_call"]
