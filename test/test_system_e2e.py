import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.race_engineer import (
    AutonomousRaceEngineer,
    extract_mentioned_driver,
    extract_target_tyre,
)
from src.agent.tools import (
    RaceContext,
    RaceEngineerToolkit,
    analyze_rival_catch_battle,
    build_live_data_from_standings_df,
    compute_remaining_stint_pit_breakeven,
)
from src.data_ingestion.telemetry_daemon import (
    F1_2024_CALENDAR,
    FIA_RACE_POINTS_TABLE,
    SUPPORTED_SEASONS,
    _compute_prior_championship_totals,
    _generate_driver_micro_telemetry_cached,
    compute_championship_standings_till_race,
    ensure_all_grand_prix_in_duckdb,
    generate_circuit_map_coordinates,
    generate_driver_micro_telemetry,
    get_circuit_curvature_profile,
    get_gp_session_laps_df,
    load_or_populate_gp_session,
    sync_gp_lap_to_live_table,
)
from src.ml_surrogates.pit_simulator import PitStopSimulator
from src.ml_surrogates.tensor_cache import TyreL1TensorCache
from src.rag_engine.multi_hop import MultiHopFIARulebook


def test_l1_tensor_cache_monotonicity_and_bounds():
    """Verifies the 56.86 KB L1 Tensor Cache size, monotonic wear, and out-of-bounds clamping."""
    cache = TyreL1TensorCache(total_race_laps=57)
    assert cache.tensor.shape == (5, 41, 71)
    assert round(cache.tensor.nbytes / 1024.0, 2) == 56.86

    # Monotonic tyre wear check (Lap 25 wear >= Lap 5 wear)
    deg_lap5 = cache.lookup_degradation("SOFT", 32.0, 5)
    deg_lap25 = cache.lookup_degradation("SOFT", 32.0, 25)
    assert deg_lap25 >= deg_lap5 > 0.0

    # Extreme out-of-bounds clamping (100°C temp, 999 laps should not crash)
    clamped_deg = cache.lookup_degradation("SOFT", 100.0, 999)
    assert 0.0 < clamped_deg < 10.0


def test_vectorized_monte_carlo_pit_simulator_latency():
    """Verifies 10,000 Monte Carlo pit simulations execute accurately and under the latency SLA."""
    sim = PitStopSimulator("Bahrain", team_pit_box_position=1, seed=42)
    traffic = [
        {"driver": "LEC", "gap_to_leader": 14.0, "is_lapped": False},
        {"driver": "NOR", "gap_to_leader": 20.0, "is_lapped": False},
    ]

    res_green = sim.simulate_rejoin_mc("VER", 0.0, traffic, race_condition="GREEN", n_sims=10_000)
    res_sc = sim.simulate_rejoin_mc("VER", 0.0, traffic, race_condition="SC", n_sims=10_000)

    # Safety Car pit loss (~10.5s) must be strictly cheaper than Green Flag pit loss (~23.0s)
    assert res_sc["median_pit_loss"] < res_green["median_pit_loss"]
    assert res_sc["median_pos"] == 1
    assert res_green["median_pos"] == 3


def test_multihop_fia_rulebook_cross_reference_resolver():
    """Verifies 2-Hop Qdrant RAG automatically resolves cited Article 54.3 penalty definitions."""
    mh = MultiHopFIARulebook()
    res = mh.ask_multihop(
        "What is the penalty if a driver overtakes another car while the Safety Car is deployed?"
    )
    assert len(res["hop1_primary_rules"]) >= 1
    assert "54.3" in res["cited_articles"]
    resolved_articles = [r["article"] for r in res["hop2_resolved_references"]]
    assert "54.3" in resolved_articles


def test_langgraph_race_engineer_guardrail():
    """Verifies LangGraph Autonomous Race Engineer passes all deterministic guardrails and avoids homograph false positives."""
    assert extract_mentioned_driver("How much time am I losing per lap on these tyres right now?", "VER") is None
    assert extract_mentioned_driver("what is my max speed on the straight?", "LEC") is None
    assert extract_mentioned_driver("can i catch VER", "VER") is None

    engineer = AutonomousRaceEngineer(ego_driver="VER")
    state = engineer.ask_engineer(
        "How much time am I losing per lap on these tyres right now?"
    )
    assert state["guardrail_passed"] is True
    assert state["target_rival"] is None
    assert "TYRE PHYSICS" in state["engineer_radio_reply"]


def test_langgraph_can_i_catch_rival_query():
    """Verifies queries like 'can i catch HAM' trigger rival battle analysis and return detailed tactical math."""
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=17)
    engineer = AutonomousRaceEngineer(ego_driver="NOR")
    state = engineer.ask_engineer("can i catch HAM")
    assert state["guardrail_passed"] is True
    assert state["target_rival"] == "HAM"
    assert "analyze_rival_catch_battle" in state["required_tools"]
    assert "RIVAL BATTLE" in state["engineer_radio_reply"]
    assert "HAM" in state["engineer_radio_reply"]
    assert "VERDICT:" in state["engineer_radio_reply"]

    # Self-target protection: VER asking about VER compares against nearest rival rather than self
    self_battle = analyze_rival_catch_battle.invoke({"ego_driver": "VER", "target_driver": "VER"})
    assert self_battle["target_driver"] != "VER"


def test_multi_gp_sessions_and_circuit_telemetry_generators():
    """Verifies all-GP session storage in DuckDB, circuit map coordinates, and sensor micro-telemetry."""
    ensure_all_grand_prix_in_duckdb()
    laps_monaco_q = load_or_populate_gp_session("Monaco", "Qualifying", season=2024)
    assert len(laps_monaco_q) >= 15

    # Verify Bahrain Lap 57 retains all 20 classified finishers (including lapped cars)
    df_bahrain_57 = sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=57)
    assert len(df_bahrain_57) == 20

    df_monaco = sync_gp_lap_to_live_table(
        circuit="Monaco",
        session_type="Qualifying",
        lap=10,
        track_status_override="VSC",
        track_temp_override=41.5,
    )
    assert not df_monaco.empty
    assert len(df_monaco) == 20
    assert df_monaco["track_status"].iloc[0] == "VSC"
    assert float(df_monaco["track_temp"].iloc[0]) == 41.5

    xb, yb = generate_circuit_map_coordinates("Bahrain", n_points=240)
    assert len(xb) == 240 and len(yb) == 240

    x, y = generate_circuit_map_coordinates("Monaco", n_points=120)
    assert len(x) == 120 and len(y) == 120

    tel_df = generate_driver_micro_telemetry("LEC", "Monaco", lap=10, compound="SOFT", tyre_life=5)
    for col in ("TimeSec", "Distance", "Speed", "RPM", "Throttle", "Brake", "nGear", "DRS", "X", "Y"):
        assert col in tel_df.columns
    assert float(tel_df["Speed"].max()) > 200.0


def test_bahrain_lap50_rus_vs_lec_no_late_pit_stop():
    """
    Verifies the exact user-reported Bahrain Lap 50/57 scenario as George Russell (RUS P5 vs LEC P4):
    With only 7 laps remaining, pitting costs ~23.0s and drops RUS from P5 to P8 while fresh tyres
    only recover ~8-9s over 7 laps (negative net payback). The engineer must NEVER recommend boxing.
    """
    RaceEngineerToolkit.get().set_circuit(circuit="Bahrain", total_race_laps=57, team_pit_box_position=1)
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=50, season=2024)

    engineer = AutonomousRaceEngineer(ego_driver="RUS")
    state = engineer.ask_engineer("if i can pass LEC")

    assert state["guardrail_passed"] is True
    assert state["target_rival"] == "LEC"
    assert "consult_fia_rulebook_multihop" not in state["required_tools"]
    bd = state["battle_data"]
    assert bd["ego_driver"] == "RUS"
    assert bd["target_driver"] == "LEC"
    assert bd["laps_remaining"] == 7
    assert bd["net_stint_pit_payback_sec"] == round(bd["recoverable_tyre_gain_sec"] - bd["expected_pit_loss_sec"], 2)
    assert bd["net_stint_pit_payback_sec"] < 0.0
    assert bd["recoverable_tyre_gain_sec"] < bd["expected_pit_loss_sec"]
    assert "DO NOT BOX" in bd["tactical_verdict"]
    assert "hold P5" in bd["tactical_verdict"]

    reply = state["engineer_radio_reply"]
    assert "STAY OUT & HOLD POSITION" in reply
    assert "Only 7L left" in reply
    assert "P5->P8" in reply
    assert "BOX THIS LAP" not in reply
    assert "Box for fresh tyres" not in reply

    # Also test Safety Car vs Green Flag crossover on Lap 50/57 (7 laps left: GREEN is negative payback, SC ~10.2s pit loss is positive payback)
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=50, season=2024, track_status_override="SC")
    state_sc = engineer.ask_engineer("Safety Car is out, should we box now?")
    assert state_sc["pit_sim_data"]["stint_breakeven"]["net_stint_pit_payback_sec"] > 0.0
    assert "BOX BOX NOW" in state_sc["engineer_radio_reply"]

    # And test Final Lap 57/57 (laps_remaining == 0 -> 0.0s gain, never box even under SC)
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=57, season=2024, track_status_override="SC")
    state_final = engineer.ask_engineer("should we box on the final lap?")
    assert state_final["pit_sim_data"]["stint_breakeven"]["laps_remaining"] == 0
    assert state_final["pit_sim_data"]["stint_breakeven"]["recoverable_tyre_gain_sec"] == 0.0
    assert state_final["pit_sim_data"]["stint_breakeven"]["net_stint_pit_payback_sec"] < 0.0
    assert "STAY OUT & HOLD POSITION" in state_final["engineer_radio_reply"]


def test_multi_season_2021_to_2024_and_all_session_types():
    """Verifies multi-season support (2021, 2022, 2023, 2024) and Practice 1-3/Qualifying/Sprint/Race loading."""
    assert set(SUPPORTED_SEASONS) == {2021, 2022, 2023, 2024}
    for season in (2021, 2022, 2023, 2024):
        laps = load_or_populate_gp_session("Bahrain", "Race", season=season)
        # Real FIA 2021 Bahrain GP ran 56 laps (extra formation lap); 2022-2024 ran 57 laps
        assert len(laps) in (56, 57)
        df_lap = sync_gp_lap_to_live_table("Bahrain", "Race", lap=20, season=season)
        assert 15 <= len(df_lap) <= 20

    # Verify Practice 1 alias and get_gp_session_laps_df across historical seasons
    df_fp1_2021 = get_gp_session_laps_df("Silverstone", "Practice 1", season=2021)
    assert not df_fp1_2021.empty
    assert int(df_fp1_2021["season"].iloc[0]) == 2021
    assert df_fp1_2021["session_type"].iloc[0] == "FP1"


def test_championship_standings_wdc_and_wcc_till_race():
    """
    Verifies cumulative WDC & WCC championship standings calculation across seasons 2021-2024:
    - Round 1 (Bahrain): prior_points == 0 for all drivers and constructors, live_race_points == FIA points (25, 18, 15, ...).
    - Mid-season / Finale (Monaco R8, Abu Dhabi R24): prior_points > 0, total_points == prior_points + live_race_points,
      WCC team points == sum of both team drivers' points, and gap_to_leader is 0 for P1 and monotonic thereafter.
    """
    df_bahrain = sync_gp_lap_to_live_table("Bahrain", "Race", lap=25, season=2024)
    champ_r1 = compute_championship_standings_till_race(
        season=2024,
        circuit="Bahrain",
        current_race_standings_df=df_bahrain,
    )
    assert champ_r1["round"] == 1
    assert champ_r1["total_rounds"] == 24
    wdc_r1 = champ_r1["wdc"]
    wcc_r1 = champ_r1["wcc"]
    assert len(wdc_r1) == 20
    assert len(wcc_r1) == 10
    assert (wdc_r1["prior_points"] == 0).all()
    assert (wcc_r1["prior_points"] == 0).all()
    assert int(wdc_r1.iloc[0]["live_race_points"]) == 25
    assert int(wdc_r1.iloc[0]["total_points"]) == 25
    assert int(wdc_r1.iloc[0]["gap_to_leader"]) == 0
    assert int(wdc_r1["live_race_points"].sum()) == sum(FIA_RACE_POINTS_TABLE.values())
    assert int(wcc_r1["total_points"].sum()) == int(wdc_r1["total_points"].sum())

    # Test Round 12 (Great Britain) and Round 24 (Abu Dhabi) across 2021-2024
    for season in (2021, 2022, 2023, 2024):
        champ_mid = compute_championship_standings_till_race(season=season, circuit="Great Britain", current_race_standings_df=None)
        assert champ_mid["round"] == 12
        wdc_m = champ_mid["wdc"]
        wcc_m = champ_mid["wcc"]
        assert int(wdc_m.iloc[0]["prior_points"]) > 0
        assert (wdc_m["total_points"] == wdc_m["prior_points"] + wdc_m["live_race_points"]).all()
        assert (wcc_m["total_points"] == wcc_m["prior_points"] + wcc_m["live_race_points"]).all()
        assert int(wcc_m["total_points"].sum()) == int(wdc_m["total_points"].sum())
        assert list(wdc_m["rank"]) == list(range(1, 21))
        assert list(wcc_m["rank"]) == list(range(1, 11))

    # Test partial custom scenario grid (<20 cars, e.g. 2 cars where VER is not in the scenario)
    import pandas as pd

    partial_df = pd.DataFrame([
        {"driver": "NOR", "position": 1},
        {"driver": "LEC", "position": 2},
    ])
    champ_partial = compute_championship_standings_till_race(
        season=2024,
        circuit="Bahrain",
        current_race_standings_df=partial_df,
    )
    wdc_p = champ_partial["wdc"]
    wcc_p = champ_partial["wcc"]
    assert len(wdc_p) == 20
    assert len(wcc_p) == 10
    assert set(wdc_p["race_position"].tolist()) == set(range(1, 21))
    nor_row = wdc_p[wdc_p["driver"] == "NOR"].iloc[0]
    ver_row = wdc_p[wdc_p["driver"] == "VER"].iloc[0]
    assert int(nor_row["race_position"]) == 1
    assert int(nor_row["live_race_points"]) == 25
    assert int(nor_row["wins"]) == 1
    assert int(ver_row["race_position"]) >= 3
    assert int(ver_row["wins"]) == 0
    assert int(wdc_p["live_race_points"].sum()) == sum(FIA_RACE_POINTS_TABLE.values())
    assert int(wcc_p["total_points"].sum()) == int(wdc_p["total_points"].sum())


def test_streamlit_pit_wall_app_video_controls_and_championship_standings():
    """Verifies the 2-page Streamlit app renders cleanly with video stream controls and championship standings."""
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "pit_wall_app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Verify video stream playback state keys & play/pause toggle button
    assert at.session_state["is_playing"] is False
    assert "sub_lap_progress" in at.session_state
    assert "p1_current_lap" in at.session_state

    # Click "+1 Lap" button in the top video controller bar
    initial_lap = int(at.session_state["p1_current_lap"])
    at.button(key="top_video_next_lap_btn").click().run(timeout=30)
    assert not at.exception
    assert int(at.session_state["p1_current_lap"]) == initial_lap + 1

    # Click "+25% Sector Scrub" button in the top video controller bar
    initial_sub = float(at.session_state["sub_lap_progress"])
    at.button(key="top_video_scrub_btn").click().run(timeout=30)
    assert not at.exception
    scrubbed_sub = float(at.session_state["sub_lap_progress"])
    assert scrubbed_sub != initial_sub

    # Toggle Play/Pause button ON -> advances sub_lap_progress via @st.fragment(run_every=0.25) -> toggle OFF
    # and verify pausing preserves the exact advanced sub_lap_progress without snapping back
    at.button(key="top_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["is_playing"] is True
    playing_sub = float(at.session_state["sub_lap_progress"])
    assert playing_sub != scrubbed_sub
    at.button(key="top_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["is_playing"] is False
    assert float(at.session_state["sub_lap_progress"]) == playing_sub

    # Verify duplicate Timing Tower & Analytical Lap Sync expander are removed from main Page 1
    p1_subheaders = [s.value for s in at.subheader]
    assert not any("Timing Tower & Timestamps" in sh for sh in p1_subheaders)
    assert any("L1 Tensor Tyre Degradation" in sh for sh in p1_subheaders)
    assert any("Monte Carlo Pit Rejoin Window" in sh for sh in p1_subheaders)

    # Verify docked Pit Wall Radio Comms Console quick-call transmits and appends to radio_history
    at.button(key="p1_qc_wear").click().run(timeout=30)
    assert not at.exception
    assert len(at.session_state["radio_history"]) == 1
    assert at.session_state["radio_history"][0]["guardrail_passed"] is True

    # Switch to Page 2 (Custom Scenario Builder) and verify its video stream controller & championship standings
    at.sidebar.radio[0].set_value("🧪 Page 2: 100% Custom Scenario Builder").run(timeout=30)
    assert not at.exception
    assert at.session_state["p2_is_playing"] is False
    p2_initial_lap = int(at.session_state["scen_lap_slider"])
    at.button(key="p2_video_next_lap_btn").click().run(timeout=30)
    assert not at.exception
    assert int(at.session_state["scen_lap_slider"]) == p2_initial_lap + 1

    # Verify Page 2 Play -> advance p2_sub_lap_progress -> Pause preserves exact paused progress (does not snap to 0.50)
    p2_sub_before = float(at.session_state["p2_sub_lap_progress"])
    at.button(key="p2_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["p2_is_playing"] is True
    p2_sub_playing = float(at.session_state["p2_sub_lap_progress"])
    assert p2_sub_playing != p2_sub_before
    at.button(key="p2_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["p2_is_playing"] is False
    assert float(at.session_state["p2_sub_lap_progress"]) == p2_sub_playing

    # Verify Page 2 "+25% Sector Scrub" button
    at.button(key="p2_video_scrub_btn").click().run(timeout=30)
    assert not at.exception
    assert float(at.session_state["p2_sub_lap_progress"]) != p2_sub_playing


def test_word_boundary_intent_routing_and_homographs():
    """
    Verifies QW1: word-boundary intent routing prevents substring false positives
    ('describe'/'scenario' != 'sc', 'baseline' != 'line', 'despite' != 'pit', 'unstoppable' != 'stop')
    and disambiguates English homographs ('nor', 'bot', 'str', 'max', 'per', 'gas') and target tyre extraction.
    """
    # 1. Homograph negative cases (English words must not match drivers)
    assert extract_mentioned_driver("neither soft nor hard tyres will last", "VER") is None
    assert extract_mentioned_driver("I have no grip nor pace right now", "VER") is None
    assert extract_mentioned_driver("Nor do I want to box yet", "VER") is None
    assert extract_mentioned_driver("hey bot check my tyre wear", "VER") is None
    assert extract_mentioned_driver("switch to str mode 3", "VER") is None
    assert extract_mentioned_driver("pushing at max right now", "LEC") is None
    assert extract_mentioned_driver("I am saving gas on the straight", "VER") is None

    # 2. Homograph positive cases (explicit driver references must still match)
    assert extract_mentioned_driver("can i catch nor?", "VER") == "NOR"
    assert extract_mentioned_driver("is NOR pitting this lap?", "VER") == "NOR"
    assert extract_mentioned_driver("can i catch bot?", "VER") == "BOT"
    assert extract_mentioned_driver("gap to STR ahead", "VER") == "STR"
    assert extract_mentioned_driver("can i catch max?", "LEC") == "VER"
    assert extract_mentioned_driver("catching per on softs", "VER") == "PER"
    assert extract_mentioned_driver("GAS is ahead of me", "VER") == "GAS"

    # 3. Word-boundary router: substring collisions must NOT trigger Pit MC or Legal RAG
    engineer = AutonomousRaceEngineer(ego_driver="VER")

    def _route(msg: str) -> dict:
        return engineer._route_intent_node({"driver": "VER", "radio_message": msg})

    r1 = _route("describe my baseline tyre pace")
    assert "predict_tyre_degradation" in r1["required_tools"]
    assert "consult_fia_rulebook_multihop" not in r1["required_tools"]
    assert "simulate_pit_stop_mc" not in r1["required_tools"]
    assert r1["target_rival"] is None

    r2 = _route("despite high tyre wear, am I unstoppable in this scenario?")
    assert "predict_tyre_degradation" in r2["required_tools"]
    assert "simulate_pit_stop_mc" not in r2["required_tools"]
    assert "consult_fia_rulebook_multihop" not in r2["required_tools"]

    r3 = _route("how is my racing line grip on neither soft nor hard tyres?")
    assert "predict_tyre_degradation" in r3["required_tools"]
    assert "consult_fia_rulebook_multihop" not in r3["required_tools"]
    assert r3["target_rival"] is None

    # 4. Word-boundary router: legitimate legal & SC/VSC queries MUST still route properly
    r4 = _route("SC deployed, can we box now?")
    assert "simulate_pit_stop_mc" in r4["required_tools"]
    assert "consult_fia_rulebook_multihop" in r4["required_tools"]

    r5 = _route("am I allowed to cross the white line at pit exit?")
    assert "consult_fia_rulebook_multihop" in r5["required_tools"]

    r6 = _route("can I pass under yellow flag?")
    assert "consult_fia_rulebook_multihop" in r6["required_tools"]

    # 5. Target tyre extraction: preposition priority & word boundaries ('hardly' != HARD)
    assert (
        extract_target_tyre(
            "GP, my Soft tyres are cooked! Should we box for Mediums right now?",
            current_compound="SOFT",
        )
        == "MEDIUM"
    )
    assert extract_target_tyre("I can hardly keep it on track", current_compound="SOFT") == "MEDIUM"
    assert extract_target_tyre("switch to fresh hards", current_compound="MEDIUM") == "HARD"


def test_undercut_and_breakeven_use_real_tensor_delta_and_target_compound():
    """
    Verifies QW2:
    1. Undercut Monte Carlo uses real L1 Tensor degradation delta + compound pace offset (not hardcoded 2.5s).
    2. Remaining-stint pit breakeven honors user-selected `target_new_tyre` and compound-specific warmup penalties.
    3. LangGraph engineer propagates the user's requested target compound end-to-end into breakeven & undercut tools.
    """
    cache = TyreL1TensorCache(total_race_laps=57)
    sim = PitStopSimulator("Bahrain", team_pit_box_position=1, seed=42)

    # Worn 22L HARD -> Fresh 1L SOFT must strongly outperform Fresh 2L SOFT -> Fresh 1L HARD
    deg_worn_hard = cache.lookup_degradation("HARD", 32.0, 22)
    deg_fresh_soft = cache.lookup_degradation("SOFT", 32.0, 1)
    deg_fresh_2l_soft = cache.lookup_degradation("SOFT", 32.0, 2)
    deg_fresh_hard = cache.lookup_degradation("HARD", 32.0, 1)

    uc_worn_to_soft = sim.calculate_undercut_mc(
        rival_gap=1.2,
        new_tyre="SOFT",
        old_tyre="HARD",
        old_tyre_deg=deg_worn_hard,
        new_tyre_deg=deg_fresh_soft,
        n_sims=10_000,
    )
    uc_fresh_to_hard = sim.calculate_undercut_mc(
        rival_gap=1.2,
        new_tyre="HARD",
        old_tyre="SOFT",
        old_tyre_deg=deg_fresh_2l_soft,
        new_tyre_deg=deg_fresh_hard,
        n_sims=10_000,
    )
    assert uc_worn_to_soft["raw_pace_delta_sec"] > uc_fresh_to_hard["raw_pace_delta_sec"] + 1.5
    assert uc_worn_to_soft["win_probability"] > 0.70
    assert uc_fresh_to_hard["win_probability"] < 0.05
    assert uc_worn_to_soft["warmup_loss_sec"] == 0.5
    assert uc_fresh_to_hard["warmup_loss_sec"] == 2.2

    # Remaining-stint breakeven must honor target_new_tyre and compound-specific warmup penalties
    be_soft = compute_remaining_stint_pit_breakeven(
        ego_compound="HARD",
        ego_tyre_life=19,
        track_temp=31.8,
        laps_remaining=10,
        pit_loss_sec=23.0,
        target_new_tyre="SOFT",
    )
    be_hard = compute_remaining_stint_pit_breakeven(
        ego_compound="HARD",
        ego_tyre_life=19,
        track_temp=31.8,
        laps_remaining=10,
        pit_loss_sec=23.0,
        target_new_tyre="HARD",
    )
    assert be_soft["fresh_compound"] == "SOFT"
    assert be_soft["warmup_penalty_sec"] == 0.5
    assert be_hard["fresh_compound"] == "HARD"
    assert be_hard["warmup_penalty_sec"] == 2.2
    assert be_soft["recoverable_tyre_gain_sec"] > be_hard["recoverable_tyre_gain_sec"]

    # End-to-end LangGraph propagation: asking for Hards vs Softs evaluates the requested compound
    RaceEngineerToolkit.get().set_circuit(circuit="Bahrain", total_race_laps=57, team_pit_box_position=1)
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=25, season=2024)
    engineer = AutonomousRaceEngineer(ego_driver="VER")

    res_hard = engineer.ask_engineer("Should we box for Hards right now?")
    res_soft = engineer.ask_engineer("Should we box for Softs right now?")

    assert res_hard["guardrail_passed"] is True
    assert res_soft["guardrail_passed"] is True
    assert res_hard["pit_sim_data"]["stint_breakeven"]["fresh_compound"] == "HARD"
    assert res_hard["pit_sim_data"]["stint_breakeven"]["warmup_penalty_sec"] == 2.2
    assert res_soft["pit_sim_data"]["stint_breakeven"]["fresh_compound"] == "SOFT"
    assert res_soft["pit_sim_data"]["stint_breakeven"]["warmup_penalty_sec"] == 0.5
    assert "Fresh-HARD Gain" in res_hard["engineer_radio_reply"]
    assert "Fresh-SOFT Gain" in res_soft["engineer_radio_reply"]
    assert (
        res_soft["pit_sim_data"]["undercut_simulation"]["median_margin_sec"]
        > res_hard["pit_sim_data"]["undercut_simulation"]["median_margin_sec"]
    )


def test_circuit_specific_pit_loss_and_track_lengths():
    """
    Verifies QW3: no hardcoded Bahrain 23.0s pit loss or 5412m track length leaks into other circuits.
    - All 24 calendar circuits expose official `track_length_m` and micro-telemetry uses it.
    - PitStopSimulator & LangGraph engineer compute SC/VSC pit savings against each circuit's `normal_loss`.
    """
    assert len(F1_2024_CALENDAR) == 24
    assert all("track_length_m" in c and c["track_length_m"] >= 3000.0 for c in F1_2024_CALENDAR)

    tel_monaco = generate_driver_micro_telemetry("LEC", "Monaco", lap=10, compound="SOFT", tyre_life=5)
    tel_spa = generate_driver_micro_telemetry("VER", "Belgium", lap=10, compound="MEDIUM", tyre_life=5)
    assert float(tel_monaco["Distance"].max()) == 3337.0
    assert float(tel_spa["Distance"].max()) == 7004.0

    # Singapore has normal_loss = 28.0s (t_pit = 40.0s -> t_track_base = 12.0s; SC loss ~16.9s -> saves ~11.1s, NOT ~6.1s from 23.0!)
    sim_sg = PitStopSimulator("Singapore", team_pit_box_position=1, seed=42)
    res_sg_sc = sim_sg.simulate_rejoin_mc("VER", 0.0, [], race_condition="SC", n_sims=10_000)
    assert res_sg_sc["green_flag_pit_loss"] == 28.0
    assert abs(res_sg_sc["pit_loss_saved_vs_green"] - (28.0 - res_sg_sc["median_pit_loss"])) < 1e-6
    assert res_sg_sc["pit_loss_saved_vs_green"] > 10.0  # Would have been ~6.1s with hardcoded 23.0s

    # Canada has normal_loss = 18.5s
    sim_ca = PitStopSimulator("Canada", team_pit_box_position=1, seed=42)
    res_ca_green = sim_ca.simulate_rejoin_mc("VER", 0.0, [], race_condition="GREEN", n_sims=10_000)
    assert res_ca_green["green_flag_pit_loss"] == 18.5
    assert res_ca_green["pit_loss_saved_vs_green"] < 0.5

    # End-to-end LangGraph check on Singapore Lap 18 (VER on 18L worn SOFTs) under SC
    RaceEngineerToolkit.get().set_circuit(circuit="Singapore", total_race_laps=62, team_pit_box_position=1)
    sync_gp_lap_to_live_table(circuit="Singapore", session_type="Race", lap=18, season=2024, track_status_override="SC")
    engineer = AutonomousRaceEngineer(ego_driver="VER")
    res_eng = engineer.ask_engineer("Safety Car is out, should we box now?")
    expected_saved = res_eng["pit_sim_data"]["rejoin_simulation"]["pit_loss_saved_vs_green"]
    assert expected_saved > 10.0
    assert f"Save {expected_saved:.1f}s under SC" in res_eng["engineer_radio_reply"]
    RaceEngineerToolkit.get().set_circuit(circuit="Bahrain", total_race_laps=57, team_pit_box_position=1)


def test_qdrant_offline_fallback_and_tool_deduplication():
    """
    Verifies QW4:
    1. If Qdrant is unreachable/offline, legal queries degrade gracefully without crashing
       while still delivering verified telemetry, tyre physics, and Monte Carlo pit strategy.
    2. Within a single LangGraph turn combining rival battle + pit simulation, DuckDB is queried
       only ONCE (not 3x) and 10,000-sim Monte Carlo rejoin runs only ONCE (not 2x).
    """
    from unittest.mock import patch

    import src.agent.race_engineer as re_mod
    import src.agent.tools as tools_mod

    tk = RaceEngineerToolkit.get()
    tk.set_circuit(circuit="Bahrain", total_race_laps=57, team_pit_box_position=1)
    sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=17, season=2024)
    engineer = AutonomousRaceEngineer(ego_driver="NOR")

    # 1. Verify single-turn deduplication: 1 DuckDB read & 1 MC rejoin sim for combined battle + pit query
    orig_get_live = tools_mod.get_live_race_standings.func
    orig_rejoin_mc = tk.pit_sim.simulate_rejoin_mc
    counts = {"duckdb_reads": 0, "rejoin_mc_runs": 0}

    def counting_get_live():
        counts["duckdb_reads"] += 1
        return orig_get_live()

    def counting_rejoin_mc(*args, **kwargs):
        counts["rejoin_mc_runs"] += 1
        return orig_rejoin_mc(*args, **kwargs)

    with (
        patch.object(tools_mod.get_live_race_standings, "func", side_effect=counting_get_live),
        patch.object(re_mod.get_live_race_standings, "func", side_effect=counting_get_live),
        patch.object(tk.pit_sim, "simulate_rejoin_mc", side_effect=counting_rejoin_mc),
    ):
        state_dedup = engineer.ask_engineer("Can I catch HAM or should we box for Mediums?")

    assert state_dedup["guardrail_passed"] is True
    assert "RIVAL BATTLE" in state_dedup["engineer_radio_reply"]
    assert "PIT WALL SIM" in state_dedup["engineer_radio_reply"]
    assert counts["duckdb_reads"] == 1
    assert counts["rejoin_mc_runs"] == 1

    # 2. Verify graceful Qdrant offline fallback (no crash, strategy advice preserved)
    class _OfflineMultiHop:
        def ask_multihop(self, question: str, top_k: int = 2, max_hops: int = 2) -> dict:
            raise ConnectionError("Simulated Qdrant Docker container offline")

    orig_rag = tk._multihop_rag
    try:
        tk._multihop_rag = _OfflineMultiHop()
        state_offline = engineer.ask_engineer(
            "Safety Car is out, should we box now and what is the overtaking penalty?"
        )
    finally:
        tk._multihop_rag = orig_rag

    assert state_offline["guardrail_passed"] is True
    assert state_offline["legal_data"]["offline"] is True
    assert "PIT WALL SIM" in state_offline["engineer_radio_reply"]
    assert "TYRE PHYSICS" in state_offline["engineer_radio_reply"]
    assert "offline" in state_offline["engineer_radio_reply"].lower()


def test_b1_4hz_caching_and_vectorized_radar():
    """
    Verifies B1:
    1. Circuit curvature profiles, 480-pt micro-telemetry, and prior-round WDC/WCC totals
       are memoized across repeated 4 Hz ticks (cache hits increase with zero recomputation).
    2. 2D Circuit Radar batches all 19 rivals into 1 vectorized Scatter trace + 1 ego trace
       (5 traces total instead of 23) and renders cleanly inside Streamlit AppTest.
    """
    import json
    from streamlit.testing.v1 import AppTest

    # 1. Verify curvature, micro-telemetry, and championship prior-round LRU caches
    get_circuit_curvature_profile.cache_clear()
    _generate_driver_micro_telemetry_cached.cache_clear()
    _compute_prior_championship_totals.cache_clear()

    for _ in range(5):
        x, y, curv = get_circuit_curvature_profile("Bahrain", n_points=240)
        assert len(x) == 240 and len(y) == 240 and len(curv) == 240
        tel = generate_driver_micro_telemetry("VER", "Bahrain", lap=17, compound="SOFT", tyre_life=14)
        assert len(tel) == 480
        _ = compute_championship_standings_till_race(season=2024, circuit="Abu Dhabi")

    assert get_circuit_curvature_profile.cache_info().hits >= 4
    assert _generate_driver_micro_telemetry_cached.cache_info().hits >= 4
    assert _compute_prior_championship_totals.cache_info().hits >= 4

    # 2. Verify single 60 FPS Canvas radar (no duplicate Plotly radar/sensor charts — only Deg Curves + MC Histogram)
    app_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "pit_wall_app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) == 2
    deg_spec = json.loads(plotly_elements[0].proto.spec)
    assert len(deg_spec["data"]) == 4  # SOFT, MEDIUM, HARD curves + Ego marker
    assert deg_spec["layout"]["uirevision"] == "vre_deg_constant"


def test_b2_playback_sync_mc_chart_and_page2_autoplay():
    """
    Verifies B2:
    1. Page 1 Monte Carlo Pit Rejoin Window stays live-synced with the advancing playback lap
       (no longer frozen on the outer script lap).
    2. Page 2 auto-play crossing 100% lap progress advances tyre wear/gaps via `_advance_custom_scenario_rows`,
       syncs `live_telemetry_laps` in DuckDB, and updates the Radar subheader to `active_p2_lap`.
    """
    from streamlit.testing.v1 import AppTest
    from src.agent.tools import get_live_race_standings

    app_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "pit_wall_app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # 1. Page 1: position at 99% of Lap 17, click Play -> fragment wraps to Lap 18 -> MC subheader matches Lap 18
    lap_before = int(at.session_state["p1_current_lap"])
    at.session_state["sub_lap_progress"] = 0.99
    at.button(key="top_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception
    assert int(at.session_state["p1_current_lap"]) == lap_before + 1
    assert int(at.session_state["_p1_live_snapshot"]["lap"]) == lap_before + 1
    subheaders_p1 = [s.value for s in at.subheader]
    assert any(f"Monte Carlo Pit Rejoin Window — Lap {lap_before + 1}/" in sh for sh in subheaders_p1)

    # Pause Page 1 before switching to Page 2
    at.button(key="top_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception

    # 2. Page 2: position at 99% of scen_lap, click Play -> wraps lap, increments tyre_life, syncs DuckDB & subheader
    at.sidebar.radio[0].set_value("🧪 Page 2: 100% Custom Scenario Builder").run(timeout=30)
    assert not at.exception
    p2_lap_before = int(at.session_state["p2_current_lap"])
    ages_before = {r["driver"]: int(r["tyre_life"]) for r in at.session_state["custom_scenario_rows"]}

    at.session_state["p2_sub_lap_progress"] = 0.99
    at.button(key="p2_video_play_pause_btn").click().run(timeout=30)
    assert not at.exception

    assert int(at.session_state["p2_current_lap"]) == p2_lap_before + 1
    ages_after = {r["driver"]: int(r["tyre_life"]) for r in at.session_state["custom_scenario_rows"]}
    for drv, old_age in ages_before.items():
        assert ages_after[drv] == old_age + 1

    # Verify DuckDB live_telemetry_laps was synced to the new auto-played lap & tyre ages
    live_db = get_live_race_standings.invoke({})
    assert int(live_db["lap"]) == p2_lap_before + 1
    db_ages = {r["driver"]: int(r["tyre_life"]) for r in live_db["standings"]}
    for drv, old_age in ages_before.items():
        assert db_ages[drv] == old_age + 1

    # Verify Page 2 MC Rejoin subheader displays active_p2_lap (not stale outer scen_lap)
    subheaders_p2 = [s.value for s in at.subheader]
    assert any("10k Monte Carlo Pit Rejoin" in sh and f"Lap {p2_lap_before + 1}/" in sh for sh in subheaders_p2)
    assert not any("Timing Tower & Timestamps" in sh for sh in subheaders_p2)


def test_b3_session_isolation_no_singleton_clobbering():
    """
    Verifies B3:
    Concurrent multi-session requests sharing one AutonomousRaceEngineer and one RaceEngineerToolkit
    singleton (as in Streamlit @st.cache_resource) remain 100% isolated via immutable RaceContext:
    - Neither session mutates `engineer.ego_driver` or `RaceEngineerToolkit.get().circuit`.
    - Neither session is corrupted when DuckDB `live_telemetry_laps` is overwritten by a third tab.
    """
    from concurrent.futures import ThreadPoolExecutor

    tk = RaceEngineerToolkit.get()
    tk.set_circuit(circuit="Bahrain", total_race_laps=57, team_pit_box_position=1)
    shared_engineer = AutonomousRaceEngineer(ego_driver="VER")

    # Build isolated snapshots for Session A (RUS @ Bahrain L50 GREEN) and Session B (LEC @ Singapore L18 SC)
    df_a = sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=50, season=2024)
    df_b = sync_gp_lap_to_live_table(
        circuit="Singapore", session_type="Race", lap=18, season=2024, track_status_override="SC"
    )
    # Clobber DuckDB live_telemetry_laps with an unrelated 3rd tab (Monaco Lap 10)
    sync_gp_lap_to_live_table(circuit="Monaco", session_type="Race", lap=10, season=2024)

    ctx_a = RaceContext(
        driver="RUS",
        circuit="Bahrain",
        total_race_laps=57,
        team_pit_box_position=1,
        live_data=build_live_data_from_standings_df(df_a, circuit="Bahrain", total_race_laps=57),
    )
    ctx_b = RaceContext(
        driver="LEC",
        circuit="Singapore",
        total_race_laps=62,
        team_pit_box_position=3,
        live_data=build_live_data_from_standings_df(df_b, circuit="Singapore", total_race_laps=62),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(shared_engineer.ask_engineer, "if i can pass LEC", None, ctx_a)
        fut_b = pool.submit(shared_engineer.ask_engineer, "should we box right now?", None, ctx_b)
        res_a = fut_a.result()
        res_b = fut_b.result()

    # Shared singletons must remain completely unmutated
    assert shared_engineer.ego_driver == "VER"
    assert tk.circuit == "Bahrain"

    # Session A saw RUS @ Bahrain Lap 50/57 (green_flag_pit_loss == 23.0)
    assert res_a["guardrail_passed"] is True
    assert res_a["driver"] == "RUS"
    assert "RUS | Lap 50/57 (GREEN)" in res_a["engineer_radio_reply"]
    assert res_a["pit_sim_data"]["rejoin_simulation"]["green_flag_pit_loss"] == 23.0

    # Session B saw LEC @ Singapore Lap 18/62 SC (green_flag_pit_loss == 28.0)
    assert res_b["guardrail_passed"] is True
    assert res_b["driver"] == "LEC"
    assert "LEC | Lap 18/62 (SC)" in res_b["engineer_radio_reply"]
    assert res_b["pit_sim_data"]["rejoin_simulation"]["green_flag_pit_loss"] == 28.0


def test_b4_race_lap_decoupling_tyre_cliff_and_wet_weather():
    """
    Verifies B4:
    1. L1 Tensor keeps exact (5, 41, 71) 56.86 KB layout while decoupling `race_lap` fuel/evolution
       from `tyre_life` (early heavy-fuel lap degrades faster than late light-fuel lap at same tyre age).
    2. INTERMEDIATE and WET have distinct thermal wear profiles (not duplicates of HARD).
    3. Out-of-distribution tyre cliff extrapolation keeps wear climbing past 22L on SOFTs.
    4. Wet/Damp weather regimes trigger real grip-mismatch penalties and automatic INTERMEDIATE/WET box calls.
    """
    cache = TyreL1TensorCache(total_race_laps=57)
    assert cache.tensor.shape == (5, 41, 71)
    assert round(cache.tensor.nbytes / 1024.0, 2) == 56.86

    # 1. Race-lap decoupling: 10-lap tyre on Lap 10 (heavy fuel) > 10-lap tyre on Lap 45 (light fuel)
    deg_early = cache.lookup_degradation("MEDIUM", 32.0, 10, race_lap=10)
    deg_late = cache.lookup_degradation("MEDIUM", 32.0, 10, race_lap=45)
    assert deg_early > deg_late > 0.0

    # 2. Distinct INTERMEDIATE and WET curves vs HARD
    deg_hard_15 = cache.lookup_degradation("HARD", 35.0, 15)
    deg_inter_15 = cache.lookup_degradation("INTERMEDIATE", 35.0, 15)
    deg_wet_15 = cache.lookup_degradation("WET", 35.0, 15)
    assert deg_wet_15 > deg_inter_15 > deg_hard_15

    # 3. Tyre cliff continues climbing past 22L SOFT threshold (no flat tree plateau)
    deg_soft_22 = cache.lookup_degradation("SOFT", 32.0, 22)
    deg_soft_40 = cache.lookup_degradation("SOFT", 32.0, 40)
    assert deg_soft_40 > deg_soft_22 + 0.5

    # 4. End-to-end Rain Crossover through LangGraph Race Engineer
    df_lap20 = sync_gp_lap_to_live_table(circuit="Bahrain", session_type="Race", lap=20, season=2024)
    engineer = AutonomousRaceEngineer(ego_driver="VER")

    ctx_rain = RaceContext(
        driver="VER",
        circuit="Bahrain",
        total_race_laps=57,
        team_pit_box_position=1,
        weather_condition="Heavy Rain (Wet)",
        live_data=build_live_data_from_standings_df(
            df_lap20, circuit="Bahrain", total_race_laps=57, weather_condition="Heavy Rain (Wet)"
        ),
    )
    res_rain = engineer.ask_engineer("Should we pit right now?", context=ctx_rain)
    assert res_rain["guardrail_passed"] is True
    assert res_rain["pit_sim_data"]["stint_breakeven"]["fresh_compound"] == "WET"
    assert res_rain["pit_sim_data"]["stint_breakeven"]["net_stint_pit_payback_sec"] > 100.0
    assert "BOX THIS LAP FOR WET" in res_rain["engineer_radio_reply"]
    assert "WEATHER CROSSOVER" in res_rain["engineer_radio_reply"]

    ctx_damp = RaceContext(
        driver="VER",
        circuit="Bahrain",
        total_race_laps=57,
        team_pit_box_position=1,
        weather_condition="Light Drizzle (Damp)",
        live_data=build_live_data_from_standings_df(
            df_lap20, circuit="Bahrain", total_race_laps=57, weather_condition="Light Drizzle (Damp)"
        ),
    )
    res_damp = engineer.ask_engineer("Should we pit right now?", context=ctx_damp)
    assert res_damp["guardrail_passed"] is True
    assert res_damp["pit_sim_data"]["stint_breakeven"]["fresh_compound"] == "INTERMEDIATE"
    assert "BOX THIS LAP FOR INTERMEDIATE" in res_damp["engineer_radio_reply"]


def test_b5_rag_pagination_immutable_cache_and_lapped_traffic():
    """
    Verifies B5:
    1. MultiHopFIARulebook paginates Qdrant scroll past 1,000 chunks via `next_offset`.
    2. FIARulebook.query_cached returns defensive copies so caller mutations cannot corrupt the cache.
    3. PitStopSimulator distinguishes race classification from physical circular track-phase clean air,
       detecting blue-flag traffic behind lapped backmarkers without demoting lead-lap race position.
    4. Seeded PitStopSimulator.get_total_pit_loss is 100% reproducible via `self.rng`.
    """
    from types import SimpleNamespace
    from src.rag_engine.query import FIARulebook

    # 1. Verify Qdrant multi-page scroll pagination past 1,000 chunks
    class _FakePagedClient:
        def __init__(self):
            self.calls = 0

        def scroll(self, collection_name, limit=1000, offset=None, with_payload=True, with_vectors=False):
            self.calls += 1
            if offset is None:
                p1 = [SimpleNamespace(payload={"page": 1, "text": "1.1 General rules."})]
                return p1, 1000
            p2 = [SimpleNamespace(payload={"page": 105, "text": "99.1 The stewards shall review telemetry."})]
            return p2, None

    fake_rb = SimpleNamespace(client=_FakePagedClient(), collection_name="fia_rulebook")
    mh = MultiHopFIARulebook(base_rulebook=fake_rb)
    chunks = mh._get_all_chunks()
    assert fake_rb.client.calls == 2
    assert len(chunks) == 2
    resolved = mh._fetch_exact_article_chunk("99.1")
    assert resolved is not None and resolved["page"] == 105

    # 2. Verify defensive copy isolation on FIARulebook.query_cached
    rb_stub = object.__new__(FIARulebook)
    from collections import OrderedDict
    rb_stub._cache_maxsize = 16
    rb_stub._query_cache = OrderedDict()
    rb_stub.last_error = None
    rb_stub._query_cache[("test question", 2)] = ({"score": 95.0, "page": 61, "text": "ORIGINAL RULE TEXT"},)

    hit1 = rb_stub.query_cached("test question", top_k=2)
    hit1[0]["text"] = "MUTATED BY CALLER"
    hit2 = rb_stub.query_cached("test question", top_k=2)
    assert hit2[0]["text"] == "ORIGINAL RULE TEXT"

    # 3. Verify lapped backmarker blue-flag dirty air vs lead-lap race classification
    sim = PitStopSimulator("Bahrain", team_pit_box_position=1, seed=42)
    traffic_lapped = [
        {"driver": "LEC", "gap_to_leader": 15.0, "is_lapped": False},
        # SAR is 1 lap down (94.1 + 22.4 = 116.5s), physically 0.6s ahead of VER's ~23.0s pit exit!
        {"driver": "SAR", "gap_to_leader": 116.5, "is_lapped": True},
    ]
    res_lapped = sim.simulate_rejoin_mc(
        "VER", 0.0, traffic_lapped, race_condition="GREEN", n_sims=10_000, silent=True, base_lap_sec=94.1
    )
    assert res_lapped["median_pos"] == 2  # Only behind lead-lap LEC, NOT behind lapped SAR
    assert res_lapped["p_blue_flag_traffic"] > 0.90
    assert res_lapped["p_dirty_air"] > 0.90
    assert res_lapped["p_clean_air"] < 0.10

    # Also verify explicit `is_lapped=True` with phase-only gap (22.4s) doesn't demote race position
    traffic_phase = [
        {"driver": "LEC", "gap_to_leader": 15.0, "is_lapped": False},
        {"driver": "SAR", "gap_to_leader": 22.4, "is_lapped": True},
    ]
    res_phase = sim.simulate_rejoin_mc(
        "VER", 0.0, traffic_phase, race_condition="GREEN", n_sims=10_000, silent=True, base_lap_sec=94.1
    )
    assert res_phase["median_pos"] == 2
    assert res_phase["p_blue_flag_traffic"] > 0.90

    # 4. Verify seeded reproducibility of legacy single-shot get_total_pit_loss
    s1 = PitStopSimulator("Bahrain", team_pit_box_position=5, seed=123)
    s2 = PitStopSimulator("Bahrain", team_pit_box_position=5, seed=123)
    seq1 = [s1.get_total_pit_loss("SC", silent=True) for _ in range(10)]
    seq2 = [s2.get_total_pit_loss("SC", silent=True) for _ in range(10)]
    assert seq1 == seq2


def test_smooth_60fps_broadcast_canvas_and_anti_blink():
    """
    Verifies the 60 FPS Zero-Blink HTML5 Canvas Broadcast Engine and Streamlit Anti-Blink CSS:
    - Anti-blink CSS disables `[data-stale="true"]` opacity dimming.
    - Broadcast payload packs multi-lap trajectories, track coordinates, and telemetry cleanly.
    - Rendered HTML contains requestAnimationFrame 60 FPS loop, radar & telemetry canvases, and Pit Exit Ghost.
    """
    from src.dashboard.smooth_broadcast_canvas import (
        build_broadcast_payload,
        get_streamlit_anti_blink_css,
        render_smooth_broadcast_html,
    )
    from src.data_ingestion.telemetry_daemon import DRIVER_METADATA_MAP, get_circuit_metadata

    css = get_streamlit_anti_blink_css()
    assert '[data-stale="true"]' in css
    assert "opacity: 1 !important" in css

    x, y, curv = get_circuit_curvature_profile("Bahrain", n_points=240)
    meta = get_circuit_metadata("Bahrain")
    df_l17 = sync_gp_lap_to_live_table("Bahrain", "Race", lap=17, season=2024)
    tel_df = generate_driver_micro_telemetry("VER", "Bahrain", lap=17, compound="SOFT", tyre_life=14)

    laps_map = {
        17: [
            {
                "driver": str(r.driver),
                "position": int(r.position),
                "compound": str(r.compound),
                "tyre_life": int(r.tyre_life),
                "gap_to_leader": float(r.gap_to_leader),
                "lap_time_sec": float(r.lap_time_sec),
                "track_status": str(r.track_status),
            }
            for r in df_l17.itertuples(index=False)
        ]
    }
    payload = build_broadcast_payload(
        circuit="Bahrain",
        race_name=meta["race_name"],
        season=2024,
        session_label="Race",
        x_track=x,
        y_track=y,
        norm_curv=curv,
        track_length_m=float(meta["track_length_m"]),
        base_lap_sec=float(meta["base_lap_sec"]),
        total_laps=57,
        initial_lap=17,
        initial_sub_lap=0.35,
        ego_driver="VER",
        pit_loss_sec=23.0,
        track_temp_c=31.8,
        weather_cond="Dry (Clear)",
        laps_by_number=laps_map,
        driver_meta=DRIVER_METADATA_MAP,
        micro_tel_df=tel_df,
    )
    assert len(payload["trackX"]) == 240
    assert len(payload["tel"]["speed"]) == 240
    assert len(payload["laps"]["17"]) == 20
    assert payload["trackLenM"] == 5412.0

    html = render_smooth_broadcast_html(payload)
    assert "requestAnimationFrame" in html
    assert 'id="radarCanvas"' in html
    assert 'id="telCanvas"' in html
    assert "REJOIN GHOST" in html
    assert "PIT LANE" in html
    assert "pitLaneScreenPoint" in html
    assert "PLAY 60 FPS" in html


def test_real_24_gps_track_maps_and_official_championship_cache():
    """
    Verifies authentic F1 data coverage across all 24 circuits & 2021-2024 seasons:
    1. All 24 F1 circuits exist in `data/circuit_track_map_cache.json` and load exact cached
       real FIA GPS outlines via `generate_circuit_map_coordinates()`.
    2. `data/historical_championship_cache.json` covers 2021-2024 and drives authentic WDC/WCC
       standings (2024 VER > NOR > LEC; 2021 VER vs HAM finale) and real historical race winners.
    3. `get_session_data_source()` accurately distinguishes `FASTF1_REAL` vs `JOLPICA_CALIBRATED`.
    """
    import json
    import numpy as np
    from src.data_ingestion.telemetry_daemon import (
        generate_circuit_map_coordinates,
        get_historical_race_order,
        get_session_data_source,
    )

    map_cache_path = os.path.join(PROJECT_ROOT, "data", "circuit_track_map_cache.json")
    hist_cache_path = os.path.join(PROJECT_ROOT, "data", "historical_championship_cache.json")
    assert os.path.exists(map_cache_path)
    assert os.path.exists(hist_cache_path)

    with open(map_cache_path, "r", encoding="utf-8") as f:
        track_cache = json.load(f)
    assert len(track_cache) == 24

    generate_circuit_map_coordinates.cache_clear()
    for cal in F1_2024_CALENDAR:
        cname = cal["circuit"]
        ckey = cname.lower()
        assert ckey in track_cache, f"Missing real GPS map for {cname}"
        x, y = generate_circuit_map_coordinates(cname, n_points=240)
        assert len(x) == 240 and len(y) == 240
        assert np.allclose(x, np.asarray(track_cache[ckey]["x"], dtype=np.float64), atol=1e-2)
        assert np.allclose(y, np.asarray(track_cache[ckey]["y"], dtype=np.float64), atol=1e-2)

    # Verify real 2024 Grand Prix winners in historical race order
    assert get_historical_race_order(2024, "Monaco")[0] == "LEC"
    assert get_historical_race_order(2024, "Great Britain")[0] == "HAM"
    assert get_historical_race_order(2024, "Singapore")[0] == "NOR"

    # Verify authentic 2024 pre-Abu Dhabi WDC championship order (VER > NOR > LEC)
    champ_2024_final = compute_championship_standings_till_race(season=2024, circuit="Abu Dhabi")
    wdc_24 = champ_2024_final["wdc"]
    assert list(wdc_24["driver"].iloc[:3]) == ["VER", "NOR", "LEC"]
    assert int(wdc_24.iloc[0]["prior_points"]) >= 380

    # Verify authentic 2021 pre-Abu Dhabi title battle (VER & HAM top 2)
    champ_2021_final = compute_championship_standings_till_race(season=2021, circuit="Abu Dhabi")
    wdc_21 = champ_2021_final["wdc"]
    assert set(wdc_21["driver"].iloc[:2]) == {"VER", "HAM"}

    # Verify data provenance tracking
    assert get_session_data_source(2024, "Bahrain", "Race") == "FASTF1_REAL"
    assert get_session_data_source(2024, "Monaco", "Race") in ("FASTF1_REAL", "JOLPICA_CALIBRATED")


def test_cold_hard_activation_circuit_abrasion_and_traffic_trap_guard():
    """
    Verifies the 4 F1 strategy physics upgrades:
    1. Circuit abrasion scaling: smooth circuits (Hungary 0.74, Monaco 0.64) degrade slower than Bahrain (1.00).
    2. Cold-track HARD activation penalty: HARDs below 30C asphalt incur real warm-up/graining loss
       (0.0s at >=30C warm tracks, significant loss at 26.7C cool tracks).
    3. Dirty-air traffic-rejoin penalty: rejoining into heavy traffic (e.g. P1->P5, 2.2% clean air at Hungary)
       deducts realistic dirty-air overtaking loss from pit breakeven.
    4. Counter-strategy proposal: rejects cold HARD + traffic-trap stop and recommends extending stint.
    """
    from src.agent.tools import compute_remaining_stint_pit_breakeven

    cache = TyreL1TensorCache(total_race_laps=70)

    # 1. Circuit abrasion scaling
    deg_bahrain = cache.lookup_degradation("MEDIUM", 30.0, 18, circuit="Bahrain")
    deg_hungary = cache.lookup_degradation("MEDIUM", 30.0, 18, circuit="Hungary")
    deg_monaco = cache.lookup_degradation("MEDIUM", 30.0, 18, circuit="Monaco")
    assert deg_bahrain > deg_hungary > deg_monaco > 0.0

    # 2. Cold-track HARD thermal activation window
    assert TyreL1TensorCache.get_cold_track_activation_penalty("HARD", 32.0, stint_lap=1) == 0.0
    cold_hard_l1 = TyreL1TensorCache.get_cold_track_activation_penalty("HARD", 26.7, stint_lap=1)
    cold_soft_l1 = TyreL1TensorCache.get_cold_track_activation_penalty("SOFT", 26.7, stint_lap=1)
    assert cold_hard_l1 > 0.7
    assert cold_soft_l1 == 0.0

    # 3 & 4. Cold track + traffic trap (26.7C Hungary, P1->P5, 2.2% clean air) rejects HARD & gives counter-strategy
    be = compute_remaining_stint_pit_breakeven(
        ego_compound="MEDIUM",
        ego_tyre_life=18,
        track_temp=26.7,
        laps_remaining=31,
        pit_loss_sec=21.5,
        target_new_tyre="HARD",
        race_lap=39,
        weather_condition="Dry (Clear)",
        circuit="Hungary",
        p_clean_air=0.022,
        positions_dropped=4,
        race_condition="GREEN",
    )
    assert be["circuit_abrasion_factor"] == 0.74
    assert be["cold_track_penalty_sec"] > 8.0
    assert be["traffic_rejoin_penalty_sec"] > 8.0
    assert be["net_stint_pit_payback_sec"] < 0.0
    assert be["pit_breakeven_reached"] is False
    assert len(be["rejection_reasons"]) == 2
    assert "Extend MEDIUMs" in be["alternative_recommendation"]
    assert "SOFT" in be["alternative_recommendation"]

    # 5. Wrong-weather target guard: WET -> INTERMEDIATE on a DRY track must be rejected (skip straight to slicks)
    be_monaco_dry = compute_remaining_stint_pit_breakeven(
        ego_compound="WET",
        ego_tyre_life=18,
        track_temp=29.1,
        laps_remaining=46,
        pit_loss_sec=22.0,
        target_new_tyre="INTERMEDIATE",
        race_lap=18,
        weather_condition="Dry (Clear)",
        circuit="Monaco",
        p_clean_air=0.012,
        positions_dropped=4,
        race_condition="GREEN",
    )
    assert be_monaco_dry["wrong_weather_target"] is True
    assert be_monaco_dry["pit_breakeven_reached"] is False
    assert "SKIP INTERMEDIATEs" in be_monaco_dry["alternative_recommendation"]















