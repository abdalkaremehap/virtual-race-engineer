"""End-to-end tests for the Multi-Page FastAPI F1 Pit Wall Web Application & REST API."""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_engine.ingest import chunk_fia_pdf
from src.web.app import app
from src.web.services import FAMOUS_SCENARIOS

client = TestClient(app)


def test_web_all_five_multipage_routes_return_200():
    routes = [
        ("/", "Formula 1 Virtual Race Engineer"),
        ("/?season=2022", "2022 World Championship Calendar"),
        ("/race/2024/Bahrain", "PIT WALL RADIO COMMS"),
        ("/race/2022/Hungary?lap=39&driver=LEC", "broadcastFrame"),
        ("/sandbox?preset=hungary_2022_l39", "What-If F1 Strategy Scenario Lab"),
        ("/standings", "FIA World Championship Standings"),
        ("/standings/2024/Monaco", "Monaco"),
        ("/rules", "FIA 2024 Sporting Regulations Explorer"),
    ]
    for path, marker in routes:
        resp = client.get(path)
        assert resp.status_code == 200, f"GET {path} failed with {resp.status_code}"
        assert marker in resp.text, f"Expected '{marker}' in HTML response for {path}"


def test_web_api_health_calendar_and_session():
    h = client.get("/api/health")
    assert h.status_code == 200
    hd = h.json()
    assert hd["status"] == "ok"
    assert hd["l1_tensor_kb"] == 56.86
    assert hd["circuits_count"] == 24

    cal = client.get("/api/calendar/2024")
    assert cal.status_code == 200
    races = cal.json()["races"]
    assert len(races) == 24
    assert all(r["svg_path"].startswith("M ") for r in races)

    sess = client.get("/api/race/2024/Bahrain/session?lap=17&ego_driver=VER&include_canvas_html=false")
    assert sess.status_code == 200
    sd = sess.json()
    assert sd["circuit"] == "Bahrain"
    assert sd["ego_driver"] == "VER"
    assert sd["mc_rejoin"]["median_pos"] >= 1
    assert len(sd["tyre_curves"]["curves"]["SOFT"]) >= 10
    assert "laps" in sd["broadcast_payload"]


def test_web_api_radio_ask_pit_battle_legal_and_sandbox():
    # 1. Pit timing question
    r_pit = client.post(
        "/api/radio/ask",
        json={"question": "do i pit now or we should wait", "season": 2024, "circuit": "Bahrain", "lap": 17, "ego_driver": "VER"},
    )
    assert r_pit.status_code == 200
    d_pit = r_pit.json()
    assert d_pit["guardrail_passed"] is True
    assert d_pit["human_radio_call"]
    assert d_pit["spoken_text"]

    # 2. Rival battle question must NOT repeat pit call and must mention Checo
    r_bat = client.post(
        "/api/radio/ask",
        json={"question": "can PER catch me ?", "season": 2024, "circuit": "Bahrain", "lap": 17, "ego_driver": "VER"},
    )
    assert r_bat.status_code == 200
    d_bat = r_bat.json()
    assert d_bat["guardrail_passed"] is True
    assert "Checo" in d_bat["human_radio_call"]
    assert d_bat["human_radio_call"] != d_pit["human_radio_call"]
    assert "BATTLE" in d_bat["verdict_label"]

    # 3. Pure FIA legal question must isolate FIA RAG
    r_leg = client.post(
        "/api/radio/ask",
        json={"question": "can i overtake under the sc ?", "season": 2024, "circuit": "Bahrain", "lap": 17, "ego_driver": "LEC"},
    )
    assert r_leg.status_code == 200
    d_leg = r_leg.json()
    assert d_leg["guardrail_passed"] is True
    assert "consult_fia_rulebook_multihop" in d_leg["required_tools"]
    assert "analyze_rival_catch_battle" not in d_leg["required_tools"]
    assert "simulate_pit_stop_mc" not in d_leg["required_tools"]

    # 4. Sandbox custom Hungary 2022 L39 cold-Hards trap
    hun = next(s for s in FAMOUS_SCENARIOS if s["id"] == "hungary_2022_l39")
    r_sb_sim = client.post(
        "/api/sandbox/simulate",
        json={
            "circuit": hun["circuit"],
            "lap": hun["lap"],
            "total_laps": hun["total_laps"],
            "ego_driver": hun["ego_driver"],
            "track_status": hun["track_status"],
            "track_temp": hun["track_temp"],
            "weather": hun["weather"],
            "target_tyre": hun["target_tyre"],
            "rows": hun["rows"],
        },
    )
    assert r_sb_sim.status_code == 200
    assert "canvas" in r_sb_sim.json()["canvas_html"].lower()

    r_sb_ask = client.post(
        "/api/radio/ask",
        json={
            "question": hun["suggested_question"],
            "circuit": hun["circuit"],
            "lap": hun["lap"],
            "ego_driver": hun["ego_driver"],
            "track_status": hun["track_status"],
            "track_temp": hun["track_temp"],
            "weather": hun["weather"],
            "custom_rows": hun["rows"],
            "total_laps_override": hun["total_laps"],
        },
    )
    assert r_sb_ask.status_code == 200
    assert "stay out" in r_sb_ask.json()["human_radio_call"].lower()


def test_web_api_standings_rules_and_pdf_chunker():
    st_resp = client.get("/api/standings/2024/Monaco")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert len(st_data["wdc"]) == 20
    assert len(st_data["wcc"]) == 10

    rules_resp = client.post("/api/rules/search", json={"question": "can i overtake under the sc ?", "top_k": 2})
    assert rules_resp.status_code == 200
    rd = rules_resp.json()
    assert rd["offline"] is True or len(rd["hop1_primary_rules"]) >= 1

    # Verify FIA PDF chunker ready for single-container embedded Qdrant indexing
    chunks = chunk_fia_pdf()
    assert len(chunks) > 100
    assert all("page" in c and "text" in c for c in chunks[:5])
