"""FastAPI REST API routes (`/api/*`) for the F1 Virtual Race Engineer Web App."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.web.services import (
    FAMOUS_SCENARIOS,
    ask_pit_wall_radio,
    fetch_real_fia_session_for_web,
    get_championship_bundle,
    get_race_session_bundle,
    get_season_calendar,
    get_system_health,
    run_sandbox_simulation,
    search_fia_rules,
)

router = APIRouter(prefix="/api", tags=["Pit Wall API"])


class RadioAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=600)
    season: int = 2024
    circuit: str = "Bahrain"
    session_type: str = "Race"
    lap: int = 17
    ego_driver: str = "VER"
    weather: str | None = None
    track_status: str | None = None
    track_temp: float | None = None
    pit_box_num: int = 1
    custom_rows: list[dict[str, Any]] | None = None
    total_laps_override: int | None = None


class SandboxRow(BaseModel):
    driver: str
    position: int
    compound: str = "MEDIUM"
    tyre_life: int = 10
    gap_to_leader: float = 0.0
    lap_time_sec: float = 90.0


class SandboxSimRequest(BaseModel):
    circuit: str = "Hungary"
    lap: int = 39
    total_laps: int = 70
    ego_driver: str = "LEC"
    track_status: str = "GREEN"
    track_temp: float = 26.7
    weather: str = "Overcast (Cool)"
    target_tyre: str = "HARD"
    pit_box_num: int = 2
    rows: list[SandboxRow]


class RulesSearchRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


@router.get("/health")
def api_health() -> dict[str, Any]:
    return get_system_health()


@router.get("/calendar/{season}")
def api_calendar(season: int) -> dict[str, Any]:
    return {"season": season, "races": get_season_calendar(season)}


@router.get("/scenarios")
def api_scenarios() -> dict[str, Any]:
    return {"scenarios": FAMOUS_SCENARIOS}


@router.get("/race/{season}/{circuit}/session")
def api_race_session(
    season: int,
    circuit: str,
    session_type: str = Query(default="Race"),
    lap: int | None = Query(default=None),
    ego_driver: str = Query(default="VER"),
    target_tyre: str = Query(default="HARD"),
    weather: str | None = Query(default=None),
    track_status: str | None = Query(default=None),
    track_temp: float | None = Query(default=None),
    pit_box_num: int = Query(default=1, ge=1, le=10),
    include_canvas_html: bool = Query(default=True),
) -> dict[str, Any]:
    try:
        return get_race_session_bundle(
            season=season,
            circuit=circuit,
            session_type=session_type,
            lap=lap,
            ego_driver=ego_driver,
            target_tyre=target_tyre,
            weather_override=weather,
            status_override=track_status,
            track_temp_override=track_temp,
            pit_box_num=pit_box_num,
            include_canvas_html=include_canvas_html,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/race/{season}/{circuit}/fetch-fia")
def api_fetch_fia(
    season: int,
    circuit: str,
    session_type: str = Query(default="Race"),
) -> dict[str, Any]:
    return fetch_real_fia_session_for_web(season=season, circuit=circuit, session_type=session_type)


@router.post("/radio/ask")
def api_radio_ask(req: RadioAskRequest) -> dict[str, Any]:
    return ask_pit_wall_radio(
        question=req.question,
        season=req.season,
        circuit=req.circuit,
        session_type=req.session_type,
        lap=req.lap,
        ego_driver=req.ego_driver,
        weather=req.weather,
        track_status=req.track_status,
        track_temp=req.track_temp,
        pit_box_num=req.pit_box_num,
        custom_rows=req.custom_rows,
        total_laps_override=req.total_laps_override,
    )


@router.post("/sandbox/simulate")
def api_sandbox_simulate(req: SandboxSimRequest) -> dict[str, Any]:
    return run_sandbox_simulation(
        circuit=req.circuit,
        lap=req.lap,
        total_laps=req.total_laps,
        ego_driver=req.ego_driver,
        track_status=req.track_status,
        track_temp=req.track_temp,
        weather=req.weather,
        target_tyre=req.target_tyre,
        pit_box_num=req.pit_box_num,
        rows=[r.model_dump() for r in req.rows],
    )


@router.get("/standings/{season}/{circuit}")
def api_standings(season: int, circuit: str) -> dict[str, Any]:
    return get_championship_bundle(season=season, circuit=circuit)


@router.post("/rules/search")
def api_rules_search(req: RulesSearchRequest) -> dict[str, Any]:
    return search_fia_rules(question=req.question, top_k=req.top_k)
