"""
Main FastAPI Multi-Page F1 TV Pro Web Application.

Routes:
- GET /                                  -> Season Hub & 24-Circuit Calendar
- GET /race/{season}/{circuit}           -> Live 60 FPS Pit Wall Command Center + Radio Comms
- GET /sandbox                           -> Custom What-If Scenario Lab
- GET /standings                         -> WDC & WCC Championship Center (defaults to 2024 Abu Dhabi)
- GET /standings/{season}/{circuit}      -> WDC & WCC Championship Center up to a specific GP
- GET /rules                             -> 2-Hop FIA Sporting Regulations Explorer
- /api/*                                 -> High-speed JSON REST API (~20-40ms radio calls)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data_ingestion.telemetry_daemon import (
    DRIVER_METADATA_MAP,
    F1_2024_CALENDAR,
    SESSION_DISPLAY_LABELS,
    SESSION_TYPES,
    SUPPORTED_SEASONS,
)
from src.web.api import router as api_router
from src.web.services import (
    FAMOUS_SCENARIOS,
    WEATHER_OPTIONS,
    PitWallWebState,
    get_championship_bundle,
    get_race_session_bundle,
    get_season_calendar,
    get_system_health,
)

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")
STATIC_DIR = os.path.join(WEB_DIR, "static")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state = PitWallWebState.get()
    state.start_rag_prewarm()
    yield


app = FastAPI(
    title="F1 Virtual Race Engineer — Pit Wall Command Center",
    description="Multi-Page F1 Pit Wall Strategy, 60 FPS Telemetry Radar, 10k Monte Carlo Simulator & 2-Hop FIA Rulebook AI.",
    version="2.0.0",
    lifespan=lifespan,
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(api_router)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _common_ctx(request: Request, active_nav: str, page_title: str) -> dict:
    return {
        "request": request,
        "active_nav": active_nav,
        "page_title": page_title,
        "seasons": list(SUPPORTED_SEASONS),
        "calendar": F1_2024_CALENDAR,
        "session_types": list(SESSION_TYPES),
        "session_labels": SESSION_DISPLAY_LABELS,
        "weather_options": WEATHER_OPTIONS,
        "health": get_system_health(),
    }


@app.get("/", response_class=HTMLResponse)
def page_home(request: Request, season: int = Query(default=2024)):
    season_i = season if season in SUPPORTED_SEASONS else 2024
    races = get_season_calendar(season_i)
    ctx = _common_ctx(request, "home", "F1 Pit Wall — Season Hub")
    ctx.update({
        "selected_season": season_i,
        "races": races,
        "famous_scenarios": FAMOUS_SCENARIOS,
    })
    return templates.TemplateResponse(request, "home.html", ctx)


@app.get("/race/{season}/{circuit}", response_class=HTMLResponse)
def page_race(
    request: Request,
    season: int,
    circuit: str,
    session: str = Query(default="Race"),
    lap: int | None = Query(default=None),
    driver: str = Query(default="VER"),
    target_tyre: str = Query(default="HARD"),
    weather: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    bundle = get_race_session_bundle(
        season=season,
        circuit=circuit,
        session_type=session,
        lap=lap,
        ego_driver=driver,
        target_tyre=target_tyre,
        weather_override=weather,
        status_override=status,
        include_canvas_html=True,
    )
    ctx = _common_ctx(request, "race", f"{bundle['season']} {bundle['race_name']} — Pit Wall")
    ctx.update({
        "bundle": bundle,
        "bundle_json": json.dumps({
            k: v for k, v in bundle.items() if k not in ("canvas_html", "broadcast_payload")
        }),
        "driver_meta": DRIVER_METADATA_MAP,
    })
    return templates.TemplateResponse(request, "race.html", ctx)


@app.get("/sandbox", response_class=HTMLResponse)
def page_sandbox(request: Request, preset: str = Query(default="hungary_2022_l39")):
    chosen = next((s for s in FAMOUS_SCENARIOS if s["id"] == preset), FAMOUS_SCENARIOS[0])
    ctx = _common_ctx(request, "sandbox", "F1 Pit Wall — What-If Scenario Lab")
    ctx.update({
        "presets": FAMOUS_SCENARIOS,
        "active_preset": chosen,
        "presets_json": json.dumps(FAMOUS_SCENARIOS),
        "all_drivers": sorted(DRIVER_METADATA_MAP.keys()),
    })
    return templates.TemplateResponse(request, "sandbox.html", ctx)


@app.get("/standings", response_class=HTMLResponse)
def page_standings_default(request: Request, season: int = Query(default=2024), circuit: str = Query(default="Abu Dhabi")):
    return page_standings(request=request, season=season, circuit=circuit)


@app.get("/standings/{season}/{circuit}", response_class=HTMLResponse)
def page_standings(request: Request, season: int, circuit: str):
    data = get_championship_bundle(season=season, circuit=circuit)
    ctx = _common_ctx(request, "standings", f"{data['season']} F1 Championship Standings")
    ctx.update({
        "standings": data,
        "standings_json": json.dumps(data),
    })
    return templates.TemplateResponse(request, "standings.html", ctx)


@app.get("/rules", response_class=HTMLResponse)
def page_rules(request: Request, q: str = Query(default="")):
    ctx = _common_ctx(request, "rules", "FIA 2024 Sporting Regulations — 2-Hop Explorer")
    ctx.update({"initial_query": q})
    return templates.TemplateResponse(request, "rules.html", ctx)
