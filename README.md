# 🏎️ F1 Virtual Race Engineer — Autonomous Pit Wall Decision Support System

> **Real-time Formula 1 Pit Wall telemetry, 10,000-simulation Monte Carlo strategy optimization, 2-Hop FIA Sporting Regulations RAG, and spoken Team Radio AI — served through a 5-page 60 FPS FastAPI web application.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Multi--Page%20Web%20App-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Pytest-47%2F47%20Passing-00e676)](#running-the-test-suite)
[![Docker Ready](https://img.shields.io/badge/Docker-Single%20Command-2496ED?logo=docker&logoColor=white)](#option-2--run-with-docker-1-command-no-python-setup)

<!-- Tip: Drop your screen-recording GIF or LinkedIn video thumbnail right here! -->

---

## ✨ Key Engineering Highlights

| Subsystem | What It Does | Performance / Specs |
| :--- | :--- | :--- |
| **60 FPS Broadcast Stage** | High-DPI HTML5 Canvas track radar with all **24 real FIA GPS circuit maps**, dedicated pit lane bypass, pit-exit ghost marker, steering-wheel RPM/THR/BRK/DRS HUD, and clickable Timing Tower | Smooth `60 FPS` client-side (`requestAnimationFrame`), zero flicker |
| **L1 Tyre Physics Cache** | Pre-computed LightGBM degradation tensor `(5 compounds × 41 track temps × 71 laps)` with circuit abrasion scaling, cold-track HARD activation penalty (`<30°C`), and wet-weather crossover physics | **56.86 KB** L1 cache, `<0.01 ms` lookup |
| **10k Monte Carlo Pit Simulator** | Vectorized NumPy SIMD pit-rejoin, clean-air probability, undercut/overcut, lapped traffic, and teammate double-stack queueing simulation | **10,000 runs in ~5–10 ms** |
| **Multi-Option Strategy Optimizer** | Ranks `STAY_OUT`, `BOX_NOW` (all compounds), and `EXTEND_AND_BOX (+1/+4/+8/+12L)` with FIA mandatory 2-compound dry rule & free Safety Car stop physics | Validated on **20 historical F1 scenarios** (Hungary 2022, Silverstone 2022, Monaco 2022, Abu Dhabi 2021, Sochi 2021, etc.) |
| **LangGraph Race Engineer + Guardrails** | Word-boundary intent router, deterministic hallucination guardrails, natural **Human F1 Pit Wall Voice** (Web Speech API + synthesized F1 dual-tone radio beep), and instant `0 ms` **More Clarification** proof drawer | **~20–40 ms** per radio call |
| **2-Hop FIA Rulebook RAG** | Semantic vector search (`all-MiniLM-L6-v2` + Qdrant) over the official **2024 FIA Sporting Regulations PDF** + automatic Hop-2 cross-reference resolution (e.g., Article `54.3` Penalties) | Thread-safe LRU cache + automatic embedded local Qdrant (works without Docker!) |
| **Dual Real FIA Telemetry Backend** | Real lap timing, stints & Race Control flags via **OpenF1** (`2023–2024`) and **FastF1** (`2021–2022`), cached permanently in **DuckDB**, plus official **2021–2024 Jolpica WDC/WCC Standings** | Auto-fetches on first click, instant thereafter |

---

## 🖥️ Multi-Page Web App Routes

- **`/` — Season Hub & 24-Circuit Calendar:** Switch `2021–2024`, filter circuits with real FIA GPS SVG silhouettes, or jump straight into famous F1 strategy stress tests.
- **`/race/{season}/{circuit}`** *(e.g. `/race/2022/Hungary?lap=39&driver=LEC`)* — **Live 60 FPS Pit Wall Command Center** with split-screen radar + Pit Lane, docked spoken Radio Comms, live telemetry KPIs, interactive Plotly.js Tyre Degradation & 10k MC Pit Rejoin charts, and shareable deep-link URLs.
- **`/sandbox`** — **What-If Scenario Lab** with 1-click historical presets (*Hungary 2022 L39 Cold Hards*, *Silverstone 2022 L39 SC*, *Monaco 2022 L18 Skip Intermediates*, *Abu Dhabi 2021 L53*), editable 20-car grid, and live strategy testing.
- **`/standings`** — **Official WDC & WCC Championship Tracker** (`2021–2024`) with round selector and interactive charts.
- **`/rules`** — **2-Hop FIA Sporting Regulations Explorer** with spoken Pit Wall verdicts, Hop-1 matched rules, and Hop-2 auto-resolved articles.

---

## 🚀 Quick Start

### Option 1 — Run Locally with `uv` (Recommended, No Docker Needed!)

**Prerequisites:** Python `3.11` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# 1. Clone the repository
git clone https://github.com/abdalkaremehap/virtual-race-engineer.git
cd virtual-race-engineer

# 2. Install all dependencies into an isolated .venv
uv sync --all-groups

# 3. Launch the Multi-Page FastAPI Pit Wall Web App
uv run uvicorn src.web.app:app --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860** in Chrome or Microsoft Edge (turn your sound on for the F1 Pit Wall radio beep + UK engineer voice!).

> **Zero external database required:** On first launch, if Docker Qdrant isn't running on port `6333`, the app automatically creates a local embedded Qdrant index (`data/qdrant_local/`) from `data/fia_2024_sporting_regulations.pdf`.

---

### Option 2 — Run with Docker (1 Command, No Python Setup!)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/abdalkaremehap/virtual-race-engineer.git
cd virtual-race-engineer
docker compose up --build
```

Open **http://localhost:7860** — Docker Compose automatically builds the app from `Dockerfile` and connects it to the `qdrant/qdrant` vector database service (`port 6333`).

Or run the **single standalone container** (using built-in embedded Qdrant):
```bash
docker build -t virtual-race-engineer .
docker run -p 7860:7860 virtual-race-engineer
```

---

### Option 3 — Legacy Streamlit Dashboard
The original single-page Streamlit cockpit is also preserved in the repository:
```bash
uv run streamlit run src/dashboard/pit_wall_app.py
```

---

## 🧪 Running the Test Suite

All **47 automated tests** cover the 5-page FastAPI web app & REST API, 21 historical F1 strategy stress-tests, 56.86 KB L1 tensor monotonicity, 10k MC latency, pedal-physics invariants across all 24 circuits, 2-Hop FIA RAG, and hallucination guardrails:

```bash
uv run pytest test/test_web_app.py test/test_f1_strategy_benchmark.py test/test_system_e2e.py -v
```

---

## 📁 Project Structure

```text
virtual-race-engineer/
├── src/
│   ├── web/                     # FastAPI multi-page web app, REST API (/api/*), templates & JS
│   │   ├── app.py               # 5 HTML routes (/, /race/..., /sandbox, /standings, /rules)
│   │   ├── api.py               # JSON REST endpoints (~20-40ms radio calls)
│   │   ├── services.py          # Shared Pit Wall backend orchestration
│   │   ├── templates/           # Jinja2 F1 TV Pro dark-mode pages
│   │   └── static/              # styles.css & zero-reload radio_client.js (Web Speech + F1 beep)
│   ├── agent/                   # LangGraph Race Engineer, intent router, guardrails & tools
│   ├── ml_surrogates/           # 56.86 KB L1 Tyre Tensor Cache, 10k MC Pit Sim & Strategy Optimizer
│   ├── rag_engine/              # 2-Hop FIA Sporting Regulations RAG + embedded local indexer
│   ├── data_ingestion/          # DuckDB telemetry store, OpenF1/FastF1 clients, 24 GPS maps & Jolpica
│   └── dashboard/               # 60 FPS HTML5 Canvas renderer, radio voice helpers & legacy Streamlit UI
├── data/                        # Pre-warmed DuckDB sessions, 24 GPS maps, WDC/WCC cache & 2024 FIA PDF
├── models/                      # Trained LightGBM tyre degradation model & encoders
├── config/                      # 24-circuit pit loss, abrasion & overtaking difficulty parameters
├── test/                        # 47 E2E, strategy benchmark, and FastAPI web tests
├── Dockerfile                   # Production single-container image (uv + FastAPI + embedded Qdrant)
├── docker-compose.yml           # Full multi-service stack (FastAPI app + Qdrant server)
└── DEPLOYMENT.md                # Cloud deployment guide (Docker / Railway / Render / VPS)
```
