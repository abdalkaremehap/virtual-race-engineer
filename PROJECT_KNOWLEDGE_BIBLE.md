# 🏎️ Virtual Race Engineer — Master Technical Knowledge Bible & Agent Handoff Specification

> **Purpose of This Document:**  
> This document is a complete, line-by-line architectural, mathematical, and operational knowledge transfer for **Virtual Race Engineer (`D:\virtual-race-engineer`)**. It is designed to be fed directly into a new Antigravity AI session so the receiving agent immediately possesses 100% context of every file, class, function signature, database schema, ML model, vector database configuration, mathematical equation, historical bug fix, and Windows/PyCharm environment quirk.

---

## 1. Executive Summary & System Capabilities

**Virtual Race Engineer** is an autonomous, real-time Formula 1 Pit-Wall Decision Support System and Multi-Hop Legal Reasoning Engine. It combines:
1. **Multi-Season & Multi-Session Telemetry Pipelines (Phases 0–1):** FastF1 (`SQLite` HTTP cache), Jolpica/Ergast (`race_calendar`), OpenF1 (`/v1/sessions`, `/v1/race_control`, `/v1/team_radio`), and a thread-safe **DuckDB Single-Writer / In-Memory Mirror Daemon** supporting **4 Seasons (`2021, 2022, 2023, 2024`)**, all **24 Grand Prix Circuits**, and all **6 Weekend Session Types (`Practice 1`, `Practice 2`, `Practice 3`, `Qualifying`, `Sprint`, `Race`)**.
2. **Leak-Free Monotonic LightGBM Physics Surrogate + 56.86 KB L1 3D Tensor Cache (Phases 2 & 4):** A 3,000-tree `LightGBM` regression model (`models/tyre_deg_model.txt`) trained on fuel-corrected relative degradation deltas with `GroupShuffleSplit` by stint and monotonic constraints (`+1`) on all tyre-age polynomials, achieving **`0.340s` true out-of-sample MAE**. Pre-evaluated into a `(5 Compounds × 41 TrackTemps × 71 TyreLife)` `float32` L1 hardware-cached 3D tensor (`56.86 KB`) for **`~7.3 µs` single-car** and **`~12.74 µs` full 20-car grid** vectorized lookups.
3. **SIMD Vectorized 10,000-Run Monte Carlo Kinematic Pit Simulator (Phases 2 & 4):** Evaluates `10,000` stochastic pit-stop and traffic rejoin worlds in **`~1.12 ms`** (`27,000x–42,000x` faster than sequential Python loops) using 2D NumPy broadcasting (`(N, 1)` vs `(1, M)`), factoring in stationary Gaussian stop variance (`mean=2.4s, std=0.2s, min=1.8s`), a `2%` botched-stop distribution (`+2.5s..6.0s`), pit-box fast-lane traffic hold probabilities (`5% × pit_box_position` under `SC/VSC`), kinematic reference-frame bypass time reductions (`VSC: t_track / 0.65`, `SC: t_track / 0.52`), and cold-tyre warmup penalties (`SOFT: 0.5s, MEDIUM: 1.2s, HARD: 2.2s`).
4. **2-Hop Qdrant Legal RAG Engine over FIA Sporting Regulations (Phases 3 & 5):** Ingests `data/fia_2024_sporting_regulations.pdf` (106 pages) using a **150-word Overlapping Sliding Window (30-word overlap)** to stay strictly beneath `all-MiniLM-L6-v2`'s `256`-token sequence truncation limit (eliminating a 60.62% silent token loss bug). Includes an L1 `@lru_cache` (`~250 ns` warm lookup) and a **2-Hop Cross-Reference Resolver** (`MultiHopFIARulebook`) that regex-extracts cited articles (e.g., `"under Article 54.3a"`) from Hop-1 chunks and locates their exact section-header definitions (e.g., `54.3 The stewards may impose...` on Page 59) in Hop 2.
5. **LangGraph Autonomous Race Engineer Brain with Remaining-Stint Breakeven Physics & Hallucination Guardrails (Phase 5):** A 4-node `StateGraph` (`router -> execute_tools -> guardrail -> synthesize_radio`) with English homograph disambiguation (`"per lap"` vs `PER`, `"max speed"` vs `VER`, `"on the gas"` vs `GAS`), rival catch/defend pace projection (`analyze_rival_catch_battle`), **Net Remaining-Stint Pit Payback math** (`compute_remaining_stint_pit_breakeven` comparing cumulative remaining fresh-tyre gain over `laps_remaining` against `median_pit_loss` so late-race stops that lose net time/positions are never recommended), and deterministic numerical/citation hallucination guardrails.
6. **2-Page Real-Time `4 Hz (250 ms)` Interactive Streamlit + Plotly Pit Wall Command Center (Phases 6–7):**
   - **Page 1 (`📡 Live & Historical Grand Prix Replay`):** Top Video-Stream Broadcast Controller (`▶️ PLAY / ⏸️ PAUSE`, `⏮️ -1 Lap`, `⏭️ +1 Lap`, `1x / 5x / 15x / 30x` speed, `+25% Sector Scrub`, live `S1/S2/S3` & `UTC Race Clock` HUD), `@st.fragment(run_every=0.25)` real-time 2D Circuit Track Radar with **20 smoothly interpolated moving cars**, clickable 20-driver pill bar synchronized with the sidebar, 480-point micro-telemetry traces (`Speed km/h`, `Throttle %`, `Brake %`, `Gear 1–8`, `RPM`, `DRS`) with moving diamond markers, real vs counterfactual team strategy radio cards, Environment Override panel (`GREEN/VSC/SC/RED`, Track Temp, Weather), and cumulative **World Drivers' Championship (WDC) & Constructors' Championship (WCC) Standings Till This Race** with live projected points.
   - **Page 2 (`🧪 100% Custom Scenario Builder`):** Full sandbox mode allowing custom circuit, lap, flag condition, track temperature, grid size (`2–20 cars`), per-position driver order, interval gaps, compounds, tyre ages, base lap times, 1-click tactical presets, `+1 Lap Forward` physics simulation, `4 Hz` video stream playback, WDC/WCC projections, and live LangGraph Race Engineer querying.

---

## 2. Critical Environment, OS & Hardware Quirks (MUST READ FOR ANY AGENT)

1. **Workspace & Python Virtual Environment:**
   - Project Root: `D:\virtual-race-engineer`
   - Python Executable: `D:\virtual-race-engineer\.venv\Scripts\python.exe` (Python `3.11.7`, managed by `uv`).
   - Always anchor file paths relative to `os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))` because running scripts directly inside PyCharm sets the working directory (`CWD`) to the script's subdirectory (e.g., `src/data_ingestion/`).
2. **Docker Containers (Auto-Start Configured):**
   - `vre_vectordb` (`qdrant/qdrant:latest`): Exposes `0.0.0.0:6333->6333/tcp` (HTTP) and `6334` (gRPC). Stores the `fia_rulebook` vector collection persistently in the `qdrant_data` volume.
   - `vre_runtime` (`virtual-race-engineer-vre-runtime`): Containerized runtime environment.
   - *Note:* If `docker ps` reports `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`, Docker Desktop simply needs to be opened on Windows; once opened, `restart: unless-stopped` in `docker-compose.yml` automatically boots `vre_vectordb` and `vre_runtime`.
3. **PyArrow vs. NumPy 1.26.4 Incompatibility (`ZERO PYARROW RULE`):**
   - `pyproject.toml` pins `numpy<2.0` (`1.26.4`), while the installed `pyarrow` wheel in `.venv` was compiled against NumPy 2.0 headers.
   - **CRITICAL RULE:** Never call `st.dataframe()`, `st.table()`, or `st.data_editor()` in Streamlit, and never call `.df()` on a DuckDB cursor (which imports `pyarrow`). Always fetch DuckDB rows via `cur.fetchall()` + `[dict(zip(cols, r)) for r in rows]` and render tables in Streamlit using custom F1 Broadcast HTML tables (`st.markdown(..., unsafe_allow_html=True)`).
4. **DuckDB Same-Process Connection Configuration Rule (`duckdb_connect`):**
   - DuckDB throws `_duckdb.ConnectionException: Can't open a connection to same database file with a different configuration than existing connections` if one call in a Python process opens `duckdb.connect(db_path, read_only=True)` while another thread/call opens `duckdb.connect(db_path)` (`read_only=False`).
   - **CRITICAL RULE:** Always use the thread-safe context manager `duckdb_connect(db_path)` from `src/data_ingestion/telemetry_daemon.py` (which acquires `_DUCKDB_PROCESS_LOCK = threading.RLock()` and opens with a uniform configuration).
5. **Streamlit File Watcher vs. `transformers.models.zoedepth` (`torchvision` Error):**
   - Streamlit's default file watcher crawls `sys.modules` on reruns and triggers lazy submodule imports inside `transformers`, hitting an unguarded `from torchvision.transforms.v2 import functional as tvF` inside `transformers/models/zoedepth/image_processing_zoedepth.py`.
   - **CRITICAL RULE:** Keep `.streamlit/config.toml` with `fileWatcherType = "none"` and `fastReruns = true` so Streamlit never crawls `site-packages/transformers`.
6. **Qdrant Client v1.19+ API:**
   - `QdrantClient.search()` was removed in `qdrant-client>=1.19`. Always use `client.query_points(collection_name=..., query=vector, limit=top_k).points`.

---

## 3. Repository Directory Tree & File Inventory

```text
D:\virtual-race-engineer\
├── .streamlit\
│   └── config.toml                        # Disables Streamlit module watcher (fixes torchvision/zoedepth crash)
├── config\
│   └── circuit_pit_data.json              # 24-circuit kinematic pit-lane loss parameters (t_pit, normal_loss)
├── data\
│   ├── fia_2024_sporting_regulations.pdf  # Official 2024 FIA Formula 1 Sporting Regulations (106 pages)
│   ├── f1_historical.duckdb               # Central DuckDB store (race_calendar, grand_prix_sessions, live_telemetry_laps)
│   ├── circuit_track_map_cache.json       # Pre-extracted 240-point GPS (X, Y) track loop cache (0.33ms load)
│   └── fastf1_cache\                      # FastF1 SQLite HTTP cache (2024 Bahrain GP telemetry & weather)
├── models\
│   ├── tyre_deg_model.txt                 # Trained 3,000-tree monotonic LightGBM model (0.340s MAE)
│   └── encoders.joblib                    # Scikit-learn LabelEncoders for compound and circuit
├── src\
│   ├── data_ingestion\
│   │   ├── fastf1_client.py               # Phase 1: FastF1 session loader with SQLite caching
│   │   ├── jolpica_client.py              # Phase 1: Jolpica/Ergast 2024 calendar fetcher + DuckDB UNIQUE INDEX upsert
│   │   ├── openf1_client.py               # Phase 1: OpenF1 API client (sessions, race_control flags, team_radio)
│   │   ├── rag_ingestion.py               # Phase 3: 150-word sliding window (30-word overlap) PDF chunker -> Qdrant
│   │   └── telemetry_daemon.py            # Phase 4/6/7: DuckDB daemon, 2021-2024 multi-session loader, 2D track GPS, WDC/WCC
│   ├── ml_surrogates\
│   │   ├── tyre_model.py                  # Phase 2: Leak-free physics-informed LightGBM training pipeline
│   │   ├── pit_simulator.py               # Phase 2/4: Kinematic reference-frame & 10,000-run NumPy SIMD Monte Carlo pit sim
│   │   └── tensor_cache.py                # Phase 4: 56.86 KB (5x41x71) float32 L1 Tyre Degradation Tensor Cache
│   ├── rag_engine\
│   │   ├── query.py                       # Phase 3/4: Qdrant query_points wrapper with L1 @lru_cache(maxsize=128)
│   │   └── multi_hop.py                   # Phase 5: 2-Hop FIA Rulebook cross-reference scanner & exact header resolver
│   ├── simulation\
│   │   ├── engine.py                      # Phase 4: Asynchronous (asyncio) 2 Hz telemetry + 1 Hz physics + event RAG loop
│   │   └── live_connector.py              # Phase 7: Live OpenF1 Race Control bridge with automatic replay fallback
│   ├── agent\
│   │   ├── tools.py                       # Phase 5: LangChain @tool functions + rival catch battle + pit breakeven math
│   │   └── race_engineer.py               # Phase 5: LangGraph 4-node state machine + homograph filter + guardrails
│   └── dashboard\
│       └── pit_wall_app.py                # Phase 6/7: 2-Page 4 Hz Streamlit + Plotly Pit Wall Command Center
├── test\
│   └── test_system_e2e.py                 # Phase 7: 10 comprehensive E2E & Streamlit AppTest verification suites
├── Dockerfile                             # Python 3.11-slim + uv multi-layer production container
├── docker-compose.yml                     # Orchestrates vre_runtime and vre_vectordb (Qdrant)
├── pyproject.toml                         # Project dependencies & tool groups (data, ml, agents, dev)
└── README.md                              # Portfolio overview, architecture diagram, and quick-start guide
```

---

## 4. Phase-by-Phase Engineering History & Deep File Reference

---

### PHASE 0: Infrastructure & Docker Stack
#### Files: [Dockerfile](file:///D:/virtual-race-engineer/Dockerfile) & [docker-compose.yml](file:///D:/virtual-race-engineer/docker-compose.yml)
- **`vre_vectordb` (`qdrant/qdrant:latest`):**
  - Container name: `vre_vectordb`
  - Ports: `6333:6333` (HTTP REST API), `6334:6334` (gRPC)
  - Volume: `qdrant_data:/qdrant/storage`
  - Restart policy: `unless-stopped` (automatically starts whenever Docker Desktop boots).
- **`vre_runtime` (`virtual-race-engineer-vre-runtime`):**
  - Built on `python:3.11-slim` with `uv` binary copied from `ghcr.io/astral-sh/uv:latest`.
  - Runs as non-root user `raceengineer` (`UID 1000`) with bind mounts for `./src`, `./data`, and `./models`.

---

### PHASE 1: Real-Time & Historical Data Ingestion Pipelines
#### 1. [src/data_ingestion/fastf1_client.py](file:///D:/virtual-race-engineer/src/data_ingestion/fastf1_client.py)
- **Purpose:** Connects to F1's official timing archives via `fastf1`, storing raw HTTP responses in `data/fastf1_cache/fastf1_http_cache.sqlite`.
- **Memory Optimization Fix:** Loads sessions with `session.load(telemetry=False, weather=True)` when training lap-level ML models so `fastf1` doesn't allocate 4+ GB of high-frequency car telemetry frames unless explicitly needed.

#### 2. [src/data_ingestion/jolpica_client.py](file:///D:/virtual-race-engineer/src/data_ingestion/jolpica_client.py)
- **Purpose:** Queries `https://api.jolpi.ca/ergast/f1/2024/races.json` (the modern Ergast replacement) and persists the 24-race calendar into `data/f1_historical.duckdb`.
- **Idempotent UPSERT Design:**
  - Creates `race_calendar` table with `CREATE UNIQUE INDEX IF NOT EXISTS idx_race ON race_calendar(season, round)` and uses `INSERT OR IGNORE INTO race_calendar SELECT * FROM stage_df` inside a `with duckdb.connect(db_path) as con:` block so the file lock is released immediately after writing.

#### 3. [src/data_ingestion/openf1_client.py](file:///D:/virtual-race-engineer/src/data_ingestion/openf1_client.py)
- **Purpose:** Live timing & Race Control client querying `https://api.openf1.org/v1/`.
- **Functions:**
  - `get_openf1_session_key(year: int, country: str, session_name: str) -> int`: Uses `httpx.Client().get(url, params=params)` for safe URL encoding of multi-word countries (e.g. `"Great Britain"` -> `"Great%20Britain"`).
  - `fetch_race_control_messages(session_key: int) -> pd.DataFrame`: Pulls live FIA flag deployments (`YELLOW`, `VSC`, `SC`, `RED`, `CLEAR`).
  - `fetch_team_radio(session_key: int, driver_number: int) -> pd.DataFrame`: Pulls driver radio audio timestamps and URLs, gracefully handling `404` or empty responses.

---

### PHASE 2: ML Surrogates & Physics Engine
#### 1. [src/ml_surrogates/tyre_model.py](file:///D:/virtual-race-engineer/src/ml_surrogates/tyre_model.py)
- **The Data Leakage Audit & Fix:**
  - *Initial Flaw:* Predicting raw `LapTime` directly leaked track length, driver base car speed, and fuel load into the tyre wear estimate, and random train/test splits leaked consecutive laps from the same stint.
  - *Production Refactor:*
    1. **Fuel Correction:** Computes `fuel_effect = 0.06 * clean_laps['LapNumber']` and `fuel_corrected_lap_time = raw_lap_time + fuel_effect`.
    2. **Target Variable (`relative_degradation`):** Subtracts each `(driver, compound)` pair's 3-fastest-lap baseline (`base_pace`) from `fuel_corrected_lap_time`, isolating pure tyre thermal/mechanical wear in seconds per lap (`+s/lap`).
    3. **Causal Anomaly Filtering:** Uses a strictly backward-looking rolling median (`rolling(3, center=False, min_periods=1).median()`) to filter traffic anomalies (`< 0.15s`) with zero future data leakage.
    4. **23 Physics-Informed Features:**
       - `tyre_life`, `tyre_life_sq`, `tyre_life_log`, `tyre_life_cube`, `tyre_life_sqrt`, `tyre_life_exp` (`exp(tyre_life / 15.0)`)
       - `compound_encoded`, `circuit_encoded`, `stint`, `fresh_tyre`
       - `track_temp`, `air_temp`, `track_temp_rolling`, `air_temp_rolling`
       - `temp_interaction` (`track_temp * air_temp`), `tyre_life_x_track_temp`, `tyre_life_x_fuel`, `fuel_x_track_evolution`, `tyre_life_x_track_evolution`
       - `track_evolution_sqrt`, `track_evolution_log`, `track_evolution_lin`, `fuel_kg_remaining`
    5. **Stint-Grouped Validation & Monotonicity:** Uses `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `stint_group = driver + "_" + stint` and enforces `monotone_constraints = [1, 1, 1, 1, 1, 1] + [0] * 17` across the 6 tyre-life features so wear is strictly non-decreasing with tyre age.
    6. **Verified Accuracy & Serialization:** Achieves **`0.340s` true out-of-sample MAE** and saves `models/tyre_deg_model.txt` and `models/encoders.joblib`.

#### 2. [src/ml_surrogates/pit_simulator.py](file:///D:/virtual-race-engineer/src/ml_surrogates/pit_simulator.py) & [config/circuit_pit_data.json](file:///D:/virtual-race-engineer/config/circuit_pit_data.json)
- **Kinematic Reference Frame Pit-Loss Math:**
  - Loads circuit-specific `t_pit` (total pit-lane traversal time including a 2.4s stationary stop) and `normal_loss` (net green-flag time lost vs racing speed on track) for all 24 F1 circuits from `config/circuit_pit_data.json`.
  - Computes baseline track bypass time at racing speed:
    $$t_{\text{track\_base}} = t_{\text{pit\_base}} - L_{\text{norm\_base}}$$
  - Under **Virtual Safety Car (`VSC`)**, cars on track slow to `65%` of racing pace, increasing track bypass time to $t_{\text{track\_base}} / 0.65$:
    $$L_{\text{VSC}} = t_{\text{pit\_actual}} - \frac{t_{\text{track\_base}}}{0.65}$$
  - Under **Full Safety Car (`SC`)**, cars on track slow to `52%` of racing pace, increasing track bypass time to $t_{\text{track\_base}} / 0.52$:
    $$L_{\text{SC}} = t_{\text{pit\_actual}} - \frac{t_{\text{track\_base}}}{0.52}$$
  - In Bahrain (`t_pit = 25.0s, normal_loss = 23.0s, t_track_base = 2.0s`): Green Flag pit loss is `~23.00s`, VSC pit loss drops to `~15.75s`, and SC pit loss drops to `~10.55s` (a **12.45s "cheap stop" saving**!).
- **Task 4.2 Vectorized SIMD Monte Carlo (`simulate_rejoin_mc` & `calculate_undercut_mc`):**
  - *The Audit Discovery:* Running 10,000 Monte Carlo loops in Python with `logger.info` inside the loop took `10.47s – 30.31s` (`99.4%` logging I/O overhead).
  - *The Fix:* `_sample_pit_loss_vectorized(race_condition, n_sims=10_000)` generates all 10,000 pit-loss samples in one C-level NumPy array call (`self.rng.normal`, `self.rng.random < 0.02` botch mask, and `self.rng.random < 0.05 * team_pit_box_position` unsafe-release traffic hold mask).
  - Uses 2D NumPy broadcasting (`ahead_mask = traffic_gaps[None, :] < new_gaps[:, None]`, shape `(10000, M)`) to compute:
    - `median_pit_loss`, `p95_pit_loss`
    - `expected_pos`, `median_pos`
    - `p_clean_air` (probability of emerging `>= 1.5s` behind the car ahead), `p_dirty_air`
    - Execution latency: **`~1.10 ms`** for 10,000 simulations (**`42,000x` speedup**).

---

### PHASE 3: Legal Memory (FIA Rulebook RAG Engine)
#### 1. [src/data_ingestion/rag_ingestion.py](file:///D:/virtual-race-engineer/src/data_ingestion/rag_ingestion.py)
- **The 60.62% Silent Truncation Bug & Sliding Window Fix:**
  - *Root Cause:* `PyPDF2.PdfReader.pages[i].extract_text()` on `data/fia_2024_sporting_regulations.pdf` does not emit double newlines (`\n\n`). Splitting by `\n\n` produced single-page chunks averaging **639 tokens**. Because `SentenceTransformer('all-MiniLM-L6-v2')` has a hard `max_seq_length = 256` tokens, it silently truncated and discarded the bottom **60.62%** of every page (`41,060` tokens lost)!
  - *The Fix:* Replaced paragraph splitting with a **150-Word Overlapping Sliding Window with a 30-Word Overlap** (`chunk_size = 150, overlap = 30`, stride `120` words). Every chunk is `~190–210` WordPiece tokens—safely under the `256`-token ceiling with zero mid-sentence rule severing.
  - Embeds all chunks into `384`-dimensional cosine vectors and upserts them into Qdrant collection `fia_rulebook`.

#### 2. [src/rag_engine/query.py](file:///D:/virtual-race-engineer/src/rag_engine/query.py)
- **`FIARulebook` Class:**
  - Connects to `http://localhost:6333` (`collection_name = "fia_rulebook"`).
  - Uses `@lru_cache(maxsize=128)` on `query_cached(question: str, top_k: int = 2) -> list[dict]` so repeated Safety Car / VSC queries return in **`~250 nanoseconds`** from RAM instead of `~50 ms` over HTTP.

---

### PHASE 4: The Live Asynchronous Simulation Engine
#### 1. [src/ml_surrogates/tensor_cache.py](file:///D:/virtual-race-engineer/src/ml_surrogates/tensor_cache.py)
- **`TyreL1TensorCache` Class:**
  - At startup, constructs the full Cartesian product grid of:
    - **5 Compounds:** `["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]`
    - **41 Track Temperatures:** `20°C` to `60°C` (`TEMP_MIN = 20, TEMP_MAX = 60`)
    - **71 Tyre Ages:** `0` to `70` laps (`MAX_LAP = 70`)
  - Evaluates all `5 × 41 × 71 = 14,555` states in a single vectorized `LightGBM` batch prediction (`~450–600 ms` at startup) and stores them in a `float32` 3D NumPy array of size **`56.86 KB`** (fits entirely inside the CPU L1/L2 cache).
  - `lookup_degradation(compound, track_temp, tyre_life) -> float`: Uses `np.clip` on temperature and tyre life so out-of-bounds inputs (e.g., `100°C` or `999` laps) never raise an `IndexError`. Latency: **`~7.3 µs`** in Python loop / **`~184 ns`** raw array index.
  - `lookup_grid_vectorized(compounds, track_temp, tyre_lives) -> np.ndarray`: Evaluates all 20 cars on the F1 grid simultaneously in **`~12.74 microseconds`**.

#### 2. [src/data_ingestion/telemetry_daemon.py](file:///D:/virtual-race-engineer/src/data_ingestion/telemetry_daemon.py)
- **`duckdb_connect(db_path)`:** Process-wide `threading.RLock()` context manager ensuring thread-safe, uniform-config DuckDB access.
- **`DuckDBSingleWriterDaemon`:**
  - Maintains an in-memory `duckdb.connect(":memory:")` mirror (`self.mem_conn`) for sub-millisecond zero-lock reads during live `asyncio` simulation, while draining disk writes sequentially through `self.write_queue = asyncio.Queue()` into `data/f1_historical.duckdb`.
- **Multi-Season (`2021–2024`), 24-Circuit & 6-Session Database Engine:**
  - `SUPPORTED_SEASONS = [2024, 2023, 2022, 2021]`
  - `SESSION_TYPES = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint", "Race"]`
  - `SEASON_PACE_MODIFIERS`: Calibrates driver pace hierarchies per historical era (e.g., `2021` Verstappen vs Hamilton title battle, `2022` Verstappen vs Leclerc, `2023` Red Bull dominance + Alonso podiums, `2024` Red Bull / McLaren / Ferrari convergence).
  - `load_or_populate_gp_session(circuit, session_type="Race", season=2024, db_path=...)`:
    - For `2024 Bahrain Race`, loads the real 57-lap FastF1 dataset from `data/fastf1_cache` and carries forward classified lapped finishers on Laps 56–57 so all 20 drivers remain on the timing tower.
    - For any other `(season, circuit, session_type)` combination, deterministically synthesizes a realistic 20-driver multi-stint session with compound strategies, pit stops, Safety Car / VSC phases, speeds, RPM, throttle, brake, gear, DRS, and UTC timestamps into the `grand_prix_sessions` table.
  - `sync_gp_lap_to_live_table(circuit, session_type, lap, season=2024, track_status_override=None, track_temp_override=None, db_path=...)`: Copies the active 20-car lap snapshot (with any active user Environment Overrides applied) into `live_telemetry_laps` so the LangGraph agent and all tools always see the exact state shown on the dashboard.
  - `generate_circuit_map_coordinates(circuit, n_points=240)`: Loads real FastF1 GPS `(X, Y)` coordinates for Bahrain from `data/circuit_track_map_cache.json` in **`0.33 ms`** (or generates unique closed harmonic track loops for the other 23 circuits).
  - `generate_driver_micro_telemetry(circuit, driver, lap, compound, tyre_life, track_status="GREEN", n_points=480)`: Generates 480 high-resolution spatial telemetry points (`Distance 0..5412m`, `Speed km/h`, `Throttle %`, `Brake %`, `Gear 1..8`, `RPM`, `DRS`) around the circuit's corners and straights.
  - `compute_championship_standings_till_race(season, circuit, current_race_standings_df=None)`: Computes cumulative **World Drivers' Championship (WDC)** and **World Constructors' Championship (WCC)** points from Round 1 up to the selected Grand Prix round (`Prior Pts`), adds live projected points (`+25, +18, +15, +12, +10, +8, +6, +4, +2, +1`) from the current on-track positions (safely assigning unique non-colliding positions `1..20` even on partial custom scenario grids), and returns ranked WDC (20 drivers) and WCC (10 teams) tables.

#### 3. [src/simulation/engine.py](file:///D:/virtual-race-engineer/src/simulation/engine.py) & [src/simulation/live_connector.py](file:///D:/virtual-race-engineer/src/simulation/live_connector.py)
- **`LiveRaceSimulationEngine` (`engine.py`):**
  - Runs 4 concurrent `asyncio` tasks:
    1. `db_daemon.writer_worker()` (background DuckDB disk persistence)
    2. `telemetry_worker()` (`2 Hz / 0.5s` telemetry ingestion)
    3. `physics_and_pit_window_worker()` (`1 Hz / 1.0s` L1 Tensor grid evaluation + 10,000-sim Monte Carlo rejoin matrix)
    4. `legal_rag_worker()` (wakes up in `<1 ms` on `self.incident_event` when `TrackStatus` flips to `VSC` or `SC`, offloading Qdrant retrieval via `asyncio.to_thread` so the `0.5s` telemetry loop drops **zero** packets).
- **`LiveGrandPrixConnector` (`live_connector.py`):**
  - Polls live OpenF1 Race Control messages and maps FIA text messages (`"VIRTUAL SAFETY CAR DEPLOYED"`, `"SAFETY CAR DEPLOYED"`, `"RED FLAG"`) to engine track statuses with automatic offline replay fallback.

---

### PHASE 5: The Autonomous Agent (Multi-Hop Legal RAG & LangGraph Race Engineer)
#### 1. [src/rag_engine/multi_hop.py](file:///D:/virtual-race-engineer/src/rag_engine/multi_hop.py)
- **`MultiHopFIARulebook` Class:**
  - Solves the legal cross-reference problem in the FIA Sporting Regulations (where e.g. Article 55.7/55.8 on Page 61 states *"The stewards may impose either of the penalties under Article 54.3a), 54.3b), 54.3c) or 54.3d)"* without defining what those penalties are).
  - **Hop 1:** Runs `query_cached(question, top_k=2)` to retrieve the primary rules.
  - **Cross-Reference Scanner (`extract_referenced_articles`):** Uses regex `Articles?\s+(\d{1,2}\s*\.\s*\d{1,2}...)` to extract cited article numbers (e.g., `['54.3', '5.3', '49.6']`).
  - **Hop 2 (`_fetch_exact_article_chunk`):** Scrolls and caches all rulebook chunks in RAM once, then matches the negative-lookbehind section-header regex `(?<!Article\s)(?<!Articles\s)(?<!\d)54\s*\.\s*3(?!\d)\s+[A-Z]` so it finds where `54.3` is **defined as a section header** (`Page 59: "54.3 The stewards may impose any one of the penalties below on any driver involved in an Incident: a) A five (5) second time penalty..."`), rather than another chunk that merely cites `"under Article 54.3"`.

#### 2. [src/agent/tools.py](file:///D:/virtual-race-engineer/src/agent/tools.py)
- **`RaceEngineerToolkit` Singleton:** Holds `TyreL1TensorCache`, `PitStopSimulator`, `db_path`, and a lazy `@property` for `MultiHopFIARulebook` (so non-legal queries never wait on `SentenceTransformer` initialization).
- **`compute_remaining_stint_pit_breakeven(ego_compound, ego_tyre_life, track_temp, laps_remaining, pit_loss_sec, target_new_tyre=None)`:**
  - Calculates whether a pit stop has enough laps left in the race (`laps_remaining = max(0, total_laps - current_lap)`) to pay back `pit_loss_sec`:
    $$\text{RecoverableGain} = \max\left(0,\; \sum_{k=1}^{L_{\text{rem}}} \max\big(0,\; \text{deg}_{\text{old}}(\text{age}+k) - \text{deg}_{\text{new}}(k)\big) - \text{warmup\_penalty}\right)$$
    $$\text{NetPayback} = \text{round}(\text{RecoverableGain}, 2) - \text{round}(\text{pit\_loss\_sec}, 2)$$
  - When `NetPayback <= 0` (such as on **Lap 50/57 in Bahrain** with only **7 laps remaining**, where fresh Softs recover `+13.90s` against a `23.00s` Green Flag pit loss, yielding `NetPayback = -9.10s` and dropping Russell from `P5 -> P8`), `pit_is_net_positive_over_stint` is `False`, preventing suicidal late-race pit calls!
- **5 Structured LangChain `@tool` Functions:**
  1. `get_live_race_standings()`
  2. `predict_tyre_degradation(driver, compound, tyre_life, track_temp)`
  3. `analyze_rival_catch_battle(ego_driver, target_driver=None)`
  4. `simulate_pit_stop_mc(driver, race_condition, target_new_tyre, rival_gap_sec)`
  5. `consult_fia_rulebook_multihop(legal_question)`

#### 3. [src/agent/race_engineer.py](file:///D:/virtual-race-engineer/src/agent/race_engineer.py)
- **`extract_mentioned_driver(message, ego_driver)`:**
  - Maps all 20 F1 drivers' 3-letter codes, first names, and surnames (`"ham" / "hamilton" / "lewis" -> "HAM"`, `"lec" / "leclerc" / "charles" -> "LEC"`, etc.).
  - **Homograph Disambiguation:** Prevents English phrases like `"per lap"` (`PER`), `"max speed"` / `"max rpm"` (`VER`), or `"on the gas"` (`GAS`) from falsely triggering rival battle analysis, and never returns `ego_driver` as its own rival.
- **4-Node LangGraph `StateGraph` (`AutonomousRaceEngineer`):**
  1. `router` (`_route_intent_node`): Routes intent to the required tools. Only triggers `consult_fia_rulebook_multihop` when legal/penalty/rulebook keywords appear (or when `"pass"`/`"overtake"` is paired with flag/procedure words like `"under"`, `"behind"`, `"safety car"`, `"sc"`, `"vsc"`, `"line"`, `"pit"`), keeping pure tactical queries like `"if i can pass LEC"` under `<450 ms`. Includes a full-telemetry fallback so no query ever returns an empty header.
  2. `execute_tools` (`_execute_tools_node`): Invokes the selected tools and strips strategy slang from compound legal questions before embedding into Qdrant.
  3. `guardrail` (`_hallucination_guardrail_node`): Deterministically verifies physical bounds (`0.0 <= deg <= 10.0`, `1 <= rejoin_pos <= 25`, `gap >= 0.0`, non-empty Qdrant rule pages).
  4. `synthesize_radio` (`_synthesize_radio_node`): Synthesizes the pit-wall radio response grounded strictly in verified tool numbers, honoring `pit_is_net_positive_over_stint` so late-race negative-payback stops always output `STAY OUT & HOLD POSITION`.

---

### PHASES 6 & 7: The 2-Page `4 Hz` Pit Wall Dashboard, Live Connector & E2E Test Suite
#### 1. [src/dashboard/pit_wall_app.py](file:///D:/virtual-race-engineer/src/dashboard/pit_wall_app.py)
- **Architecture:**
  - Uses `@st.cache_resource` (`load_core_engines()`) to initialize `TyreL1TensorCache`, `AutonomousRaceEngineer`, and `circuit_pit_data.json` once in RAM.
  - **Top Video Stream Broadcast Controller (`Page 1` & `Page 2`):**
    - `▶️ PLAY / RESUME LIVE TELEMETRY` / `⏸️ PAUSE LIVE TELEMETRY` toggle
    - `⏮️ -1 Lap`, `⏭️ +1 Lap`, `⏩ +25% Sector Scrub`, and Playback Speed Multiplier (`1x Real-Time`, `5x Fast`, `15x Race Pace`, `30x High-Speed`).
    - Preserves exact fractional `sub_lap_progress` (`0.0000 .. 0.9999`) when pausing and automatically increments the lap counter when `sub_lap_progress` wraps past `1.0`.
  - **`@st.fragment(run_every=0.25)` Real-Time Dynamic Loop:**
    - Re-renders the 2D Circuit Track Radar (with linear `(cx, cy)` sub-segment interpolation for all 20 cars), the F1 Broadcast HTML Timing Tower (with live sub-lap pace/gap fluctuations), the 6-card Live Sensor Telemetry HUD (`Speed`, `Throttle`, `Brake`, `Gear`, `RPM`, `DRS`), the 3-panel 480-point Micro-Telemetry Trace plot (with live moving diamond markers along the distance axis), the Driver Radio & Team Strategy Decision inspector, and the WDC/WCC Championship Standings tables at **4 Hz (every 250 ms)** while keeping the LangGraph Radio Chat box isolated below so typing is never interrupted.
  - **Synchronized Driver Selection:** Clicking any of the 20 driver pills (`pill_top_{DRV}`) sets `_pill_driver_pending`, which updates `st.session_state.sidebar_ego_driver` before widget instantiation so the sidebar dropdown, green `🟢` pill indicator, 2D radar highlight, micro-telemetry traces, and radio inspector always stay 100% synchronized.
  - **Page 2 Custom Scenario Widget Key Sync (`_sync_scenario_widget_keys`):** Updates `scen_drv_{i}`, `scen_comp_{i}`, `scen_age_{i}`, `scen_int_{i}`, and `scen_lap_{i}` whenever a Preset button or `+1 Lap Forward` button is clicked so Streamlit widgets never overwrite updated scenario state with stale widget values.

#### 2. [test/test_system_e2e.py](file:///D:/virtual-race-engineer/test/test_system_e2e.py)
- **10 Automated End-to-End Verification Suites (`10 passed`):**
  1. `test_l1_tensor_cache_monotonicity_and_bounds`
  2. `test_vectorized_monte_carlo_pit_simulator_latency`
  3. `test_multihop_fia_rulebook_cross_reference_resolver`
  4. `test_langgraph_race_engineer_guardrail`
  5. `test_langgraph_can_i_catch_rival_query` (includes homograph & self-comparison regressions)
  6. `test_multi_gp_sessions_and_circuit_telemetry_generators` (verifies 24 circuits, 20 drivers across all 57 Bahrain laps, and 0.33ms GPS track map loading)
  7. `test_bahrain_lap50_rus_vs_lec_no_late_pit_stop` (verifies late-race Lap 50/57 `GREEN` negative payback -> `STAY OUT & HOLD POSITION`, `SC` positive payback crossover -> `BOX BOX NOW`, and final-lap `Lap 57/57` `0L remaining` boundary)
  8. `test_multi_season_2021_to_2024_and_all_session_types` (verifies seasons `2021–2024` and `FP1/FP2/FP3/Qualifying/Sprint/Race`)
  9. `test_championship_standings_wdc_and_wcc_till_race` (verifies WDC 20-driver & WCC 10-team math across full and partial custom scenario grids)
  10. `test_streamlit_pit_wall_app_video_controls_and_championship_standings` (runs headless Streamlit `AppTest` verifying Page 1 and Page 2 video controls, play/pause fractional progress preservation, driver pills, presets, and WDC/WCC tabs with 0 exceptions).

---

## 5. Essential Commands Cheat Sheet

```powershell
# 1. Verify Qdrant & Runtime Docker Containers are Up
docker ps

# 2. Run the Full 10-Test End-to-End Pytest Verification Suite
D:\virtual-race-engineer\.venv\Scripts\python.exe -m pytest test\test_system_e2e.py -v

# 3. Launch the 2-Page 4 Hz Interactive Pit Wall Command Center (http://localhost:8501)
uv run streamlit run src/dashboard/pit_wall_app.py

# 4. Re-Ingest FIA 2024 Sporting Regulations PDF into Qdrant (if resetting Vector DB)
uv run src/data_ingestion/rag_ingestion.py

# 5. Run Standalone Phase 4 Async Simulation Engine Demo
uv run src/simulation/engine.py

# 6. Run Standalone Phase 5 LangGraph Multi-Hop Race Engineer CLI Demo
uv run src/agent/race_engineer.py
```
