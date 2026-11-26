# Deploying Virtual Race Engineer Online (Zero-Download Public Website)

The Virtual Race Engineer now runs as a **5-page FastAPI F1 TV Pro Web Application** (`src/web/app.py`) with built-in **embedded local Qdrant** (`QDRANT_EMBEDDED=1`) and pre-warmed DuckDB telemetry (`data/f1_historical.duckdb`, ~7 MB) — so the entire app runs inside **one single Docker container** with zero external database required.

---

## 1. Run Locally (Windows / Mac / Linux)

```powershell
cd D:\virtual-race-engineer
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860** — you get all 5 real routes:
- `/` — Season Hub & 24-Circuit Interactive Calendar
- `/race/2024/Bahrain` (or any shareable link like `/race/2022/Hungary?lap=39&driver=LEC`) — 60 FPS Pit Wall Command Center + Spoken Radio Comms
- `/sandbox` — Custom What-If Scenario Lab (with 1-click historical F1 stress tests)
- `/standings` — Official 2021–2024 WDC & WCC Championship Standings
- `/rules` — 2-Hop FIA 2024 Sporting Regulations Explorer

*(Note: To use embedded Qdrant locally without Docker Desktop running, set `$env:QDRANT_EMBEDDED="1"` before starting `uvicorn`!)*

---

## 2. Deploy FREE to the Internet — Hugging Face Spaces (16 GB RAM, 2 vCPU)

Hugging Face Spaces gives **16 GB RAM for free**, which easily handles `all-MiniLM-L6-v2` + PyTorch + DuckDB + LightGBM + embedded Qdrant inside one container.

1. Create a free account at **https://huggingface.co/spaces** → click **New Space**.
2. Choose:
   - **Space SDK:** `Docker` → **Blank**
   - **Hardware:** `CPU basic (2 vCPU, 16 GB RAM) — Free`
3. Add a `README.md` header at the top of your Space repo with:
   ```yaml
   ---
   title: F1 Virtual Race Engineer
   emoji: 🏎️
   colorFrom: red
   colorTo: gray
   sdk: docker
   app_port: 7860
   ---
   ```
4. Push this repository (`Dockerfile`, `pyproject.toml`, `uv.lock`, `src/`, `config/`, `models/`, `data/f1_historical.duckdb`, `data/circuit_track_map_cache.json`, `data/historical_championship_cache.json`, `data/fia_2024_sporting_regulations.pdf`) to your Space git remote.
5. Hugging Face builds the [Dockerfile](file:///D:/virtual-race-engineer/Dockerfile) automatically and publishes your live public URL (`https://huggingface.co/spaces/<your-user>/f1-virtual-race-engineer`)!

---

## 3. Deploy with a Custom Domain — Railway or Render (~$5–$7/mo)

1. Push this repo to **GitHub**.
2. In **Railway** (`railway.app`) or **Render** (`render.com`), click **New Project → Deploy from GitHub Repo**.
3. It auto-detects the [Dockerfile](file:///D:/virtual-race-engineer/Dockerfile) (and automatically binds `${PORT}`).
4. Optional environment variables:
   - `QDRANT_EMBEDDED=1` *(already default in Dockerfile — zero external DB needed!)*
   - Or connect **Qdrant Cloud Free Tier** by setting `QDRANT_URL=https://...` and `QDRANT_API_KEY=...` (with `QDRANT_EMBEDDED=0`).
5. Attach your custom domain (e.g. `f1pitwall.com`) with automatic HTTPS.
