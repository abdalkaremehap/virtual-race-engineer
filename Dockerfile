# ==========================================
# STAGE 1: Builder (uv-builder)
# ==========================================
FROM python:3.11-slim-bookworm AS uv-builder

# Copy the uv binary directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install basic system build dependencies for compiling C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optimize uv behavior for Docker
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Add network timeout resiliency for slow connections/mirrors
ENV UV_HTTP_TIMEOUT=300

# Copy ONLY the dependency files first (maximizes Docker layer caching)
COPY pyproject.toml uv.lock ./

# Sync dependencies to create the isolated .venv
# We MUST use --all-groups so uv installs our [ml], [data], and [agents] blocks!
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --all-groups


# ==========================================
# STAGE 2: Final Runtime
# ==========================================
FROM python:3.11-slim-bookworm

# Set critical environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Create our secure non-root user
RUN useradd -m -u 1000 race_engineer

# Install only essential runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fully compiled virtual environment from Stage 1
COPY --from=uv-builder --chown=race_engineer:race_engineer /app/.venv /app/.venv

# Copy application source, configuration, trained ML models, and compact FIA/DuckDB data (~8.3 MB)
COPY --chown=race_engineer:race_engineer src/ /app/src/
COPY --chown=race_engineer:race_engineer config/ /app/config/
COPY --chown=race_engineer:race_engineer models/ /app/models/
COPY --chown=race_engineer:race_engineer data/f1_historical.duckdb /app/data/f1_historical.duckdb
COPY --chown=race_engineer:race_engineer data/circuit_track_map_cache.json /app/data/circuit_track_map_cache.json
COPY --chown=race_engineer:race_engineer data/historical_championship_cache.json /app/data/historical_championship_cache.json
COPY --chown=race_engineer:race_engineer data/fia_2024_sporting_regulations.pdf /app/data/fia_2024_sporting_regulations.pdf

# Prepare writable cache & embedded Qdrant directories
RUN mkdir -p /home/race_engineer/.cache/fastf1 /app/data/qdrant_local /data/duckdb && \
    chown -R race_engineer:race_engineer /home/race_engineer /app/data /data/duckdb

# Drop root privileges for runtime execution
USER race_engineer

# Default to single-container embedded Qdrant (override with QDRANT_URL for remote Qdrant / Docker Compose)
ENV QDRANT_EMBEDDED=1 \
    QDRANT_LOCAL_PATH=/app/data/qdrant_local \
    PORT=7860

EXPOSE 7860

# Launch the Multi-Page FastAPI F1 Pit Wall Web Application (HF Spaces / Railway / Render compatible)
CMD ["sh", "-c", "uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-7860}"]