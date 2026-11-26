import json

import numpy as np
import pandas as pd


def get_streamlit_anti_blink_css() -> str:
    """
    Eliminates Streamlit's stale-element dimming/flashing (`opacity: 0.33`) and
    running-spinner flicker during fragment updates so the entire dashboard stays visually rock-solid.
    """
    return """
    <style>
        /* Kill Streamlit's grey fade/blink on fragment & rerun updates */
        [data-stale="true"],
        .element-container[data-stale="true"],
        .stPlotlyChart[data-stale="true"],
        .stMarkdown[data-stale="true"],
        .stMetric[data-stale="true"] {
            opacity: 1 !important;
            transition: none !important;
            filter: none !important;
        }
        /* Keep transitions instant on metric & chart containers to avoid pulse flicker */
        div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stMetric"],
        div[data-testid="stPlotlyChart"] {
            transition: none !important;
        }
        /* Suppress subtle status widget flash during background playback ticks */
        div[data-testid="stStatusWidget"] {
            opacity: 0.35;
        }
    </style>
    """


def build_broadcast_payload(
    circuit: str,
    race_name: str,
    season: int,
    session_label: str,
    x_track: np.ndarray,
    y_track: np.ndarray,
    norm_curv: np.ndarray,
    track_length_m: float,
    base_lap_sec: float,
    total_laps: int,
    initial_lap: int,
    initial_sub_lap: float,
    ego_driver: str,
    pit_loss_sec: float,
    track_temp_c: float,
    weather_cond: str,
    laps_by_number: dict[int, list[dict]],
    driver_meta: dict[str, dict],
    micro_tel_df: pd.DataFrame,
    auto_play: bool = False,
    initial_speed: float = 15.0,
) -> dict:
    """Builds a compact (~45 KB) JSON-serializable payload for the 60 FPS HTML5 Canvas broadcast player."""
    sorted_laps = sorted(laps_by_number.keys())
    compact_laps: dict[str, list[list]] = {}
    for lap_num in sorted_laps:
        rows = laps_by_number[lap_num]
        # [driver, pos, compound, tyre_life, gap, lap_time, status]
        compact_laps[str(int(lap_num))] = [
            [
                str(r["driver"]),
                int(r["position"]),
                str(r["compound"]).upper(),
                int(r["tyre_life"]),
                round(float(r["gap_to_leader"]), 3),
                round(float(r["lap_time_sec"]), 3),
                str(r.get("track_status", "GREEN")).upper(),
            ]
            for r in rows
        ]

    # Compact driver metadata (color, team, pace_offset)
    drv_info = {
        code: {
            "color": meta.get("color", "#00E5FF"),
            "team": meta.get("team", "F1"),
            "pace": float(meta.get("pace_offset", 0.3)),
        }
        for code, meta in driver_meta.items()
    }

    # Downsample micro-telemetry to 240 points for crisp, lightweight 60 FPS trace drawing
    if micro_tel_df is not None and not micro_tel_df.empty:
        idx = np.linspace(0, len(micro_tel_df) - 1, min(240, len(micro_tel_df))).astype(int)
        sub_tel = micro_tel_df.iloc[idx]
        tel_payload = {
            "dist": np.round(sub_tel["Distance"].to_numpy(dtype=float), 1).tolist(),
            "speed": np.round(sub_tel["Speed"].to_numpy(dtype=float), 1).tolist(),
            "thr": np.round(sub_tel["Throttle"].to_numpy(dtype=float), 1).tolist(),
            "brk": np.round(sub_tel["Brake"].to_numpy(dtype=float), 0).tolist(),
            "gear": sub_tel["nGear"].astype(int).tolist(),
            "rpm": np.round(sub_tel["RPM"].to_numpy(dtype=float), 0).tolist(),
            "drs": sub_tel["DRS"].astype(int).tolist(),
        }
    else:
        tel_payload = {"dist": [], "speed": [], "thr": [], "brk": [], "gear": [], "rpm": [], "drs": []}

    return {
        "circuit": str(circuit),
        "raceName": str(race_name),
        "season": int(season),
        "sessionLabel": str(session_label),
        "trackX": np.round(np.asarray(x_track, dtype=float), 2).tolist(),
        "trackY": np.round(np.asarray(y_track, dtype=float), 2).tolist(),
        "curv": np.round(np.asarray(norm_curv, dtype=float), 4).tolist(),
        "trackLenM": round(float(track_length_m), 1),
        "baseLapSec": max(55.0, round(float(base_lap_sec), 2)),
        "totalLaps": int(total_laps),
        "availableLaps": [int(l) for l in sorted_laps],
        "initialLap": int(initial_lap),
        "initialSub": round(float(initial_sub_lap) % 1.0, 4),
        "egoDriver": str(ego_driver),
        "pitLossSec": round(float(pit_loss_sec), 2),
        "trackTempC": round(float(track_temp_c), 1),
        "weather": str(weather_cond),
        "laps": compact_laps,
        "drivers": drv_info,
        "tel": tel_payload,
        "autoPlay": bool(auto_play),
        "initialSpeed": float(initial_speed),
    }


def render_smooth_broadcast_html(payload: dict) -> str:
    """
    Returns a self-contained 60 FPS HTML5 Canvas + requestAnimationFrame F1 Broadcast Stage:
    - Zero Streamlit reruns & zero blink during playback
    - High-DPI 2D Circuit Radar with 20 smoothly interpolated cars, Ego spotlight, Pit Exit Ghost, & hover tooltips
    - Live F1 Steering-Wheel HUD (15-LED RPM bar, Speed, Gear, Throttle/Brake bars, DRS, Race Clock)
    - Smooth 60 FPS Scrolling Telemetry Trace Canvas synced to exact car distance
    - Live Compact Timing Tower synced at 60 FPS with click-to-focus driver selection
    """
    safe_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0b0e14;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    overflow: hidden;
    user-select: none;
  }}
  .stage {{
    background: linear-gradient(180deg, #10151f 0%, #0b0e14 100%);
    border: 1px solid #262e3d;
    border-radius: 12px;
    padding: 9px 10px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    height: 632px;
  }}
  /* Top Broadcast Header + Video Controls */
  .topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    background: #141a24;
    border: 1px solid #283142;
    border-radius: 8px;
    padding: 6px 10px;
  }}
  .title-block {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }}
  .live-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #ff2800;
    box-shadow: 0 0 8px #ff2800;
    flex-shrink: 0;
  }}
  .live-dot.paused {{
    background: #6e7681;
    box-shadow: none;
  }}
  .race-title {{
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.6px;
    color: #f0f6fc;
    white-space: nowrap;
  }}
  .flag-pill {{
    padding: 2px 9px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 0.6px;
    white-space: nowrap;
  }}
  .flag-GREEN {{ background: #0f5132; color: #75f0a8; border: 1px solid #198754; }}
  .flag-VSC {{ background: #5c4300; color: #ffd43b; border: 1px solid #f59f00; }}
  .flag-SC {{ background: #7a141d; color: #ff8787; border: 1px solid #fa5252; }}
  .flag-RED {{ background: #b00020; color: #ffffff; border: 1px solid #ff4d4f; }}

  .controls {{
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: nowrap;
  }}
  .btn {{
    background: #1f2736;
    color: #e6edf3;
    border: 1px solid #354055;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.12s ease;
    white-space: nowrap;
  }}
  .btn:hover {{ background: #2c374c; }}
  .btn-play {{
    background: linear-gradient(90deg, #e10600 0%, #ff3b30 100%);
    border-color: #ff4d4f;
    color: #ffffff;
    min-width: 104px;
  }}
  .btn-play.playing {{
    background: #1f2937;
    border-color: #00e5ff;
    color: #00e5ff;
  }}
  select.speed-sel {{
    background: #1f2736;
    color: #00e5ff;
    border: 1px solid #354055;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
  }}

  /* Video Scrubber + Telemetry Clock Strip */
  .scrubber-bar {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    background: #111620;
    border: 1px solid #242c3b;
    border-radius: 8px;
    padding: 5px 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
  }}
  .scrubber-bar input[type="range"] {{
    flex: 1;
    min-width: 120px;
    accent-color: #00e5ff;
    cursor: pointer;
    height: 6px;
  }}
  .hud-stat {{ color: #9ecbff; white-space: nowrap; }}
  .hud-stat b {{ color: #ffffff; }}
  .fps-tag {{ color: #00ff88; font-weight: 700; font-size: 11px; }}

  /* Main 3-Column Grid: Radar | Telemetry & Steering HUD | Live Timing Tower */
  .main-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.18fr) minmax(0, 1.05fr) minmax(168px, 0.78fr);
    gap: 8px;
    flex: 1;
    min-height: 0;
  }}
  @media (max-width: 760px) {{
    .main-grid {{
      grid-template-columns: minmax(0, 1.25fr) minmax(160px, 0.85fr);
      grid-template-rows: 1.15fr 0.9fr;
    }}
    .main-grid > .panel:nth-child(2) {{
      grid-column: 1 / -1;
      order: 3;
    }}
  }}
  .panel {{
    background: #121722;
    border: 1px solid #242d3d;
    border-radius: 9px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    min-width: 0;
  }}
  .panel-hdr {{
    padding: 6px 10px;
    background: #161d2b;
    border-bottom: 1px solid #242d3d;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.7px;
    color: #9bb0c9;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .canvas-wrap {{
    flex: 1;
    position: relative;
    min-height: 0;
  }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  .tooltip {{
    position: absolute;
    pointer-events: none;
    background: rgba(12, 16, 24, 0.94);
    border: 1px solid #00e5ff;
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 11px;
    line-height: 1.4;
    color: #f0f6fc;
    display: none;
    z-index: 10;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  }}

  /* Steering Wheel Telemetry Strip */
  .wheel-hud {{
    padding: 8px 10px 6px 10px;
    border-bottom: 1px solid #222b3a;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: #0f141e;
  }}
  .rpm-leds {{
    display: grid;
    grid-template-columns: repeat(15, 1fr);
    gap: 4px;
    height: 8px;
  }}
  .led {{
    border-radius: 3px;
    background: #202836;
  }}
  .wheel-metrics {{
    display: grid;
    grid-template-columns: 1.1fr 0.7fr 1.15fr 0.85fr;
    gap: 8px;
    align-items: center;
  }}
  .big-val {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 22px;
    font-weight: 900;
    color: #00e5ff;
    line-height: 1.05;
  }}
  .sub-lbl {{
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .gear-box {{
    text-align: center;
    background: #182030;
    border: 1px solid #303d54;
    border-radius: 6px;
    padding: 3px 0;
  }}
  .gear-num {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 22px;
    font-weight: 900;
    color: #ffd700;
    line-height: 1.05;
  }}
  .pedal-bars {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .bar-row {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    font-family: monospace;
  }}
  .bar-track {{
    flex: 1;
    height: 7px;
    background: #1f2736;
    border-radius: 4px;
    overflow: hidden;
  }}
  .bar-fill-thr {{ height: 100%; background: #00ff66; width: 0%; }}
  .bar-fill-brk {{ height: 100%; background: #ff3333; width: 0%; }}
  .drs-pill {{
    text-align: center;
    padding: 6px 4px;
    border-radius: 6px;
    font-weight: 900;
    font-size: 11px;
    letter-spacing: 0.6px;
    background: #1c2432;
    color: #6e7681;
    border: 1px solid #2e384b;
  }}
  .drs-pill.open {{
    background: #0f5132;
    color: #00ff88;
    border-color: #00ff88;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.35);
  }}

  /* Compact 60 FPS Timing Tower */
  .tower-scroll {{
    flex: 1;
    overflow-y: auto;
    padding: 4px 6px;
  }}
  .tower-row {{
    display: grid;
    grid-template-columns: 24px 44px 36px 1fr 56px;
    align-items: center;
    gap: 4px;
    padding: 4px 6px;
    border-radius: 4px;
    font-size: 11.5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    cursor: pointer;
    border-left: 3px solid transparent;
    border-bottom: 1px solid #1b2230;
  }}
  .tower-row:hover {{ background: rgba(255,255,255,0.06); }}
  .tower-row.ego {{
    background: rgba(0, 229, 255, 0.16);
    border-left-color: #00ff66 !important;
  }}
  .tyre-dot {{
    display: inline-block;
    width: 16px;
    height: 16px;
    line-height: 16px;
    text-align: center;
    border-radius: 50%;
    font-weight: 900;
    font-size: 10px;
  }}
  .pit-pill {{
    display: inline-block;
    padding: 0 4px;
    border-radius: 3px;
    font-size: 8.5px;
    font-weight: 900;
    letter-spacing: 0.3px;
    margin-left: 3px;
    vertical-align: middle;
  }}
  .pit-in {{ background: #ffb300; color: #0b0e14; }}
  .pit-box {{ background: rgba(255, 179, 0, 0.22); color: #ffd54f; border: 1px solid #ffb300; }}
  .pit-out {{ background: rgba(0, 229, 255, 0.18); color: #7dd3fc; border: 1px solid #00e5ff; }}
</style>
</head>
<body>
<div class="stage">
  <div class="topbar">
    <div class="title-block">
      <div id="liveDot" class="live-dot paused"></div>
      <div id="raceTitle" class="race-title"></div>
      <div id="flagPill" class="flag-pill flag-GREEN">GREEN</div>
    </div>
    <div class="controls">
      <button id="btnPlay" class="btn btn-play">▶ PLAY 60 FPS</button>
      <button id="btnPrev" class="btn">⏮ -1L</button>
      <button id="btnNext" class="btn">⏭ +1L</button>
      <select id="speedSel" class="speed-sel">
        <option value="1">1x Real-Time</option>
        <option value="2">2x Smooth</option>
        <option value="5">5x Fast</option>
        <option value="15" selected>15x Race Pace</option>
        <option value="30">30x High-Speed</option>
      </select>
    </div>
  </div>

  <div class="scrubber-bar">
    <span class="hud-stat">LAP <b id="hudLap">1/57</b></span>
    <input id="scrubRange" type="range" min="0" max="1000" value="350" />
    <span class="hud-stat" id="hudSector"><b>S1</b></span>
    <span class="hud-stat" id="hudDist">0 m</span>
    <span class="hud-stat" id="hudClock"><b>00:00:00.00</b></span>
    <span class="fps-tag" id="fpsTag">60 FPS</span>
  </div>

  <div class="main-grid">
    <!-- Left: 60 FPS 2D Circuit Track Radar -->
    <div class="panel">
      <div class="panel-hdr">
        <span id="radarTitle">🗺️ LIVE 60 FPS CIRCUIT RADAR</span>
        <span style="color:#00e5ff; font-size:10px;">● EGO &nbsp; ◌ REJOIN GHOST &nbsp; <span style="color:#ffb300;">▬ PIT LANE</span></span>
      </div>
      <div class="canvas-wrap">
        <canvas id="radarCanvas"></canvas>
        <div id="radarTooltip" class="tooltip"></div>
      </div>
    </div>

    <!-- Middle: Steering Wheel HUD + 60 FPS Telemetry Traces -->
    <div class="panel">
      <div class="panel-hdr">
        <span id="telTitle">📊 LIVE CAR TELEMETRY</span>
        <span id="telMeta" style="color:#8b949e;"></span>
      </div>
      <div class="wheel-hud">
        <div id="rpmLeds" class="rpm-leds"></div>
        <div class="wheel-metrics">
          <div>
            <div class="sub-lbl">Speed</div>
            <div id="hudSpeed" class="big-val">312 <span style="font-size:11px;color:#8b949e;">km/h</span></div>
          </div>
          <div class="gear-box">
            <div class="sub-lbl">Gear</div>
            <div id="hudGear" class="gear-num">7</div>
          </div>
          <div class="pedal-bars">
            <div class="bar-row">
              <span style="width:24px;color:#00ff66;">THR</span>
              <div class="bar-track"><div id="barThr" class="bar-fill-thr"></div></div>
              <span id="valThr" style="width:30px;text-align:right;">100%</span>
            </div>
            <div class="bar-row">
              <span style="width:24px;color:#ff5555;">BRK</span>
              <div class="bar-track"><div id="barBrk" class="bar-fill-brk"></div></div>
              <span id="valBrk" style="width:30px;text-align:right;">0%</span>
            </div>
          </div>
          <div id="hudDrs" class="drs-pill">DRS OFF</div>
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="telCanvas"></canvas>
      </div>
    </div>

    <!-- Right: Live Synced 60 FPS Timing Tower -->
    <div class="panel">
      <div class="panel-hdr">
        <span>📡 LIVE TIMING TOWER</span>
        <span style="color:#6e7681;font-size:10px;">CLICK DRIVER</span>
      </div>
      <div id="towerScroll" class="tower-scroll"></div>
    </div>
  </div>
</div>

<script>
(() => {{
  const P = {safe_json};

  const TYRE_STYLE = {{
    SOFT: {{ ch: "S", bg: "#ff2800", fg: "#ffffff" }},
    MEDIUM: {{ ch: "M", bg: "#ffd700", fg: "#111111" }},
    HARD: {{ ch: "H", bg: "#f0f0f0", fg: "#111111" }},
    INTERMEDIATE: {{ ch: "I", bg: "#39b54a", fg: "#ffffff" }},
    WET: {{ ch: "W", bg: "#0072bc", fg: "#ffffff" }},
  }};

  let isPlaying = Boolean(P.autoPlay);
  let speedMult = Number(P.initialSpeed || 15.0);
  let focusDriver = P.egoDriver;
  const avail = P.availableLaps.length ? P.availableLaps : [P.initialLap];
  let curLapIdx = Math.max(0, avail.indexOf(P.initialLap));
  let subLap = Number(P.initialSub || 0.35);

  // DOM refs
  const liveDot = document.getElementById("liveDot");
  const raceTitle = document.getElementById("raceTitle");
  const flagPill = document.getElementById("flagPill");
  const btnPlay = document.getElementById("btnPlay");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  const speedSel = document.getElementById("speedSel");
  const scrubRange = document.getElementById("scrubRange");
  const hudLap = document.getElementById("hudLap");
  const hudSector = document.getElementById("hudSector");
  const hudDist = document.getElementById("hudDist");
  const hudClock = document.getElementById("hudClock");
  const fpsTag = document.getElementById("fpsTag");
  const radarTitle = document.getElementById("radarTitle");
  const telTitle = document.getElementById("telTitle");
  const telMeta = document.getElementById("telMeta");
  const hudSpeed = document.getElementById("hudSpeed");
  const hudGear = document.getElementById("hudGear");
  const barThr = document.getElementById("barThr");
  const barBrk = document.getElementById("barBrk");
  const valThr = document.getElementById("valThr");
  const valBrk = document.getElementById("valBrk");
  const hudDrs = document.getElementById("hudDrs");
  const rpmLedsEl = document.getElementById("rpmLeds");
  const towerScroll = document.getElementById("towerScroll");
  const radarCanvas = document.getElementById("radarCanvas");
  const radarTooltip = document.getElementById("radarTooltip");
  const telCanvas = document.getElementById("telCanvas");

  raceTitle.textContent = `${{P.season}} ${{P.raceName.toUpperCase()}} (${{P.sessionLabel.toUpperCase()}})`;
  radarTitle.textContent = `🗺️ ${{P.circuit.toUpperCase()}} 60 FPS RADAR`;
  telMeta.textContent = `🌡️ ${{P.trackTempC.toFixed(1)}}°C | ${{P.weather}}`;
  speedSel.value = String(speedMult);

  // Build 15 RPM shift-light LEDs
  const ledEls = [];
  for (let i = 0; i < 15; i++) {{
    const d = document.createElement("div");
    d.className = "led";
    rpmLedsEl.appendChild(d);
    ledEls.push(d);
  }}

  // Precompute normalized track coordinates (0..1)
  const nPts = P.trackX.length;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < nPts; i++) {{
    if (P.trackX[i] < minX) minX = P.trackX[i];
    if (P.trackX[i] > maxX) maxX = P.trackX[i];
    if (P.trackY[i] < minY) minY = P.trackY[i];
    if (P.trackY[i] > maxY) maxY = P.trackY[i];
  }}
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);

  function trackPointAt(frac) {{
    const f = ((frac % 1.0) + 1.0) % 1.0;
    const fIdx = f * (nPts - 1);
    const i0 = Math.floor(fIdx) % nPts;
    const i1 = (i0 + 1) % nPts;
    const a = fIdx - Math.floor(fIdx);
    return {{
      x: (1 - a) * P.trackX[i0] + a * P.trackX[i1],
      y: (1 - a) * P.trackY[i0] + a * P.trackY[i1],
      curv: (1 - a) * P.curv[i0] + a * P.curv[i1],
    }};
  }}

  function setPlayUI() {{
    btnPlay.textContent = isPlaying ? "⏸ PAUSE 60 FPS" : "▶ PLAY 60 FPS";
    btnPlay.classList.toggle("playing", isPlaying);
    liveDot.classList.toggle("paused", !isPlaying);
  }}
  setPlayUI();

  btnPlay.addEventListener("click", () => {{ isPlaying = !isPlaying; setPlayUI(); }});
  btnPrev.addEventListener("click", () => {{ curLapIdx = Math.max(0, curLapIdx - 1); subLap = 0.0; }});
  btnNext.addEventListener("click", () => {{ curLapIdx = Math.min(avail.length - 1, curLapIdx + 1); subLap = 0.0; }});
  speedSel.addEventListener("change", (e) => {{ speedMult = Number(e.target.value); }});
  scrubRange.addEventListener("input", (e) => {{ subLap = Number(e.target.value) / 1000.0; }});

  window.addEventListener("keydown", (e) => {{
    if (e.code === "Space" && document.activeElement.tagName !== "INPUT") {{
      e.preventDefault();
      isPlaying = !isPlaying;
      setPlayUI();
    }}
  }});

  // Canvas resize helpers
  function fitCanvas(canvas) {{
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(10, Math.floor(rect.width * dpr));
    const h = Math.max(10, Math.floor(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {{
      canvas.width = w;
      canvas.height = h;
    }}
    return {{ ctx: canvas.getContext("2d"), w: rect.width, h: rect.height, dpr }};
  }}

  let hoverMouse = null;
  let lastScreenCars = [];
  radarCanvas.addEventListener("mousemove", (e) => {{
    const rect = radarCanvas.getBoundingClientRect();
    hoverMouse = {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};
  }});
  radarCanvas.addEventListener("mouseleave", () => {{
    hoverMouse = null;
    radarTooltip.style.display = "none";
  }});
  radarCanvas.addEventListener("click", () => {{
    if (!hoverMouse) return;
    let best = null, bestD = 18;
    for (const c of lastScreenCars) {{
      const d = Math.hypot(c.sx - hoverMouse.x, c.sy - hoverMouse.y);
      if (d < bestD) {{ bestD = d; best = c; }}
    }}
    if (best) focusDriver = best.drv;
  }});

  // Build interpolated car states between current lap and next lap for butter-smooth gap evolution
  function getInterpolatedCars() {{
    const lapNum = avail[curLapIdx];
    const nextLapNum = avail[Math.min(avail.length - 1, curLapIdx + 1)];
    const rowsNow = P.laps[String(lapNum)] || [];
    const rowsNext = P.laps[String(nextLapNum)] || rowsNow;
    const nextMap = Object.create(null);
    for (const r of rowsNext) nextMap[r[0]] = r;

    const trackStatus = rowsNow.length ? rowsNow[0][6] : "GREEN";
    const statusCap = trackStatus === "GREEN" ? 1.0 : (trackStatus === "VSC" ? 0.72 : 0.58);

    const cars = [];
    for (const r of rowsNow) {{
      const drv = r[0];
      const pos = r[1];
      const comp = r[2];
      const age = r[3];
      const gap0 = r[4];
      const ltime = r[5];
      const rNext = nextMap[drv];
      const gap1 = rNext ? rNext[4] : gap0;
      // Smooth linear interpolation across the lap (only if not a huge pit-stop jump)
      const gap = (Math.abs(gap1 - gap0) < 12.0) ? (gap0 + (gap1 - gap0) * subLap) : gap0;
      const frac = (((subLap - (gap / P.baseLapSec)) % 1.0) + 1.0) % 1.0;

      // Real pit-stop state detection from lap-to-lap tyre_life & compound transitions
      const pittingThisLap = Boolean(lapNum > 1 && rNext && (rNext[3] === 1 || rNext[2] !== comp));
      const exitedPitThisLap = Boolean(lapNum > 1 && age === 1);
      const inPitNow = (pittingThisLap && frac >= 0.89) || (exitedPitThisLap && frac <= 0.09);
      const pitBadge = inPitNow ? "IN PIT" : (exitedPitThisLap ? "OUT" : (pittingThisLap ? "BOX" : ""));

      const pt = trackPointAt(frac);
      const dm = P.drivers[drv] || {{ color: "#00E5FF", team: "F1", pace: 0.3 }};
      const paceFactor = 1.0 - dm.pace * 0.008;
      const degPen = Math.min(0.15, age * 0.0025);
      const rawSpd = Math.max(75, Math.min(345, (335 - 235 * Math.pow(pt.curv, 0.75)) * paceFactor * (1 - degPen) * statusCap));
      const speed = inPitNow ? 80 : rawSpd;
      cars.push({{
        drv, pos, comp, age, gap, ltime, status: trackStatus,
        frac, wx: pt.x, wy: pt.y, curv: pt.curv, speed,
        inPitNow, pittingThisLap, exitedPitThisLap, pitBadge,
        color: dm.color, team: dm.team,
      }});
    }}
    return {{ lapNum, trackStatus, cars }};
  }}

  // Draw 2D Circuit Radar at 60 FPS (including Dedicated Pit Lane & In-Pit Cars)
  function drawRadar(cars, trackStatus) {{
    const {{ ctx, w, h, dpr }} = fitCanvas(radarCanvas);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = 28;
    const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
    const offX = (w - spanX * scale) * 0.5;
    const offY = (h - spanY * scale) * 0.5;
    const toSX = (wx) => offX + (wx - minX) * scale;
    const toSY = (wy) => h - (offY + (wy - minY) * scale);

    // Compute screen-space track centroid so the Pit Lane always offsets cleanly inward
    let cenX = 0, cenY = 0;
    for (let i = 0; i < nPts; i++) {{
      cenX += toSX(P.trackX[i]);
      cenY += toSY(P.trackY[i]);
    }}
    cenX /= nPts;
    cenY /= nPts;

    // Pit lane screen-space helper for frac in [0.89..1.0) U [0.0..0.09]
    const PIT_START = 0.89;
    const PIT_END = 0.09;
    const PIT_SPAN = (1.0 - PIT_START) + PIT_END; // 0.20 of lap
    function pitLaneScreenPoint(frac) {{
      const f = ((frac % 1.0) + 1.0) % 1.0;
      const u = f >= PIT_START ? (f - PIT_START) / PIT_SPAN : (f + (1.0 - PIT_START)) / PIT_SPAN;
      const p0 = trackPointAt(f);
      const pPrev = trackPointAt(f - 0.008);
      const pNext = trackPointAt(f + 0.008);
      const bx = toSX(p0.x), by = toSY(p0.y);
      const tx = toSX(pNext.x) - toSX(pPrev.x);
      const ty = toSY(pNext.y) - toSY(pPrev.y);
      const tLen = Math.hypot(tx, ty) || 1;
      let nx = -ty / tLen, ny = tx / tLen;
      // Pick the normal pointing inward toward the circuit centroid
      if (nx * (cenX - bx) + ny * (cenY - by) < 0) {{ nx = -nx; ny = -ny; }}
      const blend = Math.sin(Math.PI * Math.max(0, Math.min(1, u)));
      const offsetPx = 21.0 * Math.pow(blend, 0.65);
      return {{ sx: bx + nx * offsetPx, sy: by + ny * offsetPx }};
    }}

    // 1. Outer tarmac border
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const sx = toSX(P.trackX[i]), sy = toSY(P.trackY[i]);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }}
    ctx.closePath();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = "#262f3f";
    ctx.lineWidth = 14;
    ctx.stroke();

    // 1B. Dedicated Pit Lane Bypass (Pit Entry -> Pit Box -> Pit Exit)
    const pitSteps = 28;
    const pitPts = [];
    for (let k = 0; k <= pitSteps; k++) {{
      const f = (PIT_START + (k / pitSteps) * PIT_SPAN) % 1.0;
      pitPts.push(pitLaneScreenPoint(f));
    }}
    ctx.save();
    ctx.beginPath();
    for (let k = 0; k < pitPts.length; k++) {{
      if (k === 0) ctx.moveTo(pitPts[k].sx, pitPts[k].sy);
      else ctx.lineTo(pitPts[k].sx, pitPts[k].sy);
    }}
    ctx.strokeStyle = "#1b2230";
    ctx.lineWidth = 7.5;
    ctx.stroke();

    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#ffb300";
    ctx.lineWidth = 2.2;
    ctx.stroke();
    ctx.setLineDash([]);

    // Pit Entry, Pit Box, & Pit Exit labels
    const pitMid = pitPts[Math.floor(pitPts.length / 2)];
    ctx.fillStyle = "#ffb300";
    ctx.beginPath();
    ctx.arc(pitPts[0].sx, pitPts[0].sy, 2.8, 0, Math.PI * 2);
    ctx.arc(pitPts[pitPts.length - 1].sx, pitPts[pitPts.length - 1].sy, 2.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "800 8.5px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("PIT LANE", pitMid.sx, pitMid.sy - 7);
    ctx.restore();

    // 2. Sector-colored inner racing line (S1 cyan, S2 yellow, S3 purple)
    const secColors = ["#00e5ff", "#ffd700", "#c084fc"];
    for (let s = 0; s < 3; s++) {{
      const iStart = Math.floor((s / 3) * (nPts - 1));
      const iEnd = Math.floor(((s + 1) / 3) * (nPts - 1));
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {{
        const sx = toSX(P.trackX[i]), sy = toSY(P.trackY[i]);
        if (i === iStart) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }}
      ctx.strokeStyle = trackStatus === "SC" ? "#ff4d4f" : (trackStatus === "VSC" ? "#f59f00" : secColors[s]);
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 3.5;
      ctx.stroke();
    }}
    ctx.globalAlpha = 1.0;

    // 3. Start/Finish checkered marker
    const sfX = toSX(P.trackX[0]), sfY = toSY(P.trackY[0]);
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#ff2800";
    ctx.lineWidth = 2;
    ctx.fillRect(sfX - 5, sfY - 5, 10, 10);
    ctx.strokeRect(sfX - 5, sfY - 5, 10, 10);

    // 4. Pit Exit Ghost Car for focusDriver
    const egoCar = cars.find(c => c.drv === focusDriver) || cars[0];
    if (egoCar) {{
      const ghostFrac = (((egoCar.frac - (P.pitLossSec / P.baseLapSec)) % 1.0) + 1.0) % 1.0;
      const gp = trackPointAt(ghostFrac);
      const gx = toSX(gp.x), gy = toSY(gp.y);
      const pulse = 9 + 2.5 * Math.sin(performance.now() * 0.008);
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.arc(gx, gy, pulse, 0, Math.PI * 2);
      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 9px ui-monospace, monospace";
      ctx.fillStyle = "#00e5ff";
      ctx.textAlign = "center";
      ctx.fillText(`GHOST (-${{Math.round(P.pitLossSec)}}s)`, gx, gy + pulse + 10);
      ctx.restore();
    }}

    // 5. Render rival cars back-to-front (displacing in-pit cars onto the Pit Lane), then focusDriver on top
    lastScreenCars = [];
    const sortedDraw = [...cars].reverse();
    let egoDraw = null;
    for (const c of sortedDraw) {{
      const pitCoord = c.inPitNow ? pitLaneScreenPoint(c.frac) : null;
      const sx = pitCoord ? pitCoord.sx : toSX(c.wx);
      const sy = pitCoord ? pitCoord.sy : toSY(c.wy);
      const item = {{ ...c, sx, sy }};
      lastScreenCars.push(item);
      if (c.drv === focusDriver) {{
        egoDraw = item;
        continue;
      }}
      if (c.inPitNow) {{
        ctx.beginPath();
        ctx.arc(sx, sy, 9.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffb300";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
      ctx.beginPath();
      ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = c.color;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = c.inPitNow ? "#ffb300" : "#0b0e14";
      ctx.stroke();

      ctx.font = "700 9px sans-serif";
      ctx.fillStyle = c.inPitNow ? "#ffd54f" : "#d0d7de";
      ctx.textAlign = "center";
      ctx.fillText(c.inPitNow ? `${{c.drv}} [PIT]` : c.drv, sx, sy - 8);
    }}

    if (egoDraw) {{
      const pulseR = 12 + 2.5 * Math.sin(performance.now() * 0.01);
      ctx.beginPath();
      ctx.arc(egoDraw.sx, egoDraw.sy, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = egoDraw.inPitNow ? "rgba(255, 179, 0, 0.7)" : "rgba(0, 255, 102, 0.45)";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(egoDraw.sx, egoDraw.sy, 8, 0, Math.PI * 2);
      ctx.fillStyle = egoDraw.color;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = egoDraw.inPitNow ? "#ffb300" : "#00ff66";
      ctx.stroke();

      ctx.font = "900 11px sans-serif";
      ctx.fillStyle = egoDraw.inPitNow ? "#ffd54f" : "#00ff66";
      ctx.textAlign = "center";
      ctx.fillText(egoDraw.inPitNow ? `${{egoDraw.drv}} [IN PIT]` : egoDraw.drv, egoDraw.sx, egoDraw.sy - 13);
    }}

    // Hover tooltip hit test
    if (hoverMouse) {{
      let hit = null, bestD = 18;
      for (const c of lastScreenCars) {{
        const d = Math.hypot(c.sx - hoverMouse.x, c.sy - hoverMouse.y);
        if (d < bestD) {{ bestD = d; hit = c; }}
      }}
      if (hit) {{
        radarTooltip.style.display = "block";
        radarTooltip.style.left = Math.min(w - 165, Math.max(8, hit.sx + 12)) + "px";
        radarTooltip.style.top = Math.min(h - 85, Math.max(8, hit.sy - 14)) + "px";
        const pitLine = hit.inPitNow
          ? `<br><span style="color:#ffb300;font-weight:800;">🔧 IN PIT LANE (80 km/h)</span>`
          : (hit.pittingThisLap
            ? `<br><span style="color:#ffd54f;font-weight:700;">🔧 BOXING THIS LAP</span>`
            : (hit.exitedPitThisLap ? `<br><span style="color:#7dd3fc;font-weight:700;">🚀 PIT OUT-LAP (FRESH TYRES)</span>` : ""));
        radarTooltip.innerHTML =
          `<b>P${{hit.pos}} ${{hit.drv}}</b> (${{hit.team}})<br>` +
          `Gap: ${{hit.pos === 1 ? "LEADER" : "+" + hit.gap.toFixed(3) + "s"}}<br>` +
          `Tyre: ${{hit.comp}} (${{hit.age}}L) | ${{hit.speed.toFixed(0)}} km/h${{pitLine}}`;
      }} else {{
        radarTooltip.style.display = "none";
      }}
    }}
  }}

  // Draw 60 FPS Telemetry Trace Canvas (Speed + Throttle/Brake + Live Playhead)
  function drawTelemetry(egoCar) {{
    const {{ ctx, w, h, dpr }} = fitCanvas(telCanvas);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const T = P.tel;
    const m = T.speed.length;
    if (m < 2) return;

    const padL = 34, padR = 12, padT = 14, padB = 20;
    const plotW = Math.max(20, w - padL - padR);
    const topH = Math.max(30, (h - padT - padB) * 0.56);
    const gapY = 14;
    const botTop = padT + topH + gapY;
    const botH = Math.max(24, h - botTop - padB);

    // Grid lines
    ctx.strokeStyle = "#1f2736";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {{
      const y = padT + (g / 3) * topH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }}
    ctx.beginPath(); ctx.moveTo(padL, botTop); ctx.lineTo(padL + plotW, botTop); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, botTop + botH); ctx.lineTo(padL + plotW, botTop + botH); ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#6e7681";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText("350", padL - 4, padT + 4);
    ctx.fillText("200", padL - 4, padT + topH * 0.55);
    ctx.fillText("70", padL - 4, padT + topH + 3);
    ctx.fillText("100%", padL - 4, botTop + 5);
    ctx.fillText("0%", padL - 4, botTop + botH + 3);

    // Speed curve (cyan)
    ctx.beginPath();
    for (let i = 0; i < m; i++) {{
      const x = padL + (i / (m - 1)) * plotW;
      const y = padT + (1 - Math.max(0, Math.min(1, (T.speed[i] - 70) / 280))) * topH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Throttle curve (green)
    ctx.beginPath();
    for (let i = 0; i < m; i++) {{
      const x = padL + (i / (m - 1)) * plotW;
      const y = botTop + (1 - T.thr[i] / 100) * botH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Brake curve (red)
    ctx.beginPath();
    for (let i = 0; i < m; i++) {{
      const x = padL + (i / (m - 1)) * plotW;
      const y = botTop + (1 - T.brk[i] / 100) * botH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.strokeStyle = "#ff3333";
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // Live cursor synced to ego car's exact fractional track position
    const frac = egoCar ? egoCar.frac : subLap;
    const fIdx = frac * (m - 1);
    const i0 = Math.floor(fIdx);
    const i1 = Math.min(m - 1, i0 + 1);
    const a = fIdx - i0;
    const curSpd = T.speed[i0] + (T.speed[i1] - T.speed[i0]) * a;
    const curThr = T.thr[i0] + (T.thr[i1] - T.thr[i0]) * a;
    const curBrk = T.brk[i0] + (T.brk[i1] - T.brk[i0]) * a;
    const curGear = T.gear[i0];
    const curRpm = T.rpm[i0] + (T.rpm[i1] - T.rpm[i0]) * a;
    const curDrs = T.drs[i0];

    const cx = padL + frac * plotW;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, botTop + botH);
    ctx.stroke();
    ctx.restore();

    const sySpd = padT + (1 - Math.max(0, Math.min(1, (curSpd - 70) / 280))) * topH;
    ctx.beginPath();
    ctx.arc(cx, sySpd, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#00ff66";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Update Steering-Wheel HUD smoothly at 60 FPS
    telTitle.textContent = `📊 ${{focusDriver}} LIVE TELEMETRY (60 FPS)`;
    hudSpeed.innerHTML = `${{Math.round(curSpd)}} <span style="font-size:11px;color:#8b949e;">km/h</span>`;
    hudGear.textContent = String(curGear);
    barThr.style.width = `${{Math.round(curThr)}}%`;
    barBrk.style.width = `${{Math.round(curBrk)}}%`;
    valThr.textContent = `${{Math.round(curThr)}}%`;
    valBrk.textContent = `${{Math.round(curBrk)}}%`;
    hudDrs.textContent = curDrs ? "DRS OPEN" : "DRS OFF";
    hudDrs.classList.toggle("open", Boolean(curDrs));

    // Update 15 RPM shift-light LEDs
    const lit = Math.round(Math.max(0, Math.min(1, (curRpm - 7200) / 5000)) * 15);
    for (let i = 0; i < 15; i++) {{
      if (i < lit) {{
        ledEls[i].style.background = i < 5 ? "#00ff66" : (i < 10 ? "#ff3b30" : "#d946ef");
        ledEls[i].style.boxShadow = `0 0 6px ${{ledEls[i].style.background}}`;
      }} else {{
        ledEls[i].style.background = "#1e2634";
        ledEls[i].style.boxShadow = "none";
      }}
    }}
  }}

  // Throttled (12 Hz) DOM update for timing tower so text stays ultra-crisp with zero layout thrash
  let lastTowerTs = 0;
  function updateTower(cars, nowTs) {{
    if (nowTs - lastTowerTs < 85 && towerScroll.childElementCount > 0) return;
    lastTowerTs = nowTs;
    let html = "";
    for (const c of cars) {{
      const ts = TYRE_STYLE[c.comp] || TYRE_STYLE.MEDIUM;
      const isEgo = c.drv === focusDriver;
      const gapTxt = c.pos === 1 ? "LEADER" : `+${{c.gap.toFixed(2)}}s`;
      const pitChip = c.pitBadge === "IN PIT"
        ? `<span class="pit-pill pit-in">PIT</span>`
        : (c.pitBadge === "BOX"
          ? `<span class="pit-pill pit-box">BOX</span>`
          : (c.pitBadge === "OUT" ? `<span class="pit-pill pit-out">OUT</span>` : ""));
      html += `<div class="tower-row ${{isEgo ? "ego" : ""}}" data-drv="${{c.drv}}" style="border-left-color:${{c.color}};">` +
        `<span style="color:#8b949e;font-weight:700;">P${{c.pos}}</span>` +
        `<span style="color:#ffffff;font-weight:800;white-space:nowrap;">${{c.drv}}${{pitChip}}</span>` +
        `<span><span class="tyre-dot" style="background:${{ts.bg}};color:${{ts.fg}};">${{ts.ch}}</span><span style="font-size:10px;color:#aaa;margin-left:2px;">${{c.age}}</span></span>` +
        `<span style="color:#00ff88;">${{gapTxt}}</span>` +
        `<span style="color:#ffd700;text-align:right;">${{Math.round(c.speed)}}</span>` +
      `</div>`;
    }}
    towerScroll.innerHTML = html;
  }}

  towerScroll.addEventListener("click", (e) => {{
    const row = e.target.closest(".tower-row");
    if (row && row.dataset.drv) focusDriver = row.dataset.drv;
  }});

  // Main 60 FPS requestAnimationFrame Loop
  let lastFrameTs = performance.now();
  let fpsFrames = 0;
  let fpsTimer = lastFrameTs;

  function frame(nowTs) {{
    const dt = Math.min(0.1, (nowTs - lastFrameTs) / 1000.0);
    lastFrameTs = nowTs;

    if (isPlaying) {{
      subLap += (dt * speedMult) / P.baseLapSec;
      if (subLap >= 1.0) {{
        const wraps = Math.floor(subLap);
        subLap = subLap % 1.0;
        if (curLapIdx + wraps < avail.length) {{
          curLapIdx += wraps;
        }} else {{
          curLapIdx = avail.length - 1;
          subLap = 0.999;
          isPlaying = false;
          setPlayUI();
        }}
      }}
    }}

    const {{ lapNum, trackStatus, cars }} = getInterpolatedCars();
    const egoCar = cars.find(c => c.drv === focusDriver) || cars[0];

    // Update top scrubber & HUD stats
    scrubRange.value = String(Math.round(subLap * 1000));
    hudLap.textContent = `${{lapNum}}/${{P.totalLaps}}`;
    const secLabel = subLap < 0.333 ? "S1" : (subLap < 0.666 ? "S2" : "S3");
    hudSector.innerHTML = `<b>${{secLabel}}</b>`;
    hudDist.textContent = `${{Math.round(subLap * P.trackLenM)}} / ${{Math.round(P.trackLenM)}} m`;

    const elapsed = Math.max(0, (lapNum - 1 + subLap) * P.baseLapSec);
    const hh = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const mm = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const ss = (elapsed % 60).toFixed(2).padStart(5, "0");
    hudClock.innerHTML = `<b>${{hh}}:${{mm}}:${{ss}}</b>`;

    flagPill.textContent = trackStatus;
    flagPill.className = `flag-pill flag-${{trackStatus}}`;

    if (window.parent && window.parent !== window) {{
      if (window.__lastSentLap !== lapNum || window.__lastSentDrv !== focusDriver || window.__lastSentStatus !== trackStatus) {{
        window.__lastSentLap = lapNum;
        window.__lastSentDrv = focusDriver;
        window.__lastSentStatus = trackStatus;
        try {{
          window.parent.postMessage({{ type: "f1_canvas_state", lap: lapNum, driver: focusDriver, status: trackStatus }}, "*");
        }} catch (e) {{}}
      }}
    }}

    drawRadar(cars, trackStatus);
    drawTelemetry(egoCar);
    updateTower(cars, nowTs);

    fpsFrames++;
    if (nowTs - fpsTimer >= 500) {{
      const fps = Math.round((fpsFrames * 1000) / (nowTs - fpsTimer));
      fpsTag.textContent = `${{fps}} FPS`;
      fpsFrames = 0;
      fpsTimer = nowTs;
    }}

    requestAnimationFrame(frame);
  }}

  requestAnimationFrame(frame);
}})();
</script>
</body>
</html>"""
