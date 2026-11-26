"""
F1 Pit Wall Team Radio Voice Synthesizer (Browser Web Speech API + Web Audio Squelch Beep).

Zero-latency, zero-dependency spoken audio layer that reads the verified `human_radio_call`
out loud with an authentic F1 Pit Wall dual-tone radio beep and automatic UK Race Engineer
voice selection (Microsoft Ryan Natural / Google UK English Male / George / Daniel).

Purely presentational: never alters strategy, physics, Monte Carlo, or guardrail outputs.
"""

from __future__ import annotations

import json
import re


def format_text_for_f1_speech(text: str) -> str:
    """
    Normalizes telemetry shorthand in `human_radio_call` so browser TTS reads numbers
    and F1 acronyms like a real human Pit Wall Race Engineer.
    """
    if not text:
        return ""

    s = str(text).strip()

    # Replace Unicode dashes between numbers ("2–3 laps" -> "2 to 3 laps")
    s = re.sub(r"(\d+)\s*[–—-]\s*(\d+)", r"\1 to \2", s)

    # Signed/unsigned seconds per lap: "+1.45s/lap" or "2.1s/L" -> "1.45 seconds per lap"
    s = re.sub(r"\+?(\d+(?:\.\d+)?)\s*s/(?:lap|L)\b", r"\1 seconds per lap", s, flags=re.IGNORECASE)

    # Signed seconds: "+14.2s" -> "plus 14.2 seconds", "-19.8s" -> "minus 19.8 seconds"
    s = re.sub(r"\+(\d+(?:\.\d+)?)s\b", r"plus \1 seconds", s)
    s = re.sub(r"-(\d+(?:\.\d+)?)s\b", r"minus \1 seconds", s)

    # Unsigned seconds: "22.5s" -> "22.5 seconds"
    s = re.sub(r"(\d+(?:\.\d+)?)s\b", r"\1 seconds", s)

    # Position codes: "P1" -> "P 1", "P12" -> "P 12" (prevents robotic spelling)
    s = re.sub(r"\bP(\d{1,2})\b", r"P \1", s)

    # Strategy label shorthand: "EXTEND SOFT +4L -> MEDIUM" -> "extend Soft 4 laps onto Medium"
    s = re.sub(r"\+(\d+)L\b", r"\1 more laps", s)
    s = re.sub(r"\s*->\s*", " onto ", s)

    # Common F1 radio acronyms
    s = re.sub(r"\bVSC\b", "Virtual Safety Car", s)
    s = re.sub(r"\bSC\b", "Safety Car", s)
    s = re.sub(r"\bFIA\b", "F I A", s)
    s = re.sub(r"\bArt\.\s*", "Article ", s)

    # Clean repeated spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def render_f1_radio_voice_bar_html(
    *,
    speech_text: str,
    message_id: str,
    auto_play: bool = True,
    driver_code: str = "VER",
    team_color: str = "#00E5FF",
) -> str:
    """
    Returns self-contained HTML/JS for the compact Pit Wall Team Radio Audio Bar:
    - Synthesizes the iconic F1 Pit Wall dual-tone radio chime via Web Audio API.
    - Auto-selects the most authentic British Male Race Engineer voice available in the browser.
    - Auto-speaks each new `message_id` exactly once when `auto_play=True`.
    - Provides instant `Play Radio` and `Stop` buttons that always work on user click.
    """
    spoken = format_text_for_f1_speech(speech_text)
    spoken_json = json.dumps(spoken)
    msg_id_json = json.dumps(str(message_id))
    driver_json = json.dumps(str(driver_code))
    auto_json = "true" if auto_play else "false"
    color_safe = re.sub(r"[^#a-zA-Z0-9(),.\s%-]", "", str(team_color)) or "#00E5FF"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }}
  .radio-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: linear-gradient(135deg, #0f172a 0%, #0b1120 100%);
    border: 1px solid #24324a;
    border-left: 3px solid {color_safe};
    border-radius: 8px;
    padding: 6px 9px;
    height: 40px;
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }}
  .btn {{
    appearance: none;
    border: 1px solid #334155;
    background: #162235;
    color: #e2e8f0;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.3px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: background 0.15s ease, border-color 0.15s ease;
  }}
  .btn:hover {{
    background: #1e293b;
    border-color: #38bdf8;
    color: #f8fafc;
  }}
  .btn-play {{
    border-color: rgba(56, 189, 248, 0.55);
    color: #7dd3fc;
  }}
  .btn-stop {{
    border-color: rgba(248, 113, 113, 0.45);
    color: #fca5a5;
  }}
  .status-wrap {{
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    overflow: hidden;
  }}
  .led {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    flex-shrink: 0;
  }}
  .led.talking {{
    background: #ff2800;
    box-shadow: 0 0 9px #ff2800;
    animation: blink 0.6s infinite;
  }}
  @keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.25; }}
  }}
  .status-txt {{
    font-size: 10.5px;
    color: #94a3b8;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
</style>
</head>
<body>
  <div class="radio-bar">
    <div class="controls">
      <button id="playBtn" class="btn btn-play" type="button" title="Hear Pit Wall Race Engineer voice">🔊 PLAY RADIO</button>
      <button id="stopBtn" class="btn btn-stop" type="button" title="Stop audio">⏹ STOP</button>
    </div>
    <div class="status-wrap">
      <span id="led" class="led"></span>
      <span id="statusTxt" class="status-txt">RADIO VOICE READY</span>
    </div>
  </div>
<script>
(function() {{
  const SPEECH_TEXT = {spoken_json};
  const MSG_ID = {msg_id_json};
  const DRIVER = {driver_json};
  const AUTO_PLAY = {auto_json};

  const hostWin = (function() {{
    try {{
      if (window.parent && window.parent.speechSynthesis) return window.parent;
    }} catch (e) {{}}
    return window;
  }})();

  const synth = hostWin.speechSynthesis || window.speechSynthesis;
  const ledEl = document.getElementById("led");
  const statusEl = document.getElementById("statusTxt");
  const playBtn = document.getElementById("playBtn");
  const stopBtn = document.getElementById("stopBtn");

  function setStatus(text, talking) {{
    statusEl.textContent = text;
    if (talking) ledEl.classList.add("talking");
    else ledEl.classList.remove("talking");
  }}

  function playF1RadioBeep(isEnd) {{
    try {{
      const AudioCtx = hostWin.AudioContext || hostWin.webkitAudioContext || window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const now = ctx.currentTime;

      function tone(freq, startOffset, dur, gainPeak) {{
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, now + startOffset);
        gain.gain.setValueAtTime(0.0001, now + startOffset);
        gain.gain.exponentialRampToValueAtTime(gainPeak, now + startOffset + 0.008);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + startOffset + dur);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + startOffset);
        osc.stop(now + startOffset + dur + 0.01);
      }}

      if (!isEnd) {{
        // Iconic F1 Pit Wall open-channel dual chirp
        tone(1320, 0.0, 0.055, 0.11);
        tone(1760, 0.075, 0.065, 0.11);
      }} else {{
        // Short radio squelch close click
        tone(960, 0.0, 0.045, 0.08);
      }}
      setTimeout(() => {{ try {{ ctx.close(); }} catch (e) {{}} }}, 350);
    }} catch (e) {{}}
  }}

  function pickEngineerVoice() {{
    if (!synth) return null;
    const voices = synth.getVoices() || [];
    if (!voices.length) return null;
    const prefs = [
      v => /Ryan.*Natural|Ryan.*Online/i.test(v.name) && /en-GB/i.test(v.lang),
      v => /Google UK English Male/i.test(v.name),
      v => /George|Arthur|Daniel|Oliver/i.test(v.name) && /en-GB/i.test(v.lang),
      v => /Guy.*Natural|Christopher.*Natural|Andrew.*Natural/i.test(v.name),
      v => /en-GB/i.test(v.lang),
      v => /^en/i.test(v.lang) && /Male|David|Mark|Guy|Daniel/i.test(v.name),
      v => /^en/i.test(v.lang),
    ];
    for (const fn of prefs) {{
      const hit = voices.find(fn);
      if (hit) return hit;
    }}
    return voices[0];
  }}

  function stopSpeech() {{
    try {{
      if (synth) synth.cancel();
    }} catch (e) {{}}
    setStatus("RADIO MUTED", false);
  }}

  function speakNow(manualTrigger) {{
    if (!synth || !SPEECH_TEXT) {{
      setStatus("BROWSER VOICE UNAVAILABLE", false);
      return;
    }}

    const runSpeak = () => {{
      try {{
        synth.cancel();
      }} catch (e) {{}}

      const voice = pickEngineerVoice();
      const utt = new (hostWin.SpeechSynthesisUtterance || SpeechSynthesisUtterance)(SPEECH_TEXT);
      if (voice) utt.voice = voice;
      utt.lang = (voice && voice.lang) ? voice.lang : "en-GB";
      utt.rate = 1.06;
      utt.pitch = 0.96;
      utt.volume = 1.0;

      const shortVoiceName = voice
        ? voice.name.replace(/Microsoft |Google |Online \\(Natural\\) /g, "").split(" - ")[0].slice(0, 18)
        : "UK Pit Wall";

      utt.onstart = () => {{
        setStatus("🎙️ LIVE · " + DRIVER + " (" + shortVoiceName + ")", true);
      }};
      utt.onend = () => {{
        playF1RadioBeep(true);
        setStatus("✓ SENT · " + shortVoiceName, false);
      }};
      utt.onerror = () => {{
        setStatus("CLICK PLAY TO HEAR VOICE", false);
      }};

      playF1RadioBeep(false);
      setTimeout(() => {{
        try {{
          synth.speak(utt);
        }} catch (e) {{
          setStatus("CLICK PLAY TO HEAR VOICE", false);
        }}
      }}, 140);
    }};

    // Chrome/Edge populate getVoices() asynchronously on first load
    if ((synth.getVoices() || []).length === 0) {{
      let fired = false;
      const onReady = () => {{
        if (fired) return;
        fired = true;
        runSpeak();
      }};
      try {{ synth.onvoiceschanged = onReady; }} catch (e) {{}}
      setTimeout(onReady, 220);
    }} else {{
      runSpeak();
    }}
  }}

  playBtn.addEventListener("click", () => speakNow(true));
  stopBtn.addEventListener("click", stopSpeech);

  // Auto-speak each new Pit Wall message exactly once
  if (AUTO_PLAY && SPEECH_TEXT && MSG_ID) {{
    try {{
      if (hostWin.__f1LastSpokenMsgId !== MSG_ID) {{
        hostWin.__f1LastSpokenMsgId = MSG_ID;
        speakNow(false);
      }} else {{
        const v = pickEngineerVoice();
        const vn = v ? v.name.replace(/Microsoft |Google |Online \\(Natural\\) /g, "").split(" - ")[0].slice(0, 18) : "UK Pit Wall";
        setStatus("READY · " + vn, false);
      }}
    }} catch (e) {{
      speakNow(false);
    }}
  }} else {{
    setStatus(SPEECH_TEXT ? "VOICE MUTED · CLICK PLAY" : "WAITING FOR RADIO", false);
  }}
}})();
</script>
</body>
</html>
"""
