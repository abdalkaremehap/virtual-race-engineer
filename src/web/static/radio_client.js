/**
 * F1 Pit Wall Radio Comms & Spoken Voice Controller (Zero-Reload ES6 Client).
 * - Sends instant JSON requests to POST /api/radio/ask (~20-40ms)
 * - Renders human_radio_call + instant 0ms <details> More Clarification drawer
 * - Synthesizes F1 Pit Wall dual-tone radio beep + UK Male Race Engineer speech
 */
(function () {
  function escHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatReplyMarkdown(text, accentColor) {
    return String(text || "")
      .split(/\r?\n/)
      .map((raw) => raw.trim())
      .filter(Boolean)
      .map((line) => {
        let e = escHtml(line)
          .replace(/\*\*(.+?)\*\*/g, "<b style='color:#f8fafc;'>$1</b>")
          .replace(
            /`([^`]+)`/g,
            "<code style='background:#0b1220;color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:10.5px;'>$1</code>"
          );
        if (e.startsWith("- ") || e.startsWith("• ")) {
          e = e.slice(2);
          return `<div style="display:flex;gap:6px;margin:4px 0;line-height:1.45;"><span style="color:${accentColor};font-weight:900;">▸</span><span>${e}</span></div>`;
        }
        return `<div style="margin:4px 0;line-height:1.45;">${e}</div>`;
      })
      .join("");
  }

  // ===== F1 TEAM RADIO AUDIO ENGINE =====
  const F1Voice = {
    lastSpokenText: "",
    playBeep(isEnd) {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        const now = ctx.currentTime;
        const tone = (freq, offset, dur, peak) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(freq, now + offset);
          gain.gain.setValueAtTime(0.0001, now + offset);
          gain.gain.exponentialRampToValueAtTime(peak, now + offset + 0.008);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + dur);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(now + offset);
          osc.stop(now + offset + dur + 0.01);
        };
        if (!isEnd) {
          tone(1320, 0.0, 0.055, 0.11);
          tone(1760, 0.075, 0.065, 0.11);
        } else {
          tone(960, 0.0, 0.045, 0.08);
        }
        setTimeout(() => { try { ctx.close(); } catch (e) {} }, 350);
      } catch (e) {}
    },
    pickVoice() {
      const synth = window.speechSynthesis;
      if (!synth) return null;
      const voices = synth.getVoices() || [];
      const prefs = [
        (v) => /Ryan.*Natural|Ryan.*Online/i.test(v.name) && /en-GB/i.test(v.lang),
        (v) => /Google UK English Male/i.test(v.name),
        (v) => /George|Arthur|Daniel|Oliver/i.test(v.name) && /en-GB/i.test(v.lang),
        (v) => /Guy.*Natural|Christopher.*Natural|Andrew.*Natural/i.test(v.name),
        (v) => /en-GB/i.test(v.lang),
        (v) => /^en/i.test(v.lang) && /Male|David|Mark|Guy|Daniel/i.test(v.name),
        (v) => /^en/i.test(v.lang),
      ];
      for (const fn of prefs) {
        const hit = voices.find(fn);
        if (hit) return hit;
      }
      return voices[0] || null;
    },
    stop(statusEl) {
      try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) {}
      if (statusEl) statusEl.textContent = "RADIO MUTED";
    },
    speak(text, statusEl, driverCode) {
      if (!text) return;
      this.lastSpokenText = text;
      const synth = window.speechSynthesis;
      if (!synth) {
        if (statusEl) statusEl.textContent = "BROWSER VOICE UNAVAILABLE";
        return;
      }
      const run = () => {
        try { synth.cancel(); } catch (e) {}
        const v = this.pickVoice();
        const utt = new SpeechSynthesisUtterance(text);
        if (v) utt.voice = v;
        utt.lang = v && v.lang ? v.lang : "en-GB";
        utt.rate = 1.06;
        utt.pitch = 0.96;
        const shortName = v
          ? v.name.replace(/Microsoft |Google |Online \(Natural\) /g, "").split(" - ")[0].slice(0, 18)
          : "UK Pit Wall";
        utt.onstart = () => {
          if (statusEl) statusEl.textContent = `🎙️ LIVE · ${driverCode || "PIT"} (${shortName})`;
        };
        utt.onend = () => {
          this.playBeep(true);
          if (statusEl) statusEl.textContent = `✓ SENT · ${shortName}`;
        };
        utt.onerror = () => {
          if (statusEl) statusEl.textContent = "CLICK PLAY TO HEAR VOICE";
        };
        this.playBeep(false);
        setTimeout(() => {
          try { synth.speak(utt); } catch (e) {}
        }, 140);
      };
      if ((synth.getVoices() || []).length === 0) {
        let fired = false;
        const go = () => { if (!fired) { fired = true; run(); } };
        try { synth.onvoiceschanged = go; } catch (e) {}
        setTimeout(go, 220);
      } else {
        run();
      }
    },
  };

  window.F1Voice = F1Voice;

  window.initPitWallRadio = function initPitWallRadio(opts) {
    const feedEl = document.getElementById(opts.feedId);
    const formEl = document.getElementById(opts.formId);
    const inputEl = document.getElementById(opts.inputId);
    const sendBtn = document.getElementById(opts.sendBtnId);
    const voiceAutoEl = document.getElementById(opts.voiceAutoId);
    const playBtn = document.getElementById(opts.playBtnId);
    const stopBtn = document.getElementById(opts.stopBtnId);
    const voiceStatusEl = document.getElementById(opts.voiceStatusId);
    const clearBtn = document.getElementById(opts.clearBtnId);

    const history = [];

    function renderFeed() {
      if (!history.length) {
        const ctx = opts.getContext();
        feedEl.innerHTML = `
          <div style="margin:auto;text-align:center;padding:28px 14px;color:#94a3b8;">
            <div style="font-size:28px;margin-bottom:8px;">🎧</div>
            <div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">
              Channel Open — ${escHtml(ctx.ego_driver)} Race Engineer Ready
            </div>
            <div style="font-size:11.5px;line-height:1.5;color:#8b949e;">
              Tap a tactical quick-call above or ask any strategy, rival battle, tyre, or FIA rulebook question below.
            </div>
          </div>`;
        return;
      }

      feedEl.innerHTML = history
        .map((item, idx) => {
          const latestBadge =
            idx === 0
              ? `<span style="background:#1d4ed8;color:#eff6ff;padding:1px 6px;border-radius:4px;font-size:9.5px;font-weight:800;margin-left:6px;">LATEST</span>`
              : "";
          const guardPill = item.guardrail_passed
            ? `<span style="background:rgba(0,230,118,0.14);border:1px solid #00e676;color:#69f0ae;padding:1px 6px;border-radius:999px;font-size:9.5px;font-weight:800;">🛡️ VERIFIED</span>`
            : `<span style="background:rgba(255,75,75,0.16);border:1px solid #ff4b4b;color:#ff8a80;padding:1px 6px;border-radius:999px;font-size:9.5px;font-weight:800;">❌ GUARDRAIL</span>`;

          let stratPill = "";
          let stratTable = "";
          if (item.strategy_ranking && item.strategy_ranking.optimal) {
            const opt = item.strategy_ranking.optimal;
            const delta = Number(item.strategy_ranking.optimal_beats_requested_by_sec || 0);
            const dStr = delta >= 0.5 ? ` (-${delta.toFixed(1)}s faster)` : "";
            stratPill = `<div style="display:inline-block;margin-left:6px;background:rgba(56,189,248,0.12);border:1px solid #38bdf8;color:#7dd3fc;padding:2px 7px;border-radius:6px;font-size:10px;font-weight:800;margin-bottom:6px;">🧠 OPTIMAL: ${escHtml(opt.label)}${dStr}</div>`;

            const topOpts = (item.strategy_ranking.top_options || []).slice(0, 3);
            if (topOpts.length) {
              const bestT = Number(topOpts[0].expected_remaining_time_sec || 0);
              const rows = topOpts
                .map((o, i) => {
                  const d = Number(o.expected_remaining_time_sec || bestT) - bestT;
                  const b = i === 0 ? `<span style="color:#34d399;font-weight:800;">BEST</span>` : `<span style="color:#94a3b8;">+${d.toFixed(1)}s</span>`;
                  return `<div style="display:flex;justify-content:space-between;gap:8px;padding:2px 0;font-size:10.5px;"><span style="color:#e2e8f0;">#${i + 1} ${escHtml(o.label)}</span><span style="font-family:monospace;">${b}</span></div>`;
                })
                .join("");
              stratTable = `<div style="margin-top:7px;padding-top:6px;border-top:1px solid #1e293b;"><div style="font-size:9.5px;font-weight:800;color:#7dd3fc;letter-spacing:0.6px;margin-bottom:3px;">TOP RANKED STRATEGIES</div>${rows}</div>`;
            }
          }

          let legalExtra = "";
          if (item.legal_data && (item.legal_data.hop1_primary_rules || []).length) {
            const h1 = item.legal_data.hop1_primary_rules[0];
            const h2s = (item.legal_data.hop2_resolved_references || [])
              .map(
                (h2) =>
                  `<div style="margin-top:4px;color:#c4b5fd;"><b>🔗 Hop-2 Art. ${escHtml(h2.article)} (Page ${escHtml(h2.page)}):</b> ${escHtml(h2.text).slice(0, 260)}...</div>`
              )
              .join("");
            legalExtra = `<div style="margin-top:7px;padding-top:6px;border-top:1px solid #1e293b;font-size:10.5px;color:#ddd6fe;"><div><b>⚖️ FIA Hop-1 (Page ${escHtml(h1.page)} · ${escHtml(h1.score)}%):</b> ${escHtml(h1.text).slice(0, 280)}...</div>${h2s}</div>`;
          }

          const guardNotes = (item.guardrail_notes || []).length
            ? `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #1e293b;font-size:10px;color:#86efac;">🛡️ ${item.guardrail_notes.slice(0, 3).map(escHtml).join(" · ")}</div>`
            : "";

          const proofBody = formatReplyMarkdown(item.engineer_radio_reply, item.verdict_color);

          return `
            <div class="driver-bubble" style="border-right-color:${escHtml(item.team_color)};">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:3px;">
                <span style="font-size:10px;font-weight:800;color:${escHtml(item.team_color)};letter-spacing:0.5px;">🎙️ ${escHtml(item.driver)} #${escHtml(item.driver_num)} · DRIVER</span>
                <span style="font-size:9.5px;color:#64748b;font-family:monospace;">${escHtml(item.lap_tag)}${latestBadge}</span>
              </div>
              <div style="font-size:12px;color:#f1f5f9;">“${escHtml(item.question)}”</div>
            </div>
            <div class="eng-bubble" style="border-left-color:${escHtml(item.verdict_color)};">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:5px;">
                <span style="font-size:10px;font-weight:900;color:#38bdf8;letter-spacing:0.5px;">🎧 RACE ENGINEER · PIT WALL</span>
                <div style="display:flex;align-items:center;gap:5px;">${guardPill}<span style="font-size:9.5px;color:#94a3b8;font-family:monospace;">${escHtml(item.elapsed_ms)}ms</span></div>
              </div>
              <div style="display:flex;flex-wrap:wrap;align-items:center;">
                <div style="display:inline-block;background:rgba(15,23,42,0.95);border:1px solid ${escHtml(item.verdict_color)};color:${escHtml(item.verdict_color)};padding:2px 8px;border-radius:6px;font-size:10.5px;font-weight:800;letter-spacing:0.4px;margin-bottom:6px;">${escHtml(item.verdict_label)}</div>
                ${stratPill}
              </div>
              <div style="font-size:12.8px;color:#f8fafc;line-height:1.5;font-weight:500;">“${escHtml(item.human_radio_call)}”</div>
              <details class="clarify-drawer">
                <summary>📊 More Clarification — Telemetry, 10k MC &amp; Strategy Proof</summary>
                <div class="clarify-body">${proofBody}${stratTable}${legalExtra}${guardNotes}</div>
              </details>
            </div>`;
        })
        .join("");
    }

    async function ask(question) {
      const q = String(question || "").trim();
      if (!q) return;
      if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.textContent = "⏳ ...";
      }
      try {
        const ctx = opts.getContext();
        const resp = await fetch("/api/radio/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...ctx, question: q }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        history.unshift(data);
        if (history.length > 15) history.length = 15;
        renderFeed();
        F1Voice.lastSpokenText = data.spoken_text || data.human_radio_call;
        if (!voiceAutoEl || voiceAutoEl.checked) {
          F1Voice.speak(F1Voice.lastSpokenText, voiceStatusEl, data.driver);
        } else if (voiceStatusEl) {
          voiceStatusEl.textContent = "VOICE MUTED · CLICK PLAY";
        }
      } catch (err) {
        console.error("Radio ask failed:", err);
      } finally {
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.textContent = "🎙️ SEND";
        }
      }
    }

    if (formEl) {
      formEl.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = inputEl ? inputEl.value : "";
        if (inputEl) inputEl.value = "";
        ask(q);
      });
    }
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        const latest = history[0];
        if (latest) F1Voice.speak(latest.spoken_text || latest.human_radio_call, voiceStatusEl, latest.driver);
      });
    }
    if (stopBtn) {
      stopBtn.addEventListener("click", () => F1Voice.stop(voiceStatusEl));
    }
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        history.length = 0;
        F1Voice.stop(voiceStatusEl);
        renderFeed();
      });
    }

    renderFeed();
    return { ask, renderFeed, history };
  };
})();
