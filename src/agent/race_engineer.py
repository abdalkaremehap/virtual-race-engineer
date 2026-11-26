import os
import re
import sys
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.tools import (
    RaceContext,
    RaceEngineerToolkit,
    _analyze_rival_catch_battle_impl,
    _simulate_pit_stop_mc_impl,
    analyze_rival_catch_battle,
    compute_remaining_stint_pit_breakeven,
    consult_fia_rulebook_multihop,
    get_live_race_standings,
    predict_tyre_degradation,
    simulate_pit_stop_mc,
)
from src.ml_surrogates.tensor_cache import TyreL1TensorCache

DRIVER_NAME_TO_CODE = {
    "verstappen": "VER",
    "max": "VER",
    "ver": "VER",
    "perez": "PER",
    "checo": "PER",
    "per": "PER",
    "leclerc": "LEC",
    "charles": "LEC",
    "lec": "LEC",
    "sainz": "SAI",
    "carlos": "SAI",
    "sai": "SAI",
    "norris": "NOR",
    "lando": "NOR",
    "nor": "NOR",
    "piastri": "PIA",
    "oscar": "PIA",
    "pia": "PIA",
    "russell": "RUS",
    "george": "RUS",
    "rus": "RUS",
    "hamilton": "HAM",
    "lewis": "HAM",
    "ham": "HAM",
    "alonso": "ALO",
    "fernando": "ALO",
    "alo": "ALO",
    "stroll": "STR",
    "lance": "STR",
    "str": "STR",
    "tsunoda": "TSU",
    "yuki": "TSU",
    "tsu": "TSU",
    "ricciardo": "RIC",
    "daniel": "RIC",
    "ric": "RIC",
    "hulkenberg": "HUL",
    "nico": "HUL",
    "hul": "HUL",
    "magnussen": "MAG",
    "kevin": "MAG",
    "mag": "MAG",
    "albon": "ALB",
    "alex": "ALB",
    "alb": "ALB",
    "gasly": "GAS",
    "pierre": "GAS",
    "gas": "GAS",
    "ocon": "OCO",
    "esteban": "OCO",
    "oco": "OCO",
    "zhou": "ZHO",
    "zho": "ZHO",
    "bottas": "BOT",
    "valtteri": "BOT",
    "bot": "BOT",
    "sargeant": "SAR",
    "logan": "SAR",
    "sar": "SAR",
}

DRIVER_FIRST_NAMES = {
    "VER": "Max",
    "PER": "Checo",
    "LEC": "Charles",
    "SAI": "Carlos",
    "NOR": "Lando",
    "PIA": "Oscar",
    "RUS": "George",
    "HAM": "Lewis",
    "ALO": "Fernando",
    "STR": "Lance",
    "TSU": "Yuki",
    "RIC": "Daniel",
    "HUL": "Nico",
    "MAG": "Kevin",
    "ALB": "Alex",
    "GAS": "Pierre",
    "OCO": "Esteban",
    "ZHO": "Zhou",
    "BOT": "Valtteri",
    "SAR": "Logan",
    "VET": "Seb",
    "MSC": "Mick",
    "LAT": "Nicky",
    "MAZ": "Nikita",
    "GIO": "Antonio",
    "RAI": "Kimi",
    "DEV": "Nyck",
    "LAW": "Liam",
    "BEA": "Ollie",
    "COL": "Franco",
}


def _compound_plural(comp: str) -> str:
    """Formats an F1 compound code into natural spoken radio English."""
    c = str(comp or "").strip().upper()
    mapping = {
        "SOFT": "Softs",
        "MEDIUM": "Mediums",
        "HARD": "Hards",
        "INTERMEDIATE": "Intermediates",
        "WET": "Wets",
    }
    return mapping.get(c, f"{c.title()}s" if c else "tyres")


def _driver_first_name(code: str) -> str:
    """Returns natural spoken driver first name for authentic F1 Pit Wall radio calls."""
    c = str(code or "").strip().upper()
    return DRIVER_FIRST_NAMES.get(c, c or "mate")


_PER_NON_DRIVER_NEXT = {
    "lap", "laps", "sec", "second", "seconds", "min", "minute", "minutes",
    "stint", "stop", "stops", "cent", "km", "m", "hour", "race", "session",
    "tyre", "tyres", "tire", "tires", "set", "fia", "rule", "rules", "article",
    "turn", "corner", "sector",
}
_MAX_NON_DRIVER_NEXT = {
    "speed", "rpm", "throttle", "brake", "lap", "laps", "time", "temp",
    "temperature", "deg", "wear", "grip", "fuel", "load", "downforce",
    "drs", "gear", "pace", "attack", "power", "deployment", "deploy", "ers",
    "pressure", "stint", "out", "push", "effort", "delta", "limit", "limits",
    "gap", "revs", "mode", "charge", "harvest", "energy", "boost", "wing",
    "aero", "tyre", "tyres", "tire", "tires",
}
_MAX_NON_DRIVER_PREV = {"at", "the", "full"}
_GAS_NON_DRIVER_CONTEXT = {
    "the", "full", "off", "on", "pedal", "more", "less", "no",
    "save", "saving", "fuel", "feather", "feathering", "flat", "apply", "applying",
}
_NOR_NON_DRIVER_NEXT = {
    "hard", "hards", "soft", "softs", "medium", "mediums", "inter", "inters",
    "intermediate", "wet", "wets", "tyre", "tyres", "tire", "tires",
    "grip", "pace", "speed", "drs", "ers", "fuel", "pit", "box", "can",
    "do", "does", "did", "will", "would", "should", "could", "is", "are",
    "was", "were", "have", "has", "am", "the", "a", "an", "my", "our",
    "his", "their", "any", "enough", "i", "we", "you", "it", "they",
}
_STR_NON_DRIVER_NEXT = {"mode", "modes", "plan", "switch", "setting", "settings"}
_DRIVER_REF_PREV = {
    "catch", "catching", "pass", "passing", "overtake", "overtaking",
    "behind", "ahead", "vs", "versus", "to", "against", "beat", "beating",
    "undercut", "undercutting", "overcut", "overcutting", "from", "defend",
    "defending", "chase", "chasing", "fight", "fighting", "battle", "battling",
    "is", "where", "gap",
}


def _compile_keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Compiles a case-insensitive word-boundary regex so short tokens never match inside larger words."""
    escaped = [re.escape(k) for k in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_TYRE_INTENT_PATTERN = _compile_keyword_pattern([
    "tyre", "tyres", "tire", "tires", "deg", "degradation", "degrading",
    "wear", "wearing", "worn", "grip", "box", "boxing", "pit", "pits",
    "pitting", "pitted", "pit stop", "pit stops", "pitstop", "pitstops",
    "stint", "stints", "strategy", "losing", "lose", "pace", "lap time",
    "lap times", "laptime", "laptimes", "cooked", "life", "graining",
    "blistering", "cliff", "overheating",
])

_BATTLE_INTENT_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(k)
        for k in sorted(
            [
                "catch", "catching", "pass", "passing", "passed",
                "overtake", "overtaking", "overtaken", "gap", "gaps",
                "ahead", "behind", "rival", "rivals", "delta", "deltas",
                "close", "closing", "closer", "chase", "chasing",
                "defend", "defending", "battle", "battling", "fight", "fighting",
                "faster", "slower", "leader", "position", "positions",
                "where is", "where's",
            ],
            key=len,
            reverse=True,
        )
    )
    + r"|p(?:[1-9]|1[0-9]|20))\b",
    re.IGNORECASE,
)

_PIT_INTENT_PATTERN = _compile_keyword_pattern([
    "box", "boxing", "pit", "pits", "pitting", "pitted", "pit stop", "pit stops",
    "pitstop", "pitstops", "undercut", "undercutting", "overcut", "overcutting",
    "rejoin", "rejoining", "traffic", "sc", "vsc", "safety car",
    "virtual safety car", "strategy", "window", "windows", "stop", "stops",
])

_LEGAL_INTENT_PATTERN = _compile_keyword_pattern([
    "penalty", "penalties", "rule", "rules", "rulebook", "regulation",
    "regulations", "legal", "illegal", "allowed", "steward", "stewards",
    "sc", "vsc", "safety car", "virtual safety car", "article", "articles",
    "fia", "minimum time", "track limits", "track limit", "pit exit",
    "pit entry", "pit lane", "unsafe release", "yellow flag", "blue flag",
    "red flag", "white line", "safety car line", "sc line", "blend line",
    "control line",
])

_LEGAL_CROSS_LINE_PATTERN = re.compile(
    r"\bcross(?:ing|ed)?\s+(?:the\s+)?(?:white\s+|pit\s+|safety\s+car\s+|sc\s+)?line\b",
    re.IGNORECASE,
)

_OVERTAKE_ACTION_PATTERN = _compile_keyword_pattern([
    "overtake", "overtaking", "overtaken", "pass", "passing", "passed",
])

_OVERTAKE_LEGAL_CONTEXT_PATTERN = _compile_keyword_pattern([
    "under", "behind", "during", "while", "line", "pit", "flag", "flags",
    "car deployed", "sc", "vsc", "safety car", "virtual safety car",
])

_EXPLICIT_LEGAL_QUESTION_PATTERN = _compile_keyword_pattern([
    "penalty", "penalties", "rule", "rules", "rulebook", "regulation",
    "regulations", "legal", "illegal", "allowed", "steward", "stewards",
    "article", "articles", "fia", "minimum time", "track limits", "track limit",
    "pit exit", "pit entry", "pit lane", "unsafe release", "yellow flag",
    "blue flag", "red flag", "white line", "safety car line", "sc line",
    "blend line", "control line",
])

_EXPLICIT_STRATEGY_PATTERN = _compile_keyword_pattern([
    "box", "boxing", "pit stop", "pit stops", "pitstop", "pitstops",
    "pitting", "pitted", "undercut", "undercutting", "overcut", "overcutting",
    "rejoin", "rejoining", "strategy", "stay out", "tyre", "tyres", "tire", "tires",
    "wear", "deg", "degradation", "soft", "softs", "medium", "mediums",
    "hard", "hards", "inter", "inters", "intermediate", "intermediates",
    "wet", "wets", "stint", "cliff", "pace", "lap time", "laptime",
])

_TARGET_TYRE_PREPOSITION_PATTERN = re.compile(
    r"\b(?:for|onto|on\s+to|switch\s+to|change\s+to|fit|put\s+on)\s+"
    r"(?:fresh\s+|new\s+|used\s+)?(softs?|mediums?|hards?|intermediates?|inters?|wets?)\b",
    re.IGNORECASE,
)
_ANY_TYRE_COMPOUND_PATTERN = re.compile(
    r"\b(softs?|mediums?|hards?|intermediates?|inters?|wets?)\b",
    re.IGNORECASE,
)
_TYRE_TOKEN_NORMALIZE = {
    "soft": "SOFT",
    "softs": "SOFT",
    "medium": "MEDIUM",
    "mediums": "MEDIUM",
    "hard": "HARD",
    "hards": "HARD",
    "inter": "INTERMEDIATE",
    "inters": "INTERMEDIATE",
    "intermediate": "INTERMEDIATE",
    "intermediates": "INTERMEDIATE",
    "wet": "WET",
    "wets": "WET",
}


def extract_target_tyre(
    message: str,
    current_compound: str,
    laps_remaining: int | None = None,
    weather_condition: str | None = None,
    track_temp: float | None = None,
) -> str:
    """
    Extracts the intended target pit compound using word boundaries and pit-preposition priority.
    When no compound is explicitly mentioned, selects INTERMEDIATE/WET in damp/rainy weather
    or a stint-length & track-temp appropriate dry compound (avoiding cold HARDs below 29°C).
    """
    prep_match = _TARGET_TYRE_PREPOSITION_PATTERN.search(message)
    if prep_match:
        return _TYRE_TOKEN_NORMALIZE[prep_match.group(1).lower()]

    mentioned = [
        _TYRE_TOKEN_NORMALIZE[m.group(1).lower()]
        for m in _ANY_TYRE_COMPOUND_PATTERN.finditer(message)
    ]
    if mentioned:
        cur_upper = current_compound.upper()
        for comp in mentioned:
            if comp != cur_upper:
                return comp
        return mentioned[0]

    return TyreL1TensorCache.recommended_compound_for_weather(
        weather_condition=weather_condition,
        laps_remaining=laps_remaining,
        current_compound=current_compound,
        track_temp=track_temp,
    )


def extract_mentioned_driver(message: str, ego_driver: str) -> str | None:
    """
    Detects any F1 rival driver 3-letter code or surname/first name mentioned in the radio message,
    filtering out English homographs ('per lap', 'max speed', 'on the gas', 'neither ... nor ...',
    'hey bot', 'str mode') and ego_driver self-matches.
    """
    raw_tokens = re.findall(r"[A-Za-z]+", message)
    lower_tokens = [t.lower() for t in raw_tokens]
    ego_upper = ego_driver.upper()

    for i, tok in enumerate(lower_tokens):
        if tok not in DRIVER_NAME_TO_CODE:
            continue
        prev_tok = lower_tokens[i - 1] if i > 0 else ""
        next_tok = lower_tokens[i + 1] if i + 1 < len(lower_tokens) else ""

        # Disambiguate English preposition 'per' (e.g., 'per lap', 'per second') vs 'PER' (Sergio Perez)
        if tok == "per" and (
            next_tok in _PER_NON_DRIVER_NEXT
            or (raw_tokens[i] != "PER" and prev_tok not in _DRIVER_REF_PREV)
        ):
            continue
        # Disambiguate English adjective/noun 'max' (e.g., 'max speed', 'at max') vs 'Max' (Max Verstappen)
        if tok == "max" and (
            next_tok in _MAX_NON_DRIVER_NEXT
            or (raw_tokens[i] not in {"Max", "MAX"} and prev_tok in _MAX_NON_DRIVER_PREV)
        ):
            continue
        # Disambiguate English noun 'gas' (e.g., 'on the gas', 'gas pedal') vs 'GAS' (Pierre Gasly)
        if tok == "gas" and raw_tokens[i] != "GAS" and (
            prev_tok in _GAS_NON_DRIVER_CONTEXT or next_tok in _GAS_NON_DRIVER_CONTEXT
        ):
            continue
        # Disambiguate English conjunction 'nor' (e.g., 'neither soft nor hard') vs 'NOR' (Lando Norris)
        if tok == "nor" and raw_tokens[i] != "NOR" and (
            "neither" in lower_tokens[:i]
            or next_tok in _NOR_NON_DRIVER_NEXT
            or prev_tok not in _DRIVER_REF_PREV
        ):
            continue
        # Disambiguate English noun 'bot' (e.g., 'hey bot') vs 'BOT' (Valtteri Bottas)
        if tok == "bot" and raw_tokens[i] != "BOT" and prev_tok not in _DRIVER_REF_PREV:
            continue
        # Disambiguate steering-wheel shorthand 'str mode' vs 'STR' (Lance Stroll)
        if tok == "str" and raw_tokens[i] != "STR" and next_tok in _STR_NON_DRIVER_NEXT:
            continue

        code = DRIVER_NAME_TO_CODE[tok]
        if code != ego_upper:
            return code
    return None


class EngineerAgentState(TypedDict):
    driver: str
    radio_message: str
    context: RaceContext | None
    required_tools: list[str]
    target_rival: str | None
    live_data: dict
    tyre_data: dict
    battle_data: dict
    pit_sim_data: dict
    legal_data: dict
    guardrail_passed: bool
    guardrail_notes: list[str]
    engineer_radio_reply: str
    human_radio_call: str
    elapsed_ms: float


class AutonomousRaceEngineer:
    """
    Phase 5 (Tasks 5.3, 5.4, 5.5): LangGraph Multi-Hop Race Engineer Brain
    with Deterministic Hallucination Guardrails and Tactical Rival Battle Analysis.
    """

    def __init__(self, ego_driver: str = "VER"):
        self.ego_driver = ego_driver
        self.graph = self._build_langgraph()

    def _build_langgraph(self):
        workflow = StateGraph(EngineerAgentState)

        workflow.add_node("router", self._route_intent_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("guardrail", self._hallucination_guardrail_node)
        workflow.add_node("synthesize_radio", self._synthesize_radio_node)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "execute_tools")
        workflow.add_edge("execute_tools", "guardrail")
        workflow.add_edge("guardrail", "synthesize_radio")
        workflow.add_edge("synthesize_radio", END)

        return workflow.compile()

    def _route_intent_node(self, state: EngineerAgentState) -> dict:
        """Node 1: Analyzes driver radio transmission using word-boundary patterns to select required tools."""
        raw_msg = state["radio_message"]
        ego = state["driver"].upper()
        tools_to_run = ["get_live_race_standings"]
        target_rival = extract_mentioned_driver(raw_msg, ego)

        # 4. Multi-Hop FIA Legal Advisory (only triggered for explicit legal/penalty/rulebook/flag queries)
        is_overtake_legal = bool(
            _OVERTAKE_ACTION_PATTERN.search(raw_msg)
            and _OVERTAKE_LEGAL_CONTEXT_PATTERN.search(raw_msg)
        )
        is_explicit_legal = bool(
            _EXPLICIT_LEGAL_QUESTION_PATTERN.search(raw_msg)
            or _LEGAL_CROSS_LINE_PATTERN.search(raw_msg)
            or is_overtake_legal
        )
        is_any_legal = bool(is_explicit_legal or _LEGAL_INTENT_PATTERN.search(raw_msg))

        # Pure FIA Rulebook question (e.g. "can i overtake under the sc?", "can I pass under yellow flag?",
        # "am I allowed to cross the white line at pit exit?"): do NOT clutter with unrelated rival battle or pit sim!
        if (
            is_explicit_legal
            and target_rival is None
            and not _EXPLICIT_STRATEGY_PATTERN.search(raw_msg)
        ):
            tools_to_run = ["get_live_race_standings", "consult_fia_rulebook_multihop"]
            logger.info(f"🧭 [LangGraph Router] Selected tools (Pure FIA Legal): {tools_to_run} | Target rival: None")
            return {"required_tools": tools_to_run, "target_rival": None}

        if _TYRE_INTENT_PATTERN.search(raw_msg):
            tools_to_run.append("predict_tyre_degradation")

        if target_rival is not None or _BATTLE_INTENT_PATTERN.search(raw_msg):
            tools_to_run.append("analyze_rival_catch_battle")
            if "predict_tyre_degradation" not in tools_to_run:
                tools_to_run.append("predict_tyre_degradation")
            if "simulate_pit_stop_mc" not in tools_to_run:
                tools_to_run.append("simulate_pit_stop_mc")

        if _PIT_INTENT_PATTERN.search(raw_msg):
            if "simulate_pit_stop_mc" not in tools_to_run:
                tools_to_run.append("simulate_pit_stop_mc")

        if is_any_legal:
            tools_to_run.append("consult_fia_rulebook_multihop")

        # Fallback: If the user typed a general conversational or tactical query that didn't hit a specific keyword,
        # always run full telemetry + tyre physics + rival catch battle + pit simulation so the engineer never returns an empty header!
        if len(tools_to_run) == 1:
            tools_to_run.extend([
                "predict_tyre_degradation",
                "analyze_rival_catch_battle",
                "simulate_pit_stop_mc",
            ])

        logger.info(f"🧭 [LangGraph Router] Selected tools: {tools_to_run} | Target rival: {target_rival}")
        return {"required_tools": tools_to_run, "target_rival": target_rival}

    def _execute_tools_node(self, state: EngineerAgentState) -> dict:
        """Node 2: Invokes the selected Phase 1-4 tools using isolated per-request RaceContext when supplied."""
        req = state["required_tools"]
        driver = state["driver"]
        target_rival = state.get("target_rival")
        ctx = state.get("context")

        tk = RaceEngineerToolkit.get()
        if ctx is not None:
            active_pit_sim = tk.get_pit_sim(
                circuit=ctx.circuit,
                team_pit_box_position=ctx.team_pit_box_position,
            )
            active_total_laps = int(ctx.total_race_laps)
            live_data = ctx.live_data if ctx.live_data is not None else get_live_race_standings.invoke({})
        else:
            active_pit_sim = tk.pit_sim
            active_total_laps = int(tk.total_race_laps)
            live_data = get_live_race_standings.invoke({})

        standings = live_data.get("standings", [])
        ego_row = next((r for r in standings if r["driver"] == driver), None)

        compound = ego_row["compound"] if ego_row else "SOFT"
        tyre_life = int(ego_row["tyre_life"]) if ego_row else 15
        track_temp = float(live_data.get("track_temp_c", 32.0))
        track_status = live_data.get("track_status", "GREEN")
        weather_cond = live_data.get("weather_condition") or (ctx.weather_condition if ctx is not None else None)
        merged_extras = {}
        if weather_cond and "weather_condition" not in live_data:
            merged_extras["weather_condition"] = weather_cond
        if ctx is not None:
            if ctx.used_dry_compounds and not live_data.get("used_dry_compounds"):
                merged_extras["used_dry_compounds"] = ctx.used_dry_compounds
            if ctx.teammate_gap_ahead_sec is not None and live_data.get("teammate_gap_ahead_sec") is None:
                merged_extras["teammate_gap_ahead_sec"] = ctx.teammate_gap_ahead_sec
            if ctx.teammate_pitting and not live_data.get("teammate_pitting"):
                merged_extras["teammate_pitting"] = True
        if merged_extras:
            live_data = {**live_data, **merged_extras}
        cur_lap = int(live_data.get("lap", 17))
        tot_laps = max(cur_lap, int(live_data.get("total_race_laps", active_total_laps)))
        laps_remaining = max(0, tot_laps - cur_lap)

        target_tyre = extract_target_tyre(
            state["radio_message"],
            compound,
            laps_remaining=laps_remaining,
            weather_condition=weather_cond,
            track_temp=track_temp,
        )

        active_circuit_name = str(live_data.get("circuit", active_pit_sim.circuit))
        tyre_data = {}
        if "predict_tyre_degradation" in req:
            tyre_data = predict_tyre_degradation.invoke({
                "driver": driver,
                "compound": compound,
                "tyre_life": tyre_life,
                "track_temp": track_temp,
                "race_lap": cur_lap,
                "weather_condition": weather_cond,
                "circuit": active_circuit_name,
            })

        battle_data = {}
        cached_rejoin = None
        if "analyze_rival_catch_battle" in req:
            battle_data = _analyze_rival_catch_battle_impl(
                ego_driver=driver,
                target_driver=target_rival,
                target_new_tyre=target_tyre,
                live_data=live_data,
                pit_sim=active_pit_sim,
                total_race_laps=tot_laps,
            )
            cached_rejoin = battle_data.get("rejoin_simulation")

        pit_sim_data = {}
        if "simulate_pit_stop_mc" in req:
            rival_gap = 1.5
            if battle_data and "gap_between_sec" in battle_data and battle_data["gap_between_sec"] > 0:
                rival_gap = float(battle_data["gap_between_sec"])
            elif len(standings) >= 2:
                rival_gap = abs(standings[1]["gap_to_leader"] - standings[0]["gap_to_leader"])
            pit_sim_data = _simulate_pit_stop_mc_impl(
                driver=driver,
                race_condition=track_status,
                target_new_tyre=target_tyre,
                rival_gap_sec=max(0.3, rival_gap),
                live_data=live_data,
                rejoin_sim=cached_rejoin,
                pit_sim=active_pit_sim,
                total_race_laps=tot_laps,
            )

        legal_data = {}
        if "consult_fia_rulebook_multihop" in req:
            raw_msg = state["radio_message"]
            legal_match = re.search(
                r"(what\s+(?:is|are)\s+the\s+(?:exact\s+)?(?:fia\s+)?(?:penalty|rules?|regulations?).*)",
                raw_msg,
                re.IGNORECASE,
            )
            clean_legal_query = legal_match.group(1) if legal_match else raw_msg
            legal_data = consult_fia_rulebook_multihop.invoke({
                "legal_question": clean_legal_query
            })

        return {
            "live_data": live_data,
            "tyre_data": tyre_data,
            "battle_data": battle_data,
            "pit_sim_data": pit_sim_data,
            "legal_data": legal_data,
        }

    def _hallucination_guardrail_node(self, state: EngineerAgentState) -> dict:
        """
        Node 3 (Task 5.4): Deterministic Mathematical & Legal Citation Guardrail.
        Verifies no NaN/null physics outputs and ensures legal citations exist before speaking.
        """
        notes = []
        passed = True

        if state.get("tyre_data"):
            deg = state["tyre_data"].get("current_deg_sec_per_lap")
            if deg is None or not (0.0 <= deg <= 18.0):
                passed = False
                notes.append("Invalid tyre degradation value blocked.")
            else:
                notes.append(f"Verified L1 Tensor Deg: +{deg:.3f}s/lap")

        if state.get("battle_data") and "error" not in state["battle_data"]:
            bd = state["battle_data"]
            gap = bd.get("gap_between_sec")
            if gap is None or gap < 0.0:
                passed = False
                notes.append("Invalid rival gap blocked.")
            else:
                notes.append(
                    f"Verified Rival Battle vs {bd['target_driver']} (P{bd['target_position']}, Gap: {gap:.2f}s)"
                )

        if state.get("pit_sim_data"):
            rejoin = state["pit_sim_data"].get("rejoin_simulation", {})
            pos = rejoin.get("median_pos")
            if pos is None or not (1 <= pos <= 25):
                passed = False
                notes.append("Invalid Monte Carlo rejoin position blocked.")
            else:
                notes.append(f"Verified 10k MC Rejoin: P{pos} (Loss: {rejoin.get('median_pit_loss'):.2f}s)")
            strat = state["pit_sim_data"].get("strategy_ranking", {})
            opt = strat.get("optimal")
            if opt and opt.get("label"):
                notes.append(f"Optimal Strategy: {opt['label']}")

        # Self-consistency check between Rival Battle & Pit Wall Sim
        if state.get("battle_data") and state.get("pit_sim_data") and "error" not in state["battle_data"]:
            b_verdict = str(state["battle_data"].get("tactical_verdict", "")).upper()
            stint_be = state["pit_sim_data"].get("stint_breakeven", {})
            be_pos = bool(stint_be.get("pit_breakeven_reached", False))
            if "DO NOT BOX" in b_verdict and be_pos and float(stint_be.get("net_stint_pit_payback_sec", 0.0)) > 5.0:
                notes.append("Aligned Battle & Pit directives via Optimizer")

        if state.get("legal_data"):
            ld = state["legal_data"]
            h1 = ld.get("hop1_primary_rules", [])
            h2 = ld.get("hop2_resolved_references", [])
            if ld.get("offline"):
                notes.append("FIA Rulebook offline (degraded gracefully to telemetry/strategy)")
            elif not h1:
                passed = False
                notes.append("No FIA regulation found in Qdrant.")
            else:
                resolved_arts = [r["article"] for r in h2]
                notes.append(
                    f"Verified FIA Rule Page {h1[0]['page']} + Hop-2 Articles {resolved_arts}"
                )

        logger.success(f"🛡️ [Hallucination Guardrail] Passed={passed} | Checks={notes}")
        return {"guardrail_passed": passed, "guardrail_notes": notes}

    def _synthesize_radio_node(self, state: EngineerAgentState) -> dict:
        """
        Node 4: Synthesizes a concise, zero-hallucination F1 Race Engineer radio call
        grounded strictly in the verified tool outputs.
        """
        if not state["guardrail_passed"]:
            fallback_msg = "📻 [ENGINEER]: Telemetry validation failed on pit wall. Stand by, checking data."
            return {
                "engineer_radio_reply": fallback_msg,
                "human_radio_call": "Stand by, checking the pit-wall telemetry feed — hold position for a second.",
            }

        live = state["live_data"]
        lap = live.get("lap", "?")
        total_laps = live.get("total_race_laps", 57)
        status = live.get("track_status", "GREEN")
        parts = [f"📻 [RACE ENGINEER -> {state['driver']} | Lap {lap}/{total_laps} ({status})]:"]
        pit_human_ctx: dict | None = None

        # 1. Rival Catch / Gap / On-Track Battle Analysis
        if state.get("battle_data") and "error" not in state["battle_data"]:
            bd = state["battle_data"]
            parts.append(
                f"• RIVAL BATTLE ({state['driver']} P{bd['ego_position']} vs {bd['target_driver']} P{bd['target_position']}): "
                f"Gap is {bd['gap_between_sec']:.2f}s | Your last lap: {bd['ego_lap_time_sec']:.3f}s "
                f"({bd['ego_compound']} {bd['ego_tyre_life']}L, +{bd['ego_deg_sec_per_lap']:.2f}s/L deg) vs "
                f"{bd['target_driver']}: {bd['target_lap_time_sec']:.3f}s "
                f"({bd['target_compound']} {bd['target_tyre_life']}L, +{bd['target_deg_sec_per_lap']:.2f}s/L deg).\n"
                f"  ↳ 🎯 VERDICT: **{bd['tactical_verdict']}**"
            )

        # 2. Physics & Tyre Degradation Summary
        if state.get("tyre_data"):
            td = state["tyre_data"]
            parts.append(
                f"• TYRE PHYSICS: Your {td['current_tyre_life']}-lap {td['compound']}s are losing "
                f"+{td['current_deg_sec_per_lap']:.2f}s/lap (projected +{td['cumulative_5lap_loss_sec']:.2f}s over the next 5 laps)."
            )

        # 3. Monte Carlo Pit & Undercut Strategy Call (with End-of-Race Remaining-Laps Breakeven Physics)
        if state.get("pit_sim_data"):
            rejoin = state["pit_sim_data"]["rejoin_simulation"]
            uc = state["pit_sim_data"]["undercut_simulation"]
            stint_be = state["pit_sim_data"].get("stint_breakeven", {})
            loss = float(rejoin["median_pit_loss"])
            green_loss = float(rejoin.get("green_flag_pit_loss", loss))
            saved = float(rejoin.get("pit_loss_saved_vs_green", max(0.0, green_loss - loss)))

            standings = live.get("standings", [])
            ego_row = next((r for r in standings if r["driver"] == state["driver"]), None)
            ego_pos = int(
                state.get("battle_data", {}).get(
                    "ego_position",
                    stint_be.get("ego_position", ego_row["position"] if ego_row else 1),
                )
            )
            ego_comp = str(
                state.get("tyre_data", {}).get(
                    "compound",
                    ego_row["compound"] if ego_row else "SOFT",
                )
            ).upper()
            ego_age = int(
                state.get("tyre_data", {}).get(
                    "current_tyre_life",
                    ego_row["tyre_life"] if ego_row else 15,
                )
            )
            ego_deg = float(state.get("tyre_data", {}).get("current_deg_sec_per_lap", 0.0))
            track_temp = float(live.get("track_temp_c", 32.0))
            cur_lap_int = int(lap) if isinstance(lap, (int, float)) or str(lap).isdigit() else 17
            tot_laps_int = max(cur_lap_int, int(total_laps))
            laps_remaining = max(0, tot_laps_int - cur_lap_int)

            if stint_be and "recoverable_tyre_gain_sec" in stint_be:
                recoverable_gain = float(stint_be["recoverable_tyre_gain_sec"])
                net_payback = float(stint_be["net_stint_pit_payback_sec"])
                fresh_comp = str(stint_be.get("fresh_compound", "SOFT" if laps_remaining <= 15 else "MEDIUM"))
            else:
                be_calc = compute_remaining_stint_pit_breakeven(
                    ego_compound=ego_comp,
                    ego_tyre_life=ego_age,
                    track_temp=track_temp,
                    laps_remaining=laps_remaining,
                    pit_loss_sec=loss,
                )
                recoverable_gain = float(be_calc["recoverable_tyre_gain_sec"])
                net_payback = float(be_calc["net_stint_pit_payback_sec"])
                fresh_comp = str(be_calc["fresh_compound"])

            weather_cond = live.get("weather_condition")
            weather_penalty = TyreL1TensorCache.get_weather_compound_penalty(ego_comp, weather_cond)
            wrong_weather_target = bool(stint_be.get("wrong_weather_target", False))
            opt_weather_comp = str(stint_be.get("optimal_weather_compound", ""))
            rej_reasons = stint_be.get("rejection_reasons") or []
            alt_rec = str(stint_be.get("alternative_recommendation", ""))

            msg_low = state.get("radio_message", "").lower()
            drying_crossover_hint = any(
                k in msg_low for k in ("drying", "dry line", "skip inter", "wait for slick", "straight to slick")
            )

            free_sc_stop = bool(
                status == "SC"
                and int(rejoin["median_pos"]) <= ego_pos
                and float(rejoin["p_clean_air"]) >= 0.65
                and laps_remaining >= 1
                and recoverable_gain >= 2.0
                and not wrong_weather_target
            )

            decision_kind = "stay_push"
            chosen_target_comp = fresh_comp

            strat_rank_pre = state["pit_sim_data"].get("strategy_ranking") or {}
            opt_pre = strat_rank_pre.get("optimal") or {}
            delta_opt_pre = float(strat_rank_pre.get("optimal_beats_requested_by_sec", 0.0))

            if wrong_weather_target:
                reason_str = " + ".join(rej_reasons) if rej_reasons else f"{fresh_comp} mismatches {weather_cond}"
                if weather_penalty > 0.0 and opt_weather_comp:
                    decision_kind = "wrong_weather_skip"
                    chosen_target_comp = opt_weather_comp
                    action = (
                        f"DO NOT BOX FOR {fresh_comp}s — SKIP {fresh_comp}s & BOX STRAIGHT FOR {opt_weather_comp}s "
                        f"({reason_str} | {alt_rec})"
                    )
                else:
                    decision_kind = "wrong_weather_stay"
                    action = (
                        f"STAY OUT & HOLD POSITION — DO NOT BOX FOR {fresh_comp}s "
                        f"({reason_str}; would drop P{ego_pos}->P{rejoin['median_pos']})"
                    )
            elif (
                drying_crossover_hint
                and ego_comp == "WET"
                and fresh_comp == "INTERMEDIATE"
                and float(rejoin.get("p_clean_air", 1.0)) < 0.25
            ):
                decision_kind = "drying_skip_inter"
                action = (
                    f"STAY OUT ON WETs & SKIP INTERMEDIATES — Track is drying and pitting now drops you "
                    f"P{ego_pos}->P{rejoin['median_pos']} into {rejoin['p_clean_air']*100:.1f}% clean air + forces a "
                    f"double-stop (~{loss*2:.0f}s total). Hold 2-3 laps and go straight to Dry Slicks!"
                )
            elif free_sc_stop:
                decision_kind = "free_sc_stop"
                action = (
                    f"BOX BOX NOW (FREE SC STOP — Rejoin P{rejoin['median_pos']} with zero positions lost; "
                    f"Save {saved:.1f}s under SC + +{recoverable_gain:.1f}s fresh-{fresh_comp} sprint gain!)"
                )
            elif net_payback <= 0.0:
                if rej_reasons and laps_remaining > 10:
                    decision_kind = "reject_trap_extend"
                    reason_str = " + ".join(rej_reasons)
                    alt_suffix = f" | Counter-strategy: {alt_rec}" if alt_rec else ""
                    action = (
                        f"STAY OUT & EXTEND STINT — DO NOT BOX FOR {fresh_comp}s "
                        f"({reason_str}; would drop P{ego_pos}->P{rejoin['median_pos']}, net {net_payback:+.1f}s){alt_suffix}"
                    )
                else:
                    decision_kind = "late_hold"
                    action = (
                        f"STAY OUT & HOLD POSITION (Only {laps_remaining}L left: {loss:.1f}s pit loss exceeds "
                        f"+{recoverable_gain:.1f}s fresh-tyre gain, would drop P{ego_pos}->P{rejoin['median_pos']})"
                    )
            elif weather_penalty >= 4.0 and TyreL1TensorCache.get_weather_compound_penalty(fresh_comp, weather_cond) == 0.0:
                decision_kind = "weather_crossover_box"
                action = (
                    f"BOX THIS LAP FOR {fresh_comp} (WEATHER CROSSOVER: +{weather_penalty:.1f}s/L wet grip loss "
                    f"on {ego_comp}s, +{net_payback:.1f}s net stint gain)"
                )
            elif status in ("SC", "VSC"):
                decision_kind = "sc_cheap_box"
                action = f"BOX BOX NOW (Save {saved:.1f}s under {status}!, +{net_payback:.1f}s net stint gain)"
            elif ego_pos > 1 and uc["win_probability"] > 0.55 and ego_age >= 8:
                decision_kind = "undercut_box"
                action = f"BOX FOR UNDERCUT ({uc['win_probability']*100:.0f}% Undercut Win Prob, +{net_payback:.1f}s net stint gain)"
            elif (
                ego_deg >= 2.15
                and not (
                    opt_pre.get("action_type") == "EXTEND_AND_BOX"
                    and delta_opt_pre >= 2.0
                    and ego_deg < 2.85
                )
            ):
                decision_kind = "wear_cliff_box"
                action = f"BOX THIS LAP (Tyre wear crossover reached: +{recoverable_gain:.1f}s gain over {laps_remaining}L > {loss:.1f}s pit loss)"
            else:
                # Align primary call with the Strategy Optimizer so the Pit Call badge & Optimal chip never contradict
                if (
                    opt_pre.get("action_type") == "BOX_NOW"
                    and net_payback >= 1.5
                ):
                    decision_kind = "opt_box_now"
                    chosen_target_comp = str(opt_pre.get("target_compound", fresh_comp))
                    action = (
                        f"BOX THIS LAP FOR {chosen_target_comp} "
                        f"(Strategy Optimizer optimum: +{net_payback:.1f}s net stint gain over {laps_remaining}L)"
                    )
                elif opt_pre.get("action_type") == "EXTEND_AND_BOX":
                    decision_kind = "opt_extend"
                    delta_note = f" — {delta_opt_pre:.1f}s faster than boxing now" if delta_opt_pre >= 0.5 else ""
                    action = f"STAY OUT & EXTEND STINT ({opt_pre.get('label', 'Extend stint')}{delta_note})"
                else:
                    decision_kind = "stay_push"
                    alt_note = f" | Better option if boxing later: {alt_rec}" if alt_rec else ""
                    action = f"STAY OUT & PUSH{alt_note}"
            parts.append(
                f"• PIT WALL SIM (10,000 runs): Pit loss is {loss:.2f}s (P95: {rejoin['p95_pit_loss']:.2f}s). "
                f"You rejoin P{rejoin['median_pos']} with {rejoin['p_clean_air']*100:.1f}% Clean Air "
                f"(Undercut Win Prob: {uc['win_probability']*100:.1f}% | Remaining {laps_remaining}L Fresh-{fresh_comp} Gain: "
                f"+{recoverable_gain:.2f}s, Net Payback: {net_payback:+.2f}s). Call: **{action}**."
            )

            # Multi-Option Strategy Optimizer comparison (Optimal vs Requested + Double-Stack & FIA 2-Compound check)
            strat_rank = state["pit_sim_data"].get("strategy_ranking")
            ds_delay = float(rejoin.get("double_stack_delay_sec", 0.0))
            opt_label = ""
            if strat_rank and strat_rank.get("optimal"):
                opt = strat_rank["optimal"]
                opt_label = str(opt.get("label", ""))
                req_opt = strat_rank.get("requested") or opt
                delta_opt = float(strat_rank.get("optimal_beats_requested_by_sec", 0.0))
                opt_notes = [f"Optimal: **{opt['label']}**"]
                if req_opt["strategy_id"] != opt["strategy_id"] and delta_opt >= 0.5:
                    opt_notes.append(f"beats `{req_opt['label']}` by **{delta_opt:.1f}s**")
                if ds_delay >= 1.5:
                    opt_notes.append(f"⚠️ Teammate double-stack queuing penalty +{ds_delay:.1f}s")
                if strat_rank.get("fia_two_compound_enforced") and len(strat_rank.get("used_dry_compounds", [])) < 2:
                    opt_notes.append(
                        f"⚖️ FIA 2-Compound Rule active (only used {', '.join(strat_rank.get('used_dry_compounds', []))}; mandatory 2nd dry compound required)"
                    )
                parts.append("• STRATEGY OPTIMIZER: " + " | ".join(opt_notes) + ".")

            pit_human_ctx = {
                "decision_kind": decision_kind,
                "ego_pos": ego_pos,
                "ego_comp": ego_comp,
                "ego_age": ego_age,
                "ego_deg": ego_deg,
                "fresh_comp": fresh_comp,
                "chosen_target_comp": chosen_target_comp,
                "opt_weather_comp": opt_weather_comp,
                "rejoin_pos": int(rejoin["median_pos"]),
                "clean_air_pct": float(rejoin["p_clean_air"]) * 100.0,
                "loss": loss,
                "saved": saved,
                "uc_prob_pct": float(uc["win_probability"]) * 100.0,
                "recoverable_gain": recoverable_gain,
                "net_payback": net_payback,
                "laps_remaining": laps_remaining,
                "weather_cond": str(weather_cond or "Dry (Clear)"),
                "weather_penalty": float(weather_penalty),
                "rej_reasons": list(rej_reasons),
                "alt_rec": alt_rec,
                "opt_label": opt_label,
                "opt_delta_sec": delta_opt_pre,
                "opt_pit_offset": int(opt_pre.get("pit_lap_offset", 0)),
                "opt_target_comp": str(opt_pre.get("target_compound", "")),
                "ds_delay": ds_delay,
                "status": str(status),
            }

        # 4. Multi-Hop FIA Legal Advisory (with graceful offline fallback)
        if state.get("legal_data"):
            ld = state["legal_data"]
            h1_list = ld.get("hop1_primary_rules", [])
            if ld.get("offline") or not h1_list:
                parts.append(
                    "• FIA LEGAL: ⚠️ FIA Rulebook vector database is currently offline (Qdrant unreachable) — "
                    "telemetry & strategy recommendations above remain fully active."
                )
            else:
                h1 = h1_list[0]
                h2_list = ld.get("hop2_resolved_references", [])
                h1_clean = re.sub(r"\s+", " ", h1["text"]).strip()[:160]
                legal_line = f"• FIA LEGAL (Page {h1['page']}, {h1['score']}% match): \"{h1_clean}...\""
                if h2_list:
                    h2 = h2_list[0]
                    h2_clean = re.sub(r"\s+", " ", h2["text"]).strip()
                    idx = h2_clean.find(h2["article"])
                    excerpt = h2_clean[idx : idx + 180] if idx != -1 else h2_clean[:180]
                    legal_line += (
                        f"\n  ↳ 🔗 [Hop-2 Cross-Reference Article {h2['article']} (Page {h2['page']})]: \"{excerpt}...\""
                    )
                parts.append(legal_line)

        human_call = self._build_human_radio_call(
            driver_code=state["driver"],
            radio_message=state.get("radio_message", ""),
            target_rival=state.get("target_rival"),
            pit_ctx=pit_human_ctx,
            battle_data=state.get("battle_data") if state.get("battle_data") and "error" not in state["battle_data"] else None,
            tyre_data=state.get("tyre_data") or None,
            legal_data=state.get("legal_data") or None,
        )

        return {
            "engineer_radio_reply": "\n".join(parts),
            "human_radio_call": human_call,
        }

    @staticmethod
    def _format_battle_radio_sentence(first: str, battle_data: dict) -> str:
        """Builds a natural, question-direct spoken radio sentence for an on-track rival battle."""
        rival_first = _driver_first_name(battle_data["target_driver"])
        rival_pos = int(battle_data["target_position"])
        ego_pos = int(battle_data["ego_position"])
        gap_b = float(battle_data["gap_between_sec"])
        pace_adv = float(battle_data.get("effective_pace_advantage_sec_per_lap", 0.0))
        laps_to_int = battle_data.get("estimated_laps_to_intersect")
        laps_rem = int(battle_data.get("laps_remaining", 0))
        rival_ahead = bool(battle_data.get("rival_is_ahead", rival_pos < ego_pos))
        verdict_up = str(battle_data.get("tactical_verdict", "")).upper()

        if "RIVAL PIT LAP DETECTED" in verdict_up:
            pos_rel = "ahead" if rival_ahead else "behind"
            return (
                f"Ignore that pace spike, {first} — {rival_first} just had a pit lap and is "
                f"{gap_b:.1f}s {pos_rel} in P{rival_pos}, so hold your rhythm."
            )

        if not rival_ahead:
            # Rival is BEHIND ego (e.g. "can PER catch me?")
            if pace_adv >= 0.0:
                return (
                    f"No {first}, {rival_first} can't catch you — he's {gap_b:.1f}s behind in P{rival_pos} "
                    f"and you're pulling away by {abs(pace_adv):.2f}s a lap."
                )
            closing_rate = abs(pace_adv)
            if laps_to_int is not None and float(laps_to_int) <= max(1, laps_rem):
                return (
                    f"Watch out {first}, {rival_first} is {gap_b:.1f}s behind in P{rival_pos} and closing at "
                    f"{closing_rate:.2f}s a lap — at this pace he'd reach you in about {round(float(laps_to_int))} laps."
                )
            return (
                f"You're safe for now, {first} — {rival_first} is {gap_b:.1f}s behind in P{rival_pos}, "
                f"only closing at {closing_rate:.2f}s a lap, so he won't catch you before the flag."
            )

        # Rival is AHEAD of ego (e.g. "can I catch HAM?")
        if pace_adv > 0.0 and laps_to_int is not None and float(laps_to_int) <= max(1, laps_rem):
            return (
                f"Yes {first}, you can catch {rival_first} — he's {gap_b:.1f}s ahead in P{rival_pos} "
                f"and you're reeling him in at {pace_adv:.2f}s a lap, roughly {round(float(laps_to_int))} laps to reach him."
            )
        if pace_adv > 0.0:
            return (
                f"Tough ask on track, {first} — {rival_first} is {gap_b:.1f}s ahead in P{rival_pos}; "
                f"you're gaining {pace_adv:.2f}s a lap, which isn't quite enough laps to catch him before the flag."
            )
        return (
            f"Negative {first}, {rival_first} is {gap_b:.1f}s ahead in P{rival_pos} "
            f"and pulling away by {abs(pace_adv):.2f}s a lap right now."
        )

    @staticmethod
    def _build_human_radio_call(
        *,
        driver_code: str,
        radio_message: str = "",
        target_rival: str | None = None,
        pit_ctx: dict | None,
        battle_data: dict | None,
        tyre_data: dict | None,
        legal_data: dict | None,
    ) -> str:
        """
        Deterministically translates the verified strategy, telemetry, battle, and FIA tool outputs
        into a crisp, natural 2-3 sentence F1 Pit Wall team radio call (zero LLM hallucination risk)
        that directly answers what the driver actually asked.
        """
        first = _driver_first_name(driver_code)
        sentences: list[str] = []

        asked_pit = bool(_PIT_INTENT_PATTERN.search(radio_message or ""))
        asked_battle = bool(
            battle_data is not None
            and (target_rival is not None or _BATTLE_INTENT_PATTERN.search(radio_message or ""))
        )

        # 1. Rival-Battle Primary Path (e.g. "can PER catch me?", "can I pass LEC?")
        # Always directly answer the rival question FIRST so we never repeat a generic pit call!
        if asked_battle and battle_data is not None:
            sentences.append(AutonomousRaceEngineer._format_battle_radio_sentence(first, battle_data))
            if pit_ctx is not None:
                kind = pit_ctx["decision_kind"]
                target_pl = _compound_plural(pit_ctx["chosen_target_comp"])
                fresh_pl = _compound_plural(pit_ctx["fresh_comp"])
                opt_lbl = pit_ctx.get("opt_label", "")
                if not asked_pit:
                    # Pure battle question: add only a brief 1-sentence tactical context note
                    if kind == "opt_extend" and opt_lbl:
                        sentences.append(f"Strategy-wise, stay out for now — best plan to cover them is {opt_lbl}.")
                    elif kind in ("free_sc_stop", "sc_cheap_box", "weather_crossover_box"):
                        sentences.append(f"Box this lap for {fresh_pl} to take advantage of the window.")
                    elif kind == "undercut_box":
                        sentences.append(
                            f"If you want to jump them in the pits, the undercut onto {fresh_pl} is open ({pit_ctx['uc_prob_pct']:.0f}% win prob)."
                        )
                    elif kind in ("wear_cliff_box", "opt_box_now"):
                        sentences.append(
                            f"On tyres, your {pit_ctx['ego_age']}-lap {_compound_plural(pit_ctx['ego_comp'])} are at the crossover — we can box for fresh {target_pl} whenever you're ready."
                        )
                    elif kind == "late_hold":
                        sentences.append(
                            f"With only {pit_ctx['laps_remaining']} laps left, stay out and fight on track — pitting would drop us to P{pit_ctx['rejoin_pos']}."
                        )
                    else:
                        sentences.append(
                            f"Stay out and manage your {pit_ctx['ego_age']}-lap {_compound_plural(pit_ctx['ego_comp'])} (+{pit_ctx['ego_deg']:.2f}s/lap wear)."
                        )
                else:
                    # Combined battle + pit question (e.g. "Can I catch the car ahead or should we box to undercut?")
                    if kind == "undercut_box":
                        sentences.append(
                            f"Box this lap for {fresh_pl} — undercut win probability is {pit_ctx['uc_prob_pct']:.0f}% (+{pit_ctx['net_payback']:.1f}s net)."
                        )
                    elif kind in ("free_sc_stop", "sc_cheap_box", "weather_crossover_box", "wear_cliff_box", "opt_box_now"):
                        sentences.append(
                            f"Box this lap for fresh {target_pl} — gains +{pit_ctx['net_payback']:.1f}s net and rejoins P{pit_ctx['rejoin_pos']}."
                        )
                    elif kind == "opt_extend" and opt_lbl:
                        sentences.append(f"Hold off on pitting — optimal strategy is {opt_lbl}.")
                    elif kind == "late_hold":
                        sentences.append(
                            f"Don't box — with {pit_ctx['laps_remaining']} laps left, pitting drops us to P{pit_ctx['rejoin_pos']}."
                        )
                    else:
                        sentences.append(
                            f"Stay out for now — pitting drops us to P{pit_ctx['rejoin_pos']} ({pit_ctx['net_payback']:+.1f}s net)."
                        )

        elif pit_ctx is not None:
            kind = pit_ctx["decision_kind"]
            ego_pos = pit_ctx["ego_pos"]
            rejoin_pos = pit_ctx["rejoin_pos"]
            fresh_pl = _compound_plural(pit_ctx["fresh_comp"])
            target_pl = _compound_plural(pit_ctx["chosen_target_comp"])
            ego_pl = _compound_plural(pit_ctx["ego_comp"])
            net_pb = pit_ctx["net_payback"]
            rec_gain = pit_ctx["recoverable_gain"]
            loss = pit_ctx["loss"]
            saved = pit_ctx["saved"]
            laps_rem = pit_ctx["laps_remaining"]
            clean_air = pit_ctx["clean_air_pct"]
            uc_pct = pit_ctx["uc_prob_pct"]
            status = pit_ctx["status"]
            alt_rec = pit_ctx["alt_rec"]
            rej_reasons = pit_ctx["rej_reasons"]
            short_reason = rej_reasons[0] if rej_reasons else "the numbers don't pay back"

            if kind == "wrong_weather_skip":
                opt_w_pl = _compound_plural(pit_ctx["opt_weather_comp"])
                sentences.append(
                    f"Negative {first}, skip the {fresh_pl} — {short_reason}."
                )
                sentences.append(
                    f"Box straight for {opt_w_pl} instead so we don't waste a stop."
                )
            elif kind == "wrong_weather_stay":
                sentences.append(
                    f"Stay out {first}, negative on {fresh_pl} — {short_reason}."
                )
                sentences.append(
                    f"Pitting now would drop us from P{ego_pos} to P{rejoin_pos}, so hold track position on the {ego_pl}."
                )
            elif kind == "drying_skip_inter":
                sentences.append(
                    f"Stay out on the Wets for now, {first} — skip the Intermediates."
                )
                sentences.append(
                    f"Pitting now drops us to P{rejoin_pos} in traffic ({clean_air:.0f}% clean air) and forces a double stop; hold 2–3 laps and we'll go straight to slicks."
                )
            elif kind == "free_sc_stop":
                sentences.append(
                    f"Box, box now {first}! Free Safety Car stop onto fresh {fresh_pl} — you rejoin P{rejoin_pos} without losing a place."
                )
                sentences.append(
                    f"We save {saved:.1f}s under the SC and gain +{rec_gain:.1f}s of fresh-tyre pace for the restart."
                )
            elif kind == "weather_crossover_box":
                sentences.append(
                    f"Box this lap for {fresh_pl}, {first} — we're losing {pit_ctx['weather_penalty']:.1f}s a lap on {ego_pl} in these conditions."
                )
                sentences.append(
                    f"Switching now pays back +{net_pb:.1f}s over the stint and you rejoin P{rejoin_pos}."
                )
            elif kind == "sc_cheap_box":
                sentences.append(
                    f"Box, box now {first} — take the cheap {status} stop onto fresh {fresh_pl}."
                )
                sentences.append(
                    f"We save {saved:.1f}s vs green-flag pit loss, rejoin P{rejoin_pos}, and gain +{net_pb:.1f}s net over the remaining {laps_rem} laps."
                )
            elif kind == "reject_trap_extend":
                sentences.append(
                    f"Stay out {first}, negative on the {fresh_pl} — {short_reason}."
                )
                counter_txt = f" {alt_rec}." if alt_rec else f" Keep extending on the {ego_pl} until traffic clears."
                sentences.append(
                    f"Boxing now drops us P{ego_pos} to P{rejoin_pos} ({net_pb:+.1f}s net).{counter_txt}"
                )
            elif kind == "late_hold":
                sentences.append(
                    f"Stay out and hold P{ego_pos}, {first} — only {laps_rem} laps to the flag."
                )
                sentences.append(
                    f"The {loss:.1f}s pit loss outweighs the +{rec_gain:.1f}s fresh-tyre gain and would drop us to P{rejoin_pos}."
                )
            elif kind == "undercut_box":
                sentences.append(
                    f"Box this lap for {fresh_pl}, {first} — the undercut is on ({uc_pct:.0f}% win probability, +{net_pb:.1f}s net gain)."
                )
                sentences.append(
                    f"You rejoin P{rejoin_pos} with {clean_air:.0f}% clean air — push on the in-lap."
                )
            elif kind in ("wear_cliff_box", "opt_box_now"):
                sentences.append(
                    f"Box this lap for fresh {target_pl}, {first} — we've hit the crossover window."
                )
                sentences.append(
                    f"Fresh rubber gains +{net_pb:.1f}s net over the remaining {laps_rem} laps and drops you out in P{rejoin_pos} ({clean_air:.0f}% clean air)."
                )
            elif kind == "opt_extend":
                ext_laps = int(pit_ctx.get("opt_pit_offset", 0))
                ext_comp = _compound_plural(pit_ctx.get("opt_target_comp") or pit_ctx["fresh_comp"])
                delta_s = float(pit_ctx.get("opt_delta_sec", 0.0))
                if ext_laps > 0 and delta_s >= 0.5:
                    sentences.append(
                        f"Wait {ext_laps} more laps before boxing, {first} — extending these {ego_pl} and switching to {ext_comp} is {delta_s:.1f}s faster than stopping now."
                    )
                else:
                    sentences.append(
                        f"Stay out and extend the {ego_pl}, {first} — best plan is {pit_ctx['opt_label'] or 'extend before boxing'}."
                    )
                sentences.append(
                    f"Your {pit_ctx['ego_age']}-lap {ego_pl} are losing +{pit_ctx['ego_deg']:.2f}s/lap, so manage them a few more laps for the stronger stint."
                )
            else:  # stay_push
                sentences.append(
                    f"Stay out and push, {first} — holding track position in P{ego_pos} is the stronger play right now."
                )
                if alt_rec:
                    sentences.append(f"Next window: {alt_rec}.")
                elif tyre_data:
                    sentences.append(
                        f"Your {tyre_data['current_tyre_life']}-lap {ego_pl} are losing +{tyre_data['current_deg_sec_per_lap']:.2f}s/lap."
                    )

            if pit_ctx["ds_delay"] >= 1.5:
                sentences.append(
                    f"Heads up: double-stacking behind your teammate would add ~{pit_ctx['ds_delay']:.1f}s."
                )

        elif battle_data is not None:
            sentences.append(AutonomousRaceEngineer._format_battle_radio_sentence(first, battle_data))

        elif tyre_data is not None:
            comp_pl = _compound_plural(tyre_data["compound"])
            sentences.append(
                f"{first}, your {tyre_data['current_tyre_life']}-lap {comp_pl} are losing +{tyre_data['current_deg_sec_per_lap']:.2f}s per lap right now."
            )
            sentences.append(
                f"We're projecting +{tyre_data['cumulative_5lap_loss_sec']:.1f}s of cumulative wear over the next 5 laps."
            )

        if legal_data:
            h1_list = legal_data.get("hop1_primary_rules", [])
            if legal_data.get("offline") or not h1_list:
                sentences.append("FIA rulebook search is offline right now, but telemetry & strategy calls are live.")
            else:
                h1 = h1_list[0]
                h2_list = legal_data.get("hop2_resolved_references", [])
                art_note = f", Article {h2_list[0]['article']}" if h2_list else ""
                msg_low = (radio_message or "").lower()
                is_pure_legal_reply = len(sentences) == 0

                if re.search(r"\b(?:overtake|overtaking|pass|passing)\b", msg_low) and re.search(
                    r"\b(?:sc|vsc|safety car|virtual safety car)\b", msg_low
                ):
                    sentences.append(
                        f"Negative {first}, no overtaking under the Safety Car — hold position unless Race Control clears lapped cars to unlap or a car ahead has an obvious mechanical problem."
                    )
                    sentences.append(
                        f"Penalties are enforced by the stewards under FIA Sporting Regulations (Page {h1['page']}{art_note}); open More Clarification for the exact wording."
                    )
                elif re.search(r"\b(?:overtake|overtaking|pass|passing)\b", msg_low) and "yellow" in msg_low:
                    sentences.append(
                        f"Negative {first}, no overtaking under yellow flags — lift and hold position through the marshalling sector to avoid a steward penalty."
                    )
                    sentences.append(f"Verified in the FIA Rulebook (Page {h1['page']}{art_note}).")
                elif re.search(r"\b(?:white line|pit exit|pit entry|blend line)\b", msg_low):
                    sentences.append(
                        f"Negative {first}, do not cross the solid white line at pit entry or pit exit — stay strictly inside the painted boundary or the stewards will penalize us."
                    )
                    sentences.append(f"Verified in the FIA Rulebook (Page {h1['page']}{art_note}).")
                elif "track limit" in msg_low:
                    sentences.append(
                        f"Watch the white lines, {first} — all four wheels over the track edge counts as a strike, and repeated breaches trigger a time penalty."
                    )
                    sentences.append(f"Verified in the FIA Rulebook (Page {h1['page']}{art_note}).")
                elif "unsafe release" in msg_low:
                    sentences.append(
                        f"Wait for the green light in the box, {first} — an unsafe release into oncoming pit-lane traffic carries a steward penalty."
                    )
                    sentences.append(f"Verified in the FIA Rulebook (Page {h1['page']}{art_note}).")
                elif is_pure_legal_reply:
                    excerpt = re.sub(r"\s+", " ", h1["text"]).strip()
                    short_rule = excerpt[:155].rsplit(" ", 1)[0] if len(excerpt) > 155 else excerpt
                    sentences.append(
                        f"Copy {first}, per the FIA Sporting Regulations (Page {h1['page']}{art_note}): \"{short_rule}...\""
                    )
                    sentences.append("Open More Clarification below for the full article text.")
                else:
                    sentences.append(
                        f"FIA rule check confirmed on Page {h1['page']}{art_note} — tap More Clarification for the exact wording."
                    )

        if not sentences:
            return f"Copy {first}, all systems look green on the pit wall — keep pushing and we'll watch the gaps."

        return " ".join(sentences)

    def ask_engineer(
        self,
        radio_message: str,
        driver: str | None = None,
        context: RaceContext | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        active_driver = (
            context.driver
            if context is not None
            else (driver if driver is not None else self.ego_driver)
        )
        initial_state: EngineerAgentState = {
            "driver": active_driver,
            "radio_message": radio_message,
            "context": context,
            "required_tools": [],
            "target_rival": None,
            "live_data": {},
            "tyre_data": {},
            "battle_data": {},
            "pit_sim_data": {},
            "legal_data": {},
            "guardrail_passed": False,
            "guardrail_notes": [],
            "engineer_radio_reply": "",
            "human_radio_call": "",
            "elapsed_ms": 0.0,
        }
        final_state = self.graph.invoke(initial_state)
        final_state["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return final_state


if __name__ == "__main__":
    engineer = AutonomousRaceEngineer(ego_driver="NOR")

    scenarios = [
        "can i catch HAM?",
        "GP, Safety Car is out and my Soft tyres are cooked! Should we box for Mediums right now, and what is the penalty if I overtake a slow car under the Safety Car?",
        "How much time am I losing per lap on these tyres right now?",
    ]

    for i, msg in enumerate(scenarios, 1):
        print("\n" + "=" * 75)
        print(f"🎙️  SCENARIO {i} — DRIVER RADIO: \"{msg}\"")
        print("=" * 75)
        res = engineer.ask_engineer(msg)
        print(res["engineer_radio_reply"])
        print(f"\n⏱️  Total LangGraph Reasoning + Tool Execution Latency: {res['elapsed_ms']} ms")
