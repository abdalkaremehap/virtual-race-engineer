import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
from loguru import logger


_CIRCUIT_PHYSICS_CACHE: dict[str, dict] | None = None


def _load_circuit_physics_table() -> dict[str, dict]:
    global _CIRCUIT_PHYSICS_CACHE
    if _CIRCUIT_PHYSICS_CACHE is None:
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/circuit_pit_data.json")
        )
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _CIRCUIT_PHYSICS_CACHE = {k.lower(): v for k, v in raw.items()}
        except Exception:
            _CIRCUIT_PHYSICS_CACHE = {}
    return _CIRCUIT_PHYSICS_CACHE


class TyreL1TensorCache:
    """
    Precomputes a (5 Compounds x 41 TrackTemps x 71 TyreLife) 3D Float32 Tensor (56.86 KB)
    from the trained LightGBM physics model for O(1) ~180ns hardware cache lookups.
    """

    COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

    # Compound-specific thermal cliff onset (laps) beyond LightGBM's dry-stint training horizon
    CLIFF_ONSET_LAPS = {
        "SOFT": 22.0,
        "MEDIUM": 32.0,
        "HARD": 42.0,
        "INTERMEDIATE": 26.0,
        "WET": 28.0,
    }
    CLIFF_COEFF = {
        "SOFT": 0.0080,
        "MEDIUM": 0.0055,
        "HARD": 0.0038,
        "INTERMEDIATE": 0.0065,
        "WET": 0.0070,
    }

    # Minimum asphalt temperature (°C) required for compound thermal activation (below -> cold graining / low grip)
    MIN_ACTIVATION_TEMP_C: dict[str, float] = {
        "HARD": 30.0,
        "MEDIUM": 23.0,
        "SOFT": 16.0,
        "INTERMEDIATE": 12.0,
        "WET": 10.0,
    }
    COLD_ACTIVATION_COEFF: dict[str, float] = {
        "HARD": 0.28,    # e.g. at 26.7°C (3.3°C below 30°C): +0.92s/L out-lap, ~11s lost over 31L stint
        "MEDIUM": 0.14,
        "SOFT": 0.05,
        "INTERMEDIATE": 0.0,
        "WET": 0.0,
    }

    # Weather-vs-compound grip mismatch penalty (+s/lap lost when running wrong rubber for track wetness)
    WEATHER_COMPOUND_PENALTY_SEC: dict[str, dict[str, float]] = {
        "DRY": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0, "INTERMEDIATE": 3.2, "WET": 6.5},
        "DAMP": {"SOFT": 4.8, "MEDIUM": 4.8, "HARD": 5.2, "INTERMEDIATE": 0.0, "WET": 2.1},
        "WET": {"SOFT": 9.5, "MEDIUM": 9.5, "HARD": 10.0, "INTERMEDIATE": 2.6, "WET": 0.0},
    }

    TEMP_MIN = 20
    TEMP_MAX = 60  # 41 discrete integer steps (20°C .. 60°C)
    MAX_LAP = 70   # 71 discrete steps (0 .. 70 laps)

    @classmethod
    def get_circuit_abrasion_factor(cls, circuit: str | None) -> float:
        """Returns asphalt abrasion multiplier relative to Bahrain (1.00 = highest abrasion benchmark)."""
        if not circuit:
            return 1.0
        entry = _load_circuit_physics_table().get(str(circuit).strip().lower())
        if entry and "abrasion_factor" in entry:
            return float(entry["abrasion_factor"])
        return 1.0

    @classmethod
    def get_circuit_overtaking_difficulty(cls, circuit: str | None) -> float:
        """Returns circuit overtaking difficulty scale (1.0 = easy DRS track .. 3.0 = Monaco)."""
        if not circuit:
            return 1.2
        entry = _load_circuit_physics_table().get(str(circuit).strip().lower())
        if entry and "overtaking_difficulty" in entry:
            return float(entry["overtaking_difficulty"])
        return 1.2

    @classmethod
    def get_cold_track_activation_penalty(
        cls,
        compound: str,
        track_temp: float,
        stint_lap: int = 1,
        weather_condition: str | None = None,
    ) -> float:
        """
        Models cold-asphalt compound activation / graining pace loss (+s/lap) when a dry compound
        (especially HARD below 30°C) cannot reach its Pirelli thermal operating window.
        """
        if cls.normalize_weather_regime(weather_condition) != "DRY":
            return 0.0
        comp = str(compound).upper()
        min_temp = cls.MIN_ACTIVATION_TEMP_C.get(comp, 20.0)
        coeff = cls.COLD_ACTIVATION_COEFF.get(comp, 0.0)
        deficit = max(0.0, min_temp - float(track_temp))
        if deficit <= 0.0 or coeff <= 0.0:
            return 0.0
        # Steep initial warm-up lack of bite over opening ~12 laps, decaying to persistent 25% cold-grip floor
        lap_num = max(1, int(stint_lap))
        warmup_decay = max(0.25, 1.0 - (lap_num - 1) / 12.0)
        return float(coeff * deficit * warmup_decay)

    @classmethod
    def normalize_weather_regime(cls, weather_condition: str | None) -> str:
        """Maps UI/telemetry weather strings to canonical regimes: 'DRY', 'DAMP', or 'WET'."""
        if not weather_condition:
            return "DRY"
        w = str(weather_condition).upper()
        if "HEAVY" in w or "RAIN" in w or "WET" in w:
            return "WET"
        if "DRIZZLE" in w or "DAMP" in w or "INTER" in w or "LIGHT" in w:
            return "DAMP"
        return "DRY"

    @classmethod
    def get_weather_compound_penalty(cls, compound: str, weather_condition: str | None) -> float:
        regime = cls.normalize_weather_regime(weather_condition)
        return float(cls.WEATHER_COMPOUND_PENALTY_SEC[regime].get(str(compound).upper(), 0.0))

    @classmethod
    def estimate_drying_crossover_penalty(
        cls,
        target_compound: str,
        pit_loss_sec: float,
        drying_laps_to_slicks: int = 3,
    ) -> float:
        """
        Estimates the wasted double-stop cost when fitting INTERMEDIATEs on a rapidly drying track
        that will require switching to dry slicks within `drying_laps_to_slicks` laps.
        """
        if str(target_compound).upper() != "INTERMEDIATE":
            return 0.0
        if int(drying_laps_to_slicks) <= 5:
            return round(float(pit_loss_sec) * 0.85, 2)
        return 0.0

    @classmethod
    def recommended_compound_for_weather(
        cls,
        weather_condition: str | None,
        laps_remaining: int | None = None,
        current_compound: str = "MEDIUM",
        track_temp: float | None = None,
    ) -> str:
        """Returns the physically optimal target compound for the active weather regime, track temp & stint length."""
        regime = cls.normalize_weather_regime(weather_condition)
        if regime == "WET":
            return "WET"
        if regime == "DAMP":
            return "INTERMEDIATE"
        cur = str(current_compound).upper()
        is_cold_for_hard = track_temp is not None and float(track_temp) < cls.MIN_ACTIVATION_TEMP_C["HARD"] - 1.0
        if laps_remaining is not None:
            if laps_remaining <= 15:
                return "SOFT"
            if laps_remaining <= 30:
                return "MEDIUM" if cur != "MEDIUM" else "SOFT"
            if is_cold_for_hard:
                return "MEDIUM" if cur != "MEDIUM" else "SOFT"
            return "HARD" if cur != "HARD" else "MEDIUM"
        if is_cold_for_hard:
            return "SOFT" if cur == "MEDIUM" else "MEDIUM"
        return "HARD" if cur == "MEDIUM" else "MEDIUM"

    def __init__(self, total_race_laps: int = 57):
        self.total_race_laps = max(1, total_race_laps)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.model_path = os.path.join(self.project_root, "models", "tyre_deg_model.txt")
        self.encoders_path = os.path.join(self.project_root, "models", "encoders.joblib")

        # Shape: (5, 41, 71) -> 14,555 float32 values = 58,220 bytes (~56.86 KB)
        self.tensor = np.zeros(
            (len(self.COMPOUNDS), self.TEMP_MAX - self.TEMP_MIN + 1, self.MAX_LAP + 1),
            dtype=np.float32,
        )
        # Factorized 1D race-lap fuel & track-evolution load multiplier (decouples race_lap from tyre_life)
        # Heavy fuel + green track early in race (~1.12x wear) -> light fuel + rubbered-in track late (~0.90x)
        race_laps = np.arange(self.MAX_LAP + 1, dtype=np.float32)
        fuel_frac = np.clip(1.0 - (race_laps / float(self.total_race_laps)), 0.0, 1.0)
        self.race_lap_load_factor = (0.90 + 0.22 * fuel_frac).astype(np.float32)
        self._build_tensor()

    def _build_tensor(self):
        t0 = time.perf_counter()
        if not os.path.exists(self.model_path) or not os.path.exists(self.encoders_path):
            raise FileNotFoundError(
                f"Missing trained model or encoders in {self.project_root}/models. "
                "Run src/ml_surrogates/tyre_model.py first!"
            )

        model = lgb.Booster(model_file=self.model_path)
        encoders = joblib.load(self.encoders_path)
        compound_le = encoders["compound"]
        known_compounds = set(compound_le.classes_)

        # Build full cartesian grid of (5 x 41 x 71 = 14,555 rows) in one vectorized batch
        c_idx, t_idx, l_idx = np.meshgrid(
            np.arange(len(self.COMPOUNDS)),
            np.arange(self.TEMP_MAX - self.TEMP_MIN + 1),
            np.arange(self.MAX_LAP + 1),
            indexing="ij",
        )

        flat_c = c_idx.ravel()
        flat_temp = (t_idx.ravel() + self.TEMP_MIN).astype(np.float64)
        flat_life = np.maximum(1, l_idx.ravel()).astype(np.float64)
        n_rows = flat_life.size

        # Map compound names safely to LabelEncoder integers
        encoded_compounds = np.zeros(n_rows, dtype=np.float64)
        for i, comp_name in enumerate(self.COMPOUNDS):
            if comp_name in known_compounds:
                code = float(compound_le.transform([comp_name])[0])
            else:
                # Fallback to HARD if wet/inter wasn't present in dry Bahrain training data
                fallback = "HARD" if "HARD" in known_compounds else compound_le.classes_[0]
                code = float(compound_le.transform([fallback])[0])
            encoded_compounds[flat_c == i] = code

        # Approximate ambient air_temp as track_temp - 6.5°C (Bahrain thermodynamic offset)
        air_temp = flat_temp - 6.5
        # Decouple base tyre-wear curve from race lap by evaluating at neutral mid-race fuel/evolution;
        # per-race-lap fuel & track evolution scaling is applied via `self.race_lap_load_factor` at lookup time.
        lap_num = np.full(n_rows, max(1.0, 0.5 * float(self.total_race_laps)), dtype=np.float64)

        # Construct the exact 23-feature matrix expected by LightGBM in zero-copy NumPy
        tyre_life_sq = flat_life ** 2
        tyre_life_log = np.log1p(flat_life)
        tyre_life_cube = flat_life ** 3
        tyre_life_sqrt = np.sqrt(flat_life)
        tyre_life_exp = np.exp(flat_life / 15.0)

        track_evo_sqrt = np.sqrt(lap_num)
        track_evo_log = np.log1p(lap_num)
        track_evo_lin = lap_num / float(self.total_race_laps)
        fuel_kg = np.maximum(0.0, 110.0 - (110.0 / float(self.total_race_laps)) * lap_num)

        temp_interaction = flat_temp * air_temp
        tyre_x_temp = flat_life * flat_temp
        tyre_x_fuel = flat_life * fuel_kg
        fuel_x_evo = fuel_kg * track_evo_lin
        tyre_x_evo = flat_life * track_evo_lin

        circuit_encoded = np.zeros(n_rows, dtype=np.float64)
        stint = np.full(n_rows, 2.0, dtype=np.float64)
        fresh_tyre = np.ones(n_rows, dtype=np.float64)

        X_grid = np.column_stack([
            flat_life, tyre_life_sq, tyre_life_log, tyre_life_cube, tyre_life_sqrt, tyre_life_exp,
            encoded_compounds,
            flat_temp, air_temp, flat_temp, air_temp,
            temp_interaction, tyre_x_temp, tyre_x_fuel, fuel_x_evo, tyre_x_evo,
            circuit_encoded,
            stint, fresh_tyre,
            track_evo_sqrt, track_evo_log, track_evo_lin,
            fuel_kg,
        ])

        preds = np.maximum(0.0, model.predict(X_grid)).astype(np.float32)
        tensor = preds.reshape(self.tensor.shape)

        # Apply physically distinct thermal wear profiles for INTERMEDIATE & WET (not duplicates of HARD)
        temps_axis = np.arange(self.TEMP_MIN, self.TEMP_MAX + 1, dtype=np.float32)[:, None]
        lives_axis = np.arange(self.MAX_LAP + 1, dtype=np.float32)[None, :]
        inter_idx = self.COMPOUND_TO_IDX["INTERMEDIATE"]
        wet_idx = self.COMPOUND_TO_IDX["WET"]
        temp_thermal_scale = 1.0 + 0.008 * np.maximum(0.0, temps_axis - 25.0)
        tensor[inter_idx] = tensor[inter_idx] * 1.12 * temp_thermal_scale + 0.012 * lives_axis
        tensor[wet_idx] = tensor[wet_idx] * 1.24 * temp_thermal_scale + 0.020 * lives_axis

        # Smooth monotonic thermal tyre-cliff extrapolation beyond dry-stint training horizon
        for comp_name, c_i in self.COMPOUND_TO_IDX.items():
            onset = float(self.CLIFF_ONSET_LAPS[comp_name])
            coeff = float(self.CLIFF_COEFF[comp_name])
            over_cliff = np.maximum(0.0, lives_axis - onset)
            cliff_delta = (coeff * (over_cliff ** 1.5)).astype(np.float32)
            tensor[c_i] = tensor[c_i] + cliff_delta

        # Enforce strict monotonicity along tyre_life axis across all compounds & temperatures
        self.tensor = np.maximum.accumulate(tensor, axis=2).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        size_kb = self.tensor.nbytes / 1024.0
        logger.success(
            f"🧊 L1 Tyre Tensor Cache Built: {self.tensor.shape} ({size_kb:.2f} KB) in {elapsed_ms:.1f}ms"
        )

    def lookup_degradation(
        self,
        compound: str,
        track_temp: float,
        tyre_life: int,
        race_lap: int | None = None,
        weather_condition: str | None = None,
        circuit: str | None = None,
    ) -> float:
        """
        O(1) bounded L1 cache lookup for a single car.
        - Safely clamps extreme out-of-distribution temperatures or tyre ages.
        - When `race_lap` is provided, applies factorized fuel-load & track-evolution scaling
          decoupled from `tyre_life`.
        - When `circuit` is provided, scales wear by the circuit's Pirelli asphalt abrasion factor
          (relative to Bahrain = 1.00).
        - When `weather_condition` is provided, adds wet/dry compound mismatch pace penalty.
        """
        comp_upper = str(compound).upper()
        c_idx = self.COMPOUND_TO_IDX.get(comp_upper, 1)  # Default MEDIUM if unknown
        t_idx = int(np.clip(round(track_temp) - self.TEMP_MIN, 0, self.TEMP_MAX - self.TEMP_MIN))
        l_idx = int(np.clip(tyre_life, 0, self.MAX_LAP))
        base_deg = float(self.tensor[c_idx, t_idx, l_idx])

        if circuit is not None:
            base_deg *= self.get_circuit_abrasion_factor(circuit)

        if race_lap is not None:
            r_idx = int(np.clip(int(race_lap), 0, self.MAX_LAP))
            base_deg *= float(self.race_lap_load_factor[r_idx])

        if weather_condition is not None:
            base_deg += self.get_weather_compound_penalty(comp_upper, weather_condition)

        return base_deg

    def lookup_grid_vectorized(
        self,
        compounds: list[str],
        track_temp: float,
        tyre_lives: np.ndarray,
        race_lap: int | None = None,
        weather_condition: str | None = None,
        circuit: str | None = None,
    ) -> np.ndarray:
        """
        Simultaneously looks up tyre degradation for all 20 cars on the grid in < 5 microseconds,
        with optional circuit abrasion scaling, race-lap fuel/evolution scaling, and weather mismatch penalties.
        """
        comp_uppers = [str(c).upper() for c in compounds]
        c_indices = np.array([self.COMPOUND_TO_IDX.get(c, 1) for c in comp_uppers], dtype=np.intp)
        t_idx = int(np.clip(round(track_temp) - self.TEMP_MIN, 0, self.TEMP_MAX - self.TEMP_MIN))
        l_indices = np.clip(tyre_lives, 0, self.MAX_LAP).astype(np.intp)
        degs = self.tensor[c_indices, t_idx, l_indices].astype(np.float32, copy=True)

        if circuit is not None:
            degs *= float(self.get_circuit_abrasion_factor(circuit))

        if race_lap is not None:
            r_idx = int(np.clip(int(race_lap), 0, self.MAX_LAP))
            degs *= float(self.race_lap_load_factor[r_idx])

        if weather_condition is not None:
            penalties = np.array(
                [self.get_weather_compound_penalty(c, weather_condition) for c in comp_uppers],
                dtype=np.float32,
            )
            degs += penalties

        return degs


if __name__ == "__main__":
    cache = TyreL1TensorCache(total_race_laps=57)

    # 1. Verify physics monotonicity & compound hierarchy at 32°C Track Temp
    print("\n" + "=" * 60 + "\nPHYSICS CHECK: Degradation Delta (s/lap) at 32.0°C\n" + "=" * 60)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        l5 = cache.lookup_degradation(comp, 32.0, 5)
        l15 = cache.lookup_degradation(comp, 32.0, 15)
        l25 = cache.lookup_degradation(comp, 32.0, 25)
        logger.info(f"🛞 {comp:<6} | Lap 5: +{l5:.3f}s | Lap 15: +{l15:.3f}s | Lap 25: +{l25:.3f}s")

    # 2. Benchmark single-car lookup speed (100,000 queries)
    n_queries = 100_000
    t0 = time.perf_counter()
    for _ in range(n_queries):
        _ = cache.lookup_degradation("SOFT", 32.5, 18)
    ns_per_lookup = ((time.perf_counter() - t0) / n_queries) * 1e9
    logger.success(f"⚡ Single-Car L1 Tensor Lookup Latency: {ns_per_lookup:.1f} nanoseconds/query")

    # 3. Benchmark full 20-car F1 grid batch lookup
    grid_compounds = ["SOFT", "SOFT", "MEDIUM", "HARD"] * 5
    grid_lives = np.array([14, 15, 8, 22] * 5)
    t1 = time.perf_counter()
    for _ in range(10_000):
        grid_deg = cache.lookup_grid_vectorized(grid_compounds, 32.5, grid_lives)
    us_per_grid = ((time.perf_counter() - t1) / 10_000) * 1e6
    logger.success(f"🚀 Full 20-Car Grid Vectorized Lookup: {us_per_grid:.2f} microseconds/tick")