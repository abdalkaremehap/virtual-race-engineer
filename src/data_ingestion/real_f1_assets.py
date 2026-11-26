import json
import math
import os
import time

import httpx
import numpy as np
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRACK_MAP_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "circuit_track_map_cache.json")
CHAMP_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "historical_championship_cache.json")

CIRCUIT_GEOJSON_ID_MAP: dict[str, str] = {
    "bahrain": "bh-2002",
    "saudi arabia": "sa-2021",
    "australia": "au-1953",
    "japan": "jp-1962",
    "china": "cn-2004",
    "miami": "us-2022",
    "emilia romagna": "it-1953",
    "monaco": "mc-1929",
    "canada": "ca-1978",
    "spain": "es-1991",
    "austria": "at-1969",
    "great britain": "gb-1948",
    "hungary": "hu-1986",
    "belgium": "be-1925",
    "netherlands": "nl-1948",
    "italy": "it-1922",
    "azerbaijan": "az-2016",
    "singapore": "sg-2008",
    "usa": "us-2012",
    "mexico": "mx-1962",
    "brazil": "br-1940",
    "las vegas": "us-2023",
    "qatar": "qa-2004",
    "abu dhabi": "ae-2009",
}


def _project_and_resample_lonlat(coords: list[list[float]], n_points: int = 240) -> tuple[list[float], list[float]]:
    """Projects WGS84 (lon, lat) GPS track points to local meters (X, Y) and resamples uniformly by arc length."""
    arr = np.asarray(coords, dtype=np.float64)
    lons = arr[:, 0]
    lats = arr[:, 1]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    cos_lat = math.cos(math.radians(lat0))

    x_m = (lons - lon0) * cos_lat * 111320.0
    y_m = (lats - lat0) * 111320.0

    # Ensure loop closure
    if math.hypot(x_m[-1] - x_m[0], y_m[-1] - y_m[0]) > 1.0:
        x_m = np.append(x_m, x_m[0])
        y_m = np.append(y_m, y_m[0])

    # Cumulative arc-length parameterization for uniform point spacing around corners & straights
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    seg_lens = np.hypot(dx, dy)
    cum_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_s = float(cum_s[-1]) if cum_s[-1] > 0 else 1.0

    # Remove duplicate zero-length vertices before interpolation
    uniq_mask = np.concatenate(([True], seg_lens > 1e-6))
    cum_s_u = cum_s[uniq_mask]
    x_u = x_m[uniq_mask]
    y_u = y_m[uniq_mask]

    target_s = np.linspace(0.0, total_s, n_points, endpoint=True)
    x_res = np.interp(target_s, cum_s_u, x_u)
    y_res = np.interp(target_s, cum_s_u, y_u)
    return np.round(x_res, 2).tolist(), np.round(y_res, 2).tolist()


def populate_all_24_real_circuit_track_maps(
    cache_path: str = TRACK_MAP_CACHE_PATH,
    n_points: int = 240,
) -> dict[str, dict[str, list[float]]]:
    """
    Downloads real-world FIA GPS circuit outlines for all 24 F1 circuits and stores
    uniformly resampled 240-point local-meter (X, Y) arrays in `data/circuit_track_map_cache.json`.
    """
    existing: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    url = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
    logger.info("🗺️ Fetching official GPS outlines for all 24 F1 circuits...")
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        geojson = resp.json()

    feat_by_id = {
        str(f.get("properties", {}).get("id", "")): f
        for f in geojson.get("features", [])
    }

    updated = dict(existing)
    added = 0
    for circuit_key, geo_id in CIRCUIT_GEOJSON_ID_MAP.items():
        # Preserve existing real FastF1 Bahrain GPS if present; populate all other circuits
        if circuit_key == "bahrain" and "bahrain" in existing and len(existing["bahrain"].get("x", [])) >= 200:
            continue
        feat = feat_by_id.get(geo_id)
        if not feat:
            logger.warning(f"⚠️ Missing GeoJSON feature for {circuit_key} ({geo_id})")
            continue
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "MultiLineString" and coords:
            coords = [pt for line in coords for pt in line]
        if len(coords) < 10:
            continue
        x_list, y_list = _project_and_resample_lonlat(coords, n_points=n_points)
        updated[circuit_key] = {"x": x_list, "y": y_list, "source": f"gps:{geo_id}"}
        added += 1

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(updated, f, separators=(",", ":"))
    logger.success(f"✅ Cached real GPS track maps for {len(updated)}/24 circuits ({added} newly added) -> {cache_path}")
    return updated


def populate_historical_championship_cache(
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024),
    cache_path: str = CHAMP_CACHE_PATH,
) -> dict:
    """
    Fetches authentic round-by-round FIA Grand Prix results (points, positions, wins, podiums)
    from Jolpica/Ergast for 2021–2024 and caches them locally in `data/historical_championship_cache.json`.
    """
    result_store: dict[str, dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                result_store = json.load(f)
        except Exception:
            result_store = {}

    with httpx.Client(timeout=20.0) as client:
        for season in seasons:
            s_key = str(int(season))
            logger.info(f"🏆 Fetching official {season} FIA race results from Jolpica...")
            rounds_map: dict[int, dict] = {}
            offset = 0
            limit = 100

            while True:
                url = f"https://api.jolpi.ca/ergast/f1/{season}/results.json"
                resp = None
                for attempt in range(4):
                    r = client.get(url, params={"limit": limit, "offset": offset})
                    if r.status_code == 429:
                        time.sleep(1.2 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    resp = r.json()
                    break
                if resp is None:
                    break

                mr = resp.get("MRData", {})
                total = int(mr.get("total", 0))
                races = mr.get("RaceTable", {}).get("Races", [])
                if not races:
                    break

                for race in races:
                    rnd = int(race["round"])
                    entry = rounds_map.setdefault(
                        rnd,
                        {
                            "round": rnd,
                            "race_name": race.get("raceName", ""),
                            "circuit_id": race.get("Circuit", {}).get("circuitId", ""),
                            "date": race.get("date", ""),
                            "results": [],
                        },
                    )
                    seen_codes = {x["driver"] for x in entry["results"]}
                    for res in race.get("Results", []):
                        drv_obj = res.get("Driver", {})
                        code = str(drv_obj.get("code") or drv_obj.get("familyName", "UNK")[:3]).upper()
                        if code in seen_codes:
                            continue
                        seen_codes.add(code)
                        pos = int(res.get("position", 20))
                        pts = float(res.get("points", 0.0))
                        team = str(res.get("Constructor", {}).get("name", "F1 Team"))
                        full_name = f"{drv_obj.get('givenName', '')} {drv_obj.get('familyName', '')}".strip()
                        num_str = str(drv_obj.get("permanentNumber") or res.get("number") or "0")
                        num_val = int(num_str) if num_str.isdigit() else 0
                        entry["results"].append({
                            "driver": code,
                            "name": full_name,
                            "number": num_val,
                            "team": team,
                            "position": pos,
                            "points": pts,
                        })

                offset += limit
                if offset >= total:
                    break
                time.sleep(0.35)

            sorted_rounds = [rounds_map[k] for k in sorted(rounds_map.keys())]
            result_store[s_key] = {
                "season": int(season),
                "total_rounds": len(sorted_rounds),
                "rounds": sorted_rounds,
            }
            logger.success(f"✅ Cached {len(sorted_rounds)} real rounds for {season} FIA Championship!")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result_store, f, separators=(",", ":"))
    logger.success(f"💾 Saved authentic 2021–2024 FIA championship cache -> {cache_path}")
    return result_store


if __name__ == "__main__":
    populate_all_24_real_circuit_track_maps()
    populate_historical_championship_cache()
