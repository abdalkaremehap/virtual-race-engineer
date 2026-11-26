"""
Resumable CLI utility to pre-download authentic FastF1 FIA lap-by-lap telemetry for all
2021-2024 Grand Prix Races into `data/f1_historical.duckdb`.

Usage:
    python src/data_ingestion/prefetch_all_fastf1_races.py --seasons 2024 2023 2022 2021
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion.telemetry_daemon import (
    F1_2024_CALENDAR,
    SUPPORTED_SEASONS,
    fetch_real_fastf1_session,
    get_historical_race_order,
    get_session_data_source,
)


def main():
    parser = argparse.ArgumentParser(description="Bulk prefetch authentic FastF1 races into DuckDB.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(SUPPORTED_SEASONS),
        help="Seasons to prefetch (default: 2024 2023 2022 2021).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="Race",
        help="Session type to prefetch (default: Race).",
    )
    args = parser.parse_args()

    fetched = 0
    skipped_cached = 0
    skipped_calendar = 0
    failed = 0

    for season in args.seasons:
        print(f"\n=== Season {season} ({args.session}) ===", flush=True)
        for item in F1_2024_CALENDAR:
            circuit = item["circuit"]
            if not get_historical_race_order(season, circuit):
                skipped_calendar += 1
                print(f"  [SKIP-CALENDAR] {season} {circuit} (not held in {season})", flush=True)
                continue
            src = get_session_data_source(season, circuit, args.session)
            if src in ("FASTF1_REAL", "OPENF1_REAL"):
                skipped_cached += 1
                print(f"  [CACHED] {season} {circuit} {args.session}", flush=True)
                continue
            ok, msg = fetch_real_fastf1_session(circuit, args.session, season=season)
            if ok:
                fetched += 1
                print(f"  [OK] {msg}", flush=True)
            else:
                failed += 1
                print(f"  [WARN] {msg}", flush=True)

    print(
        f"\nDone: fetched={fetched}, already_cached={skipped_cached}, "
        f"not_on_calendar={skipped_calendar}, failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
