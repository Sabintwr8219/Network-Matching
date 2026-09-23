import argparse
import json
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from paths import (
    FINAL_NETWORK_FILE,
    FINAL_VALIDATION_FILE,
    NCTCOG_COVERAGE_FILE,
    NCTCOG_PREPARED_FILE,
    OSM_ASSIGNMENT_FILE,
    OSM_INPUT,
)


BATCH_SIZE = 250_000

REQUIRED_NCTCOG_COLUMNS = [
    "nctcog_direction_key",
    "nctcog_id",
    "nctcog_direction",
    "nctcog_directional_volume",
    "assignment_method",
]


# Read source OSM keys and reject missing or duplicate identifiers.
def read_source_keys():
    keys = set()
    rows = 0
    for chunk in pd.read_csv(
        OSM_INPUT,
        usecols=["link_key"],
        dtype={"link_key": "string"},
        chunksize=BATCH_SIZE,
    ):
        values = chunk["link_key"]
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError("OSM source contains a missing link key.")
        batch_keys = set(values.tolist())
        if len(batch_keys) != len(values) or keys.intersection(batch_keys):
            raise ValueError("OSM source contains duplicate link keys.")
        keys.update(batch_keys)
        rows += len(values)
    return rows, keys


# Read the authoritative NCTCOG values used to verify final attributes.
def read_nctcog_reference():
    reference = pd.read_csv(
        NCTCOG_PREPARED_FILE,
        usecols=[
            "ID",
            "nctcog_direction_key",
            "nctcog_direction",
            "directional_volume",
        ],
        dtype={
            "ID": "string",
            "nctcog_direction_key": "string",
            "nctcog_direction": "string",
        },
    ).rename(columns={
        "ID": "nctcog_id",
        "directional_volume": "nctcog_directional_volume",
    })
    if (
        reference["nctcog_direction_key"].isna().any()
        or reference["nctcog_direction_key"].duplicated().any()
    ):
        raise ValueError("Prepared NCTCOG keys are invalid.")
    return reference.set_index("nctcog_direction_key")


# Validate row identity, assignment logic, and copied NCTCOG attributes.
def validate_final_network():
    required = [
        OSM_INPUT,
        OSM_ASSIGNMENT_FILE,
        NCTCOG_PREPARED_FILE,
        FINAL_NETWORK_FILE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])
    last_report = started

    source_columns = pd.read_csv(OSM_INPUT, nrows=0).columns.tolist()
    parquet = pq.ParquetFile(FINAL_NETWORK_FILE)
    final_columns = parquet.schema_arrow.names
    missing_columns = [
        column
        for column in source_columns + REQUIRED_NCTCOG_COLUMNS
        if column not in final_columns
    ]
    if missing_columns:
        raise ValueError(f"Final output is missing columns: {missing_columns}")

    reference = read_nctcog_reference()
    final_keys = set()
    method_counts = {}
    facility_counts = {}
    represented_nctcog = set()
    final_rows = assigned_rows = 0

    columns = [
        "link_key",
        "facility_type",
        "nctcog_direction_key",
        "nctcog_id",
        "nctcog_direction",
        "nctcog_directional_volume",
        "assignment_method",
    ]

    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        frame = batch.to_pandas()
        frame["link_key"] = frame["link_key"].astype("string")
        frame["nctcog_direction_key"] = frame[
            "nctcog_direction_key"
        ].astype("string")

        if frame["link_key"].isna().any():
            raise ValueError("Final output contains a missing OSM link key.")
        batch_keys = set(frame["link_key"].tolist())
        if len(batch_keys) != len(frame) or final_keys.intersection(batch_keys):
            raise ValueError("Final output contains duplicate OSM link keys.")
        final_keys.update(batch_keys)

        if frame["assignment_method"].isna().any():
            raise ValueError("Final output contains a missing assignment method.")
        assigned = frame["nctcog_direction_key"].notna()
        method_assigned = frame["assignment_method"].ne("unmatched")
        if not assigned.eq(method_assigned).all():
            raise ValueError("Final assignment key and method disagree.")

        selected = frame.loc[assigned].copy()
        unknown = ~selected["nctcog_direction_key"].isin(reference.index)
        if unknown.any():
            examples = selected.loc[unknown, "nctcog_direction_key"].head().tolist()
            raise ValueError(f"Unknown NCTCOG keys in final output: {examples}")

        expected = reference.loc[
            selected["nctcog_direction_key"].to_numpy()
        ].reset_index(drop=True)
        actual_id = selected["nctcog_id"].astype("string").reset_index(drop=True)
        actual_direction = selected["nctcog_direction"].astype("string").reset_index(drop=True)
        if not actual_id.eq(expected["nctcog_id"].reset_index(drop=True)).all():
            raise ValueError("Final NCTCOG IDs differ from their source records.")
        if not actual_direction.eq(
            expected["nctcog_direction"].reset_index(drop=True)
        ).all():
            raise ValueError("Final NCTCOG directions differ from their source records.")
        if not np.allclose(
            pd.to_numeric(selected["nctcog_directional_volume"]).to_numpy(),
            expected["nctcog_directional_volume"].to_numpy(),
            rtol=0,
            atol=0.01,
            equal_nan=False,
        ):
            raise ValueError("Final directional volumes differ from NCTCOG.")

        final_rows += len(frame)
        assigned_rows += int(assigned.sum())
        represented_nctcog.update(selected["nctcog_direction_key"].tolist())
        for method, count in frame["assignment_method"].value_counts().items():
            method_counts[str(method)] = method_counts.get(str(method), 0) + int(count)
        for facility, count in frame.loc[assigned, "facility_type"].value_counts().items():
            facility_counts[str(facility)] = facility_counts.get(str(facility), 0) + int(count)

        now = time.perf_counter()
        if now - last_report >= 5:
            elapsed = now - started
            cpu_used = sum(process.cpu_times()[:2]) - cpu_start
            print(
                f"Validated {final_rows:,} links | {elapsed:.1f}s | "
                f"CPU: {100 * cpu_used / elapsed / psutil.cpu_count():.1f}% | "
                f"RAM: {process.memory_info().rss / 1024**3:.2f} GB",
                flush=True,
            )
            last_report = now

    source_rows, source_keys = read_source_keys()
    if final_rows != source_rows:
        raise ValueError(
            f"Final rows {final_rows:,} differ from OSM source rows {source_rows:,}."
        )
    if final_keys != source_keys:
        raise ValueError("Final OSM link keys do not exactly match the source network.")

    assignments = pd.read_parquet(
        OSM_ASSIGNMENT_FILE,
        columns=["link_key", "assigned_nctcog_key", "assignment_method"],
    )
    if len(assignments) != final_rows:
        raise ValueError("Final row count differs from the assignment table.")
    expected_methods = {
        str(key): int(value)
        for key, value in assignments["assignment_method"].value_counts().items()
    }
    if method_counts != expected_methods:
        raise ValueError("Final assignment-method counts differ from Stage 8.")
    expected_assigned = int(assignments["assigned_nctcog_key"].notna().sum())
    if assigned_rows != expected_assigned:
        raise ValueError("Final assigned-link count differs from Stage 8.")

    nctcog_count = len(reference)
    report = {
        "status": "passed",
        "final_file": str(FINAL_NETWORK_FILE),
        "osm_source_rows": source_rows,
        "final_rows": final_rows,
        "unique_link_keys": len(final_keys),
        "assigned_osm_links": assigned_rows,
        "unmatched_osm_links": final_rows - assigned_rows,
        "assigned_osm_link_pct": round(100 * assigned_rows / final_rows, 4),
        "assignment_methods": method_counts,
        "assigned_links_by_facility": facility_counts,
        "nctcog_directional_records": nctcog_count,
        "represented_nctcog_directions": len(represented_nctcog),
        "represented_nctcog_direction_pct": round(
            100 * len(represented_nctcog) / nctcog_count, 4
        ),
        "checks": {
            "original_osm_columns_retained": True,
            "row_count_preserved": True,
            "link_keys_complete_and_unique": True,
            "assignment_methods_preserved": True,
            "assigned_keys_exist_in_nctcog": True,
            "directional_attributes_equal_source": True,
        },
    }

    if NCTCOG_COVERAGE_FILE.is_file():
        coverage = pd.read_parquet(
            NCTCOG_COVERAGE_FILE,
            columns=["length_m", "piece_support_m", "retained_direct_m"],
        )
        total_length = float(coverage["length_m"].sum())
        report["nctcog_length_coverage"] = {
            "piece_support_pct": round(
                100 * float(coverage["piece_support_m"].sum()) / total_length, 4
            ),
            "retained_direct_pct": round(
                100 * float(coverage["retained_direct_m"].sum()) / total_length, 4
            ),
        }

    FINAL_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = FINAL_VALIDATION_FILE.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(FINAL_VALIDATION_FILE)

    elapsed = time.perf_counter() - started
    print("\nFINAL VALIDATION PASSED")
    print(f"OSM rows and unique keys: {final_rows:,}")
    print(
        f"OSM links with NCTCOG attributes: {assigned_rows:,} "
        f"({100 * assigned_rows / final_rows:.2f}%)"
    )
    print(
        f"NCTCOG directions represented: {len(represented_nctcog):,}/"
        f"{nctcog_count:,} ({100 * len(represented_nctcog) / nctcog_count:.2f}%)"
    )
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Saved: {FINAL_VALIDATION_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate_final_network()


if __name__ == "__main__":
    main()
