import argparse
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from paths import (
    FINAL_NETWORK_CSV,
    FINAL_NETWORK_FILE,
    NCTCOG_PREPARED_FILE,
    OSM_ASSIGNMENT_FILE,
    OSM_INPUT,
)


CHUNK_SIZE = 250_000
PARQUET_COMPRESSION = "zstd"

ASSIGNMENT_COLUMNS = [
    "link_key",
    "total_length_m",
    "matched_length_m",
    "ambiguous_length_m",
    "matched_fraction",
    "dominant_fraction",
    "dominant_share_of_matched",
    "assigned_nctcog_key",
    "assignment_method",
    "gap_repaired",
    "reverse_pair_repaired",
]

NCTCOG_COLUMNS = [
    "ID",
    "nctcog_direction_key",
    "nctcog_direction",
    "travel_from_node",
    "travel_to_node",
    "LNKNM",
    "FUNCL",
    "Dir",
    "Length",
    "MODEL_LENGTH",
    "DAYVOL_AB",
    "DAYVOL_BA",
    "DAYVOL",
    "directional_volume",
]

NCTCOG_RENAME = {
    "ID": "nctcog_id",
    "travel_from_node": "nctcog_from_node",
    "travel_to_node": "nctcog_to_node",
    "LNKNM": "nctcog_name",
    "FUNCL": "nctcog_funcl",
    "Dir": "nctcog_dir_code",
    "Length": "nctcog_length",
    "MODEL_LENGTH": "nctcog_model_length",
    "DAYVOL_AB": "nctcog_dayvol_ab",
    "DAYVOL_BA": "nctcog_dayvol_ba",
    "DAYVOL": "nctcog_dayvol_total",
    "directional_volume": "nctcog_directional_volume",
}

FINAL_ADDED_COLUMNS = [
    "nctcog_direction_key",
    "nctcog_id",
    "nctcog_direction",
    "nctcog_from_node",
    "nctcog_to_node",
    "nctcog_name",
    "nctcog_funcl",
    "nctcog_dir_code",
    "nctcog_length",
    "nctcog_model_length",
    "nctcog_dayvol_ab",
    "nctcog_dayvol_ba",
    "nctcog_dayvol_total",
    "nctcog_directional_volume",
    "assignment_method",
    "matched_fraction",
    "dominant_fraction",
    "dominant_share_of_matched",
    "matched_length_m",
    "ambiguous_length_m",
    "gap_repaired",
    "reverse_pair_repaired",
]

OSM_DTYPES = {
    "link_key": "string",
    "name": "string",
    "osm_way_id": "Int64",
    "from_node_id": "Int64",
    "to_node_id": "Int64",
    "geometry": "string",
    "facility_type": "string",
    "link_type": "Int64",
    "free_speed": "Float64",
    "lanes": "Float64",
    "capacity": "Float64",
    "allowed_uses": "string",
    "heading": "Float64",
    "directional": "string",
    "length": "Float64",
}


# Load the one-row-per-OSM-link assignment table produced by Stage 8.
def load_assignments():
    assignments = pd.read_parquet(
        OSM_ASSIGNMENT_FILE,
        columns=ASSIGNMENT_COLUMNS,
    )
    assignments["link_key"] = assignments["link_key"].astype("string")
    assignments["assigned_nctcog_key"] = assignments[
        "assigned_nctcog_key"
    ].astype("string")

    if assignments["link_key"].isna().any():
        raise ValueError("Assignment table contains a missing OSM link key.")
    if assignments["link_key"].duplicated().any():
        raise ValueError("Assignment table contains duplicate OSM link keys.")

    assigned = assignments["assigned_nctcog_key"].notna()
    method_assigned = assignments["assignment_method"].ne("unmatched")
    if not assigned.eq(method_assigned).all():
        raise ValueError("Assignment key and assignment method disagree.")

    assignments["_assignment_present"] = True
    return assignments


# Load the directional NCTCOG attributes used by the final network.
def load_nctcog_attributes():
    nctcog = pd.read_csv(
        NCTCOG_PREPARED_FILE,
        usecols=NCTCOG_COLUMNS,
        dtype={
            "ID": "string",
            "nctcog_direction_key": "string",
            "nctcog_direction": "string",
        },
    ).rename(columns=NCTCOG_RENAME)

    key = nctcog["nctcog_direction_key"]
    if key.isna().any() or key.duplicated().any():
        raise ValueError("Prepared NCTCOG directional keys must be complete and unique.")
    if not nctcog["nctcog_direction"].isin(["AB", "BA"]).all():
        raise ValueError("Prepared NCTCOG directions must be AB or BA.")

    expected_volume = np.where(
        nctcog["nctcog_direction"].eq("AB"),
        nctcog["nctcog_dayvol_ab"],
        nctcog["nctcog_dayvol_ba"],
    )
    if not np.allclose(
        nctcog["nctcog_directional_volume"],
        expected_volume,
        rtol=0,
        atol=0.01,
        equal_nan=False,
    ):
        raise ValueError("Prepared NCTCOG directional volumes are inconsistent.")

    return nctcog


# Normalize the source boolean text to a nullable Boolean column.
def normalize_directional(values):
    normalized = values.astype("string").str.strip().str.lower()
    result = normalized.map({"true": True, "false": False})
    unexpected = normalized.notna() & result.isna()
    if unexpected.any():
        examples = sorted(normalized.loc[unexpected].dropna().unique())[:5]
        raise ValueError(f"Unexpected OSM directional values: {examples}")
    return result.astype("boolean")


# Attach assignment provenance and directional NCTCOG attributes to one OSM batch.
def attach_attributes(osm, assignments, nctcog):
    osm["link_key"] = osm["link_key"].astype("string")
    if osm["link_key"].isna().any() or osm["link_key"].duplicated().any():
        raise ValueError("OSM batch contains missing or duplicate link keys.")
    if "directional" in osm:
        osm["directional"] = normalize_directional(osm["directional"])

    result = osm.merge(
        assignments,
        on="link_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if not result["_assignment_present"].fillna(False).all():
        missing = result.loc[
            result["_assignment_present"].isna(), "link_key"
        ].head().tolist()
        raise ValueError(f"OSM links missing from assignment table: {missing}")

    result = result.drop(columns="_assignment_present").rename(
        columns={"assigned_nctcog_key": "nctcog_direction_key"}
    )
    result = result.merge(
        nctcog,
        on="nctcog_direction_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    assigned = result["nctcog_direction_key"].notna()
    if result.loc[assigned, "nctcog_id"].isna().any():
        raise ValueError("An assigned OSM link references an unknown NCTCOG record.")
    if result.loc[~assigned, "nctcog_id"].notna().any():
        raise ValueError("An unmatched OSM link received NCTCOG attributes.")

    return result


# Write the complete OSM network with attached NCTCOG attributes.
def export_final_network(write_csv=False, overwrite=False):
    required = [OSM_INPUT, OSM_ASSIGNMENT_FILE, NCTCOG_PREPARED_FILE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    if FINAL_NETWORK_FILE.exists() and not overwrite:
        raise FileExistsError(
            f"Final output already exists: {FINAL_NETWORK_FILE}\n"
            "Use --overwrite only when replacing it intentionally."
        )
    if write_csv and FINAL_NETWORK_CSV.exists() and not overwrite:
        raise FileExistsError(
            f"Final CSV already exists: {FINAL_NETWORK_CSV}\n"
            "Use --overwrite only when replacing it intentionally."
        )

    FINAL_NETWORK_FILE.parent.mkdir(parents=True, exist_ok=True)
    parquet_partial = FINAL_NETWORK_FILE.with_suffix(".partial.parquet")
    csv_partial = FINAL_NETWORK_CSV.with_suffix(".partial.csv")
    parquet_partial.unlink(missing_ok=True)
    if write_csv:
        csv_partial.unlink(missing_ok=True)

    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])
    last_report = started

    assignments = load_assignments()
    nctcog = load_nctcog_attributes()
    original_columns = pd.read_csv(OSM_INPUT, nrows=0).columns.tolist()
    output_columns = original_columns + FINAL_ADDED_COLUMNS

    writer = None
    rows = assigned_rows = 0
    method_counts = {}
    csv_header = True

    try:
        reader = pd.read_csv(
            OSM_INPUT,
            dtype={
                column: dtype
                for column, dtype in OSM_DTYPES.items()
                if column in original_columns
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for osm in reader:
            result = attach_attributes(osm, assignments, nctcog)
            result = result[output_columns]

            table = pa.Table.from_pandas(result, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update({
                b"network": b"DFW OSM short-link network",
                b"nctcog_join": b"directional parent assignment",
                b"volume_semantics": (
                    b"nctcog_directional_volume is a parent attribute; "
                    b"do not sum it across OSM child links"
                ),
            })
            table = table.replace_schema_metadata(metadata)

            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_partial,
                    table.schema,
                    compression=PARQUET_COMPRESSION,
                )
            writer.write_table(table)

            if write_csv:
                result.to_csv(
                    csv_partial,
                    mode="w" if csv_header else "a",
                    header=csv_header,
                    index=False,
                )
                csv_header = False

            rows += len(result)
            assigned_rows += int(result["nctcog_direction_key"].notna().sum())
            for method, count in result["assignment_method"].value_counts().items():
                method_counts[str(method)] = method_counts.get(str(method), 0) + int(count)

            now = time.perf_counter()
            if now - last_report >= 5:
                elapsed = now - started
                cpu_used = sum(process.cpu_times()[:2]) - cpu_start
                print(
                    f"Exported {rows:,} OSM links | {elapsed:.1f}s | "
                    f"{rows / elapsed:,.0f} links/s | "
                    f"CPU: {100 * cpu_used / elapsed / psutil.cpu_count():.1f}% | "
                    f"RAM: {process.memory_info().rss / 1024**3:.2f} GB",
                    flush=True,
                )
                last_report = now
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("OSM input contained no rows.")
    if rows != len(assignments):
        raise ValueError(
            f"Final row count {rows:,} differs from assignment count "
            f"{len(assignments):,}. Partial output was retained for review."
        )
    if pq.read_metadata(parquet_partial).num_rows != rows:
        raise ValueError("Saved Parquet row count differs from processed rows.")

    parquet_partial.replace(FINAL_NETWORK_FILE)
    if write_csv:
        csv_partial.replace(FINAL_NETWORK_CSV)

    elapsed = time.perf_counter() - started
    print("\nFINAL NETWORK EXPORT COMPLETE")
    print(f"OSM links: {rows:,}")
    print(f"Links carrying NCTCOG attributes: {assigned_rows:,}")
    print(f"Unmatched OSM links retained: {rows - assigned_rows:,}")
    print("Assignment methods:")
    for method, count in sorted(method_counts.items(), key=lambda item: -item[1]):
        print(f"  {method}: {count:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Saved: {FINAL_NETWORK_FILE}")
    if write_csv:
        print(f"Saved: {FINAL_NETWORK_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write the much larger CSV version.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final output.",
    )
    args = parser.parse_args()
    export_final_network(write_csv=args.csv, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
