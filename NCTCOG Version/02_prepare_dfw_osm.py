"""Create an OSM-only DFW short-link network for NCTCOG matching.

The script reads explicitly configured district files in chunks, removes
TxDOT attributes, and retains complete OSM links intersecting the NCTCOG
network bounding rectangle plus a small margin. Source files are never modified.
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import shapely

from paths import (
    DISTRICT_OSM_INPUTS,
    NCTCOG_INPUT,
    OSM_INPUT,
    PREPARED_INPUT_DIR,
)


NCTCOG_LINKS = NCTCOG_INPUT
OSM_INPUTS = DISTRICT_OSM_INPUTS
OUTPUT_DIR = PREPARED_INPUT_DIR
OUTPUT_CSV = OSM_INPUT
SUMMARY_JSON = OUTPUT_DIR / "DFW_OSM_short_links_only_summary.json"

CHUNK_SIZE = 100_000
MARGIN_DEGREES = 0.02

OSM_COLUMNS = [
    "link_key",
    "name",
    "osm_way_id",
    "from_node_id",
    "to_node_id",
    "geometry",
    "facility_type",
    "link_type",
    "free_speed",
    "lanes",
    "capacity",
    "allowed_uses",
    "heading",
    "directional",
    "length",
]

TXDOT_COLUMNS_REMOVED = [
    "Matched_GID",
    "RDBD_TYPE",
    "Functional Class",
    "TXDOT_DI_1",
    "TXDOT_DI_2",
]


def require_files(paths):
    """Stop with a clear message if an input is missing."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))


def get_nctcog_extent():
    """Calculate the NCTCOG link-coordinate extent in WGS84 degrees."""
    cols = ["FROM_LON", "FROM_LAT", "TO_LON", "TO_LAT"]
    links = pd.read_csv(NCTCOG_LINKS, usecols=cols)

    if links[cols].isna().any().any():
        raise ValueError("NCTCOG endpoint coordinates contain missing values.")

    west = min(links["FROM_LON"].min(), links["TO_LON"].min())
    east = max(links["FROM_LON"].max(), links["TO_LON"].max())
    south = min(links["FROM_LAT"].min(), links["TO_LAT"].min())
    north = max(links["FROM_LAT"].max(), links["TO_LAT"].max())

    raw_extent = [float(west), float(south), float(east), float(north)]
    buffered_extent = [
        float(west - MARGIN_DEGREES),
        float(south - MARGIN_DEGREES),
        float(east + MARGIN_DEGREES),
        float(north + MARGIN_DEGREES),
    ]
    return len(links), raw_extent, buffered_extent


def validate_osm_header(path):
    """Confirm that expected OSM and TxDOT columns are present."""
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    required = OSM_COLUMNS + TXDOT_COLUMNS_REMOVED
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    unexpected = [column for column in columns if column not in required]
    if unexpected:
        raise ValueError(f"{path.name} has unexpected columns requiring review: {unexpected}")


def prepare_output(overwrite=False):
    """Stream, spatially select, and write the OSM-only DFW network."""
    require_files([NCTCOG_LINKS, *OSM_INPUTS])
    for path in OSM_INPUTS:
        validate_osm_header(path)

    nctcog_rows, raw_extent, buffered_extent = get_nctcog_extent()
    west, south, east, north = buffered_extent
    study_rectangle = shapely.box(west, south, east, north)

    print("NCTCOG rows:", f"{nctcog_rows:,}")
    print("NCTCOG extent [west, south, east, north]:", raw_extent)
    print("Selection extent with margin:", buffered_extent)
    print("Columns retained:", OSM_COLUMNS)
    print("Columns removed:", TXDOT_COLUMNS_REMOVED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUTPUT_CSV.exists() or SUMMARY_JSON.exists()) and not overwrite:
        raise FileExistsError(
            "A finished output already exists. Move/delete it only if you intend "
            f"to rerun, or use --overwrite:\n{OUTPUT_CSV}\n{SUMMARY_JSON}"
        )

    input_counts = {}
    retained_counts = {}
    invalid_geometry_counts = {}
    total_read = 0
    total_retained = 0
    selected_keys = set()

    with NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="DFW_OSM_partial_",
        dir=OUTPUT_DIR, delete=False, newline="", encoding="utf-8"
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        first_write = True
        for source in OSM_INPUTS:
            source_read = 0
            source_retained = 0
            source_invalid = 0
            print(f"\nProcessing {source.name} ...")

            reader = pd.read_csv(
                source,
                usecols=OSM_COLUMNS,
                dtype="string",
                keep_default_na=False,
                encoding="utf-8-sig",
                chunksize=CHUNK_SIZE,
            )

            for chunk in reader:
                geometries = shapely.from_wkt(
                    chunk["geometry"].to_numpy(), on_invalid="ignore"
                )
                invalid = shapely.is_missing(geometries) | shapely.is_empty(geometries)
                invalid_count = int(invalid.sum())
                source_invalid += invalid_count
                if invalid_count:
                    examples = chunk.loc[invalid, "link_key"].head(10).tolist()
                    raise ValueError(
                        f"Invalid/empty geometry in {source.name}; example link_key(s): "
                        f"{examples}"
                    )

                keep = shapely.intersects(geometries, study_rectangle)
                selected = chunk.loc[keep, OSM_COLUMNS]

                if not selected.empty:
                    keys = selected["link_key"].astype("string")
                    if keys.isna().any() or keys.str.strip().eq("").any():
                        raise ValueError(f"Missing OSM link key in {source.name}.")
                    batch_keys = set(keys)
                    overlap = batch_keys.intersection(selected_keys)
                    if len(batch_keys) != len(keys) or overlap:
                        examples = keys[keys.duplicated()].head(5).tolist()
                        examples += sorted(overlap)[:5]
                        raise ValueError(
                            "Duplicate OSM link_key across selected district "
                            f"rows in {source.name}: {examples[:10]}"
                        )
                    selected_keys.update(batch_keys)
                    selected.to_csv(
                        temp_path,
                        mode="a",
                        header=first_write,
                        index=False,
                        encoding="utf-8",
                    )
                    first_write = False

                source_read += len(chunk)
                source_retained += len(selected)
                total_read += len(chunk)
                total_retained += len(selected)
                print(
                    f"  read {source_read:,} | retained {source_retained:,}",
                    flush=True,
                )

            input_counts[source.name] = source_read
            retained_counts[source.name] = source_retained
            invalid_geometry_counts[source.name] = source_invalid

        if total_retained == 0:
            raise ValueError("No OSM links intersect the NCTCOG selection extent.")
        if len(selected_keys) != total_retained:
            raise ValueError("Selected OSM keys are not unique.")

        temp_path.replace(OUTPUT_CSV)

        summary = {
            "nctcog_link_rows": nctcog_rows,
            "nctcog_extent_wgs84": raw_extent,
            "selection_margin_degrees": MARGIN_DEGREES,
            "selection_extent_wgs84": buffered_extent,
            "input_files": [str(path) for path in OSM_INPUTS],
            "input_rows": input_counts,
            "retained_rows": retained_counts,
            "total_input_rows": total_read,
            "total_retained_rows": total_retained,
            "invalid_geometry_rows": invalid_geometry_counts,
            "columns_retained": OSM_COLUMNS,
            "columns_removed": TXDOT_COLUMNS_REMOVED,
            "output_csv": str(OUTPUT_CSV),
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nPreparation complete")
        print(f"Total OSM rows read: {total_read:,}")
        print(f"OSM rows retained: {total_retained:,}")
        print(f"Output: {OUTPUT_CSV}")
        print(f"Summary: {SUMMARY_JSON}")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the prepared OSM input and its summary.",
    )
    args = parser.parse_args()
    prepare_output(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
