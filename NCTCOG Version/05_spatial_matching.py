import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import shapely

from paths import INTERMEDIATE_DIR
import pyarrow as pa

MAX_DISTANCE_M = 6.0
MAX_HEADING_DIFF_DEG = 15.0


METRIC_CRS = "EPSG:32614"
OSM_SEGMENTS = INTERMEDIATE_DIR / "OSM_segmented_10m.parquet"
NCTCOG_SEGMENTS = INTERMEDIATE_DIR / "NCTCOG_segmented_10m.parquet"

def load_nctcog_index():
    """Read prepared NCTCOG pieces and build their spatial index."""
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    metadata = pq.read_schema(NCTCOG_SEGMENTS).metadata or {}
    if metadata.get(b"crs") != METRIC_CRS.encode():
        raise ValueError("Unexpected NCTCOG coordinate system.")
    if metadata.get(b"geometry_encoding") != b"WKB":
        raise ValueError("Expected WKB geometry.")

    print("Reading NCTCOG Parquet...", flush=True)
    frame = pd.read_parquet(
        NCTCOG_SEGMENTS,
        columns=[
            "parent_key",
            "piece_index",
            "heading_grid_deg",
            "geometry_wkb",
        ],
    )
    read_finished = time.perf_counter()

    print("Decoding geometry...", flush=True)
    geometry = shapely.from_wkb(
        frame.pop("geometry_wkb").to_numpy()
    )
    decode_finished = time.perf_counter()

    if frame.empty:
        raise ValueError("No NCTCOG pieces to match.")

    headings = frame["heading_grid_deg"].to_numpy()
    if not np.all(
        np.isfinite(headings) & (headings >= 0) & (headings < 360)
    ):
        raise ValueError("Invalid NCTCOG headings.")

    print("Building spatial index...", flush=True)
    tree = shapely.STRtree(geometry)
    finished = time.perf_counter()

    elapsed = finished - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start

    print(f"\nNCTCOG pieces: {len(frame):,}")
    print(f"Metadata + Parquet read: {read_finished - started:.2f}s")
    print(f"Geometry decoding: {decode_finished - read_finished:.2f}s")
    print(f"Validation + index: {finished - decode_finished:.2f}s")
    print(f"Total setup: {elapsed:.2f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")

    return frame, geometry, tree

def match_batch(osm, nctcog, nct_geometry, tree, parent_codes):
    """Match OSM pieces using fixed distance and heading limits."""
    osm_geometry = shapely.from_wkb(
        osm["geometry_wkb"].to_numpy()
    )
    result = osm.drop(columns="geometry_wkb").reset_index(drop=True).copy()
    size = len(result)

    matched_keys = np.full(size, None, dtype=object)
    matched_pieces = np.full(size, -1, dtype=np.int64)
    matched_distances = np.full(size, np.nan)
    matched_angles = np.full(size, np.nan)
    candidate_counts = np.zeros(size, dtype=np.int32)

    source, target = tree.query(
        osm_geometry,
        predicate="dwithin",
        distance=MAX_DISTANCE_M,
    )

    if len(source):
        distances = shapely.distance(
            osm_geometry[source], nct_geometry[target]
        )
        angles = np.abs(
            (
                osm["heading_grid_deg"].to_numpy()[source]
                - nctcog["heading_grid_deg"].to_numpy()[target]
                + 180
            ) % 360 - 180
        )

        eligible = (
            (distances <= MAX_DISTANCE_M)
            & (angles <= MAX_HEADING_DIFF_DEG)
        )
        source = source[eligible]
        target = target[eligible]
        distances = distances[eligible]
        angles = angles[eligible]

        if len(source):
            # Count distinct directional parents, not their pieces.
            pairs = np.unique(
                np.column_stack([source, parent_codes[target]]),
                axis=0,
            )
            rows, counts = np.unique(pairs[:, 0], return_counts=True)
            candidate_counts[rows] = counts

            target_piece_numbers = nctcog["piece_index"].to_numpy()
            order = np.lexsort((
                target_piece_numbers[target],
                parent_codes[target],
                angles,
                distances,
                source,
            ))
            sorted_source = source[order]
            first = np.r_[True, sorted_source[1:] != sorted_source[:-1]]
            chosen = order[first]
            rows = source[chosen]
            targets = target[chosen]

            matched_keys[rows] = nctcog["parent_key"].to_numpy()[targets]
            matched_pieces[rows] = target_piece_numbers[targets]
            matched_distances[rows] = distances[chosen]
            matched_angles[rows] = angles[chosen]

    result["matched_nctcog_key"] = pd.array(matched_keys, dtype="string")
    piece_ids = pd.array(matched_pieces, dtype="Int64")
    piece_ids[matched_pieces < 0] = pd.NA
    result["matched_nctcog_piece"] = piece_ids
    result["match_distance_m"] = matched_distances
    result["heading_difference_deg"] = matched_angles
    result["candidate_parent_count"] = candidate_counts
    result["match_status"] = np.select(
        [candidate_counts == 0, candidate_counts == 1],
        ["unmatched", "matched_unique"],
        default="matched_multiple_candidates",
    )

    return result

def run_matching():
    output_file = INTERMEDIATE_DIR / "OSM_NCTCOG_piece_matches.parquet"
    temporary = output_file.with_suffix(".partial.parquet")

    parquet = pq.ParquetFile(OSM_SEGMENTS)
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"crs") != METRIC_CRS.encode():
        raise ValueError("OSM coordinate system differs from NCTCOG.")
    if metadata.get(b"geometry_encoding") != b"WKB":
        raise ValueError("Expected OSM geometry stored as WKB.")

    expected_rows = parquet.metadata.num_rows
    if expected_rows == 0:
        raise ValueError("No OSM pieces to match.")

    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    nctcog, geometry, tree = load_nctcog_index()
    parent_codes, _ = pd.factorize(nctcog["parent_key"], sort=True)

    totals = {
        "matched_unique": 0,
        "matched_multiple_candidates": 0,
        "unmatched": 0,
    }
    processed = 0
    writer = None
    last_report = time.perf_counter()

    try:
        for batch in parquet.iter_batches(batch_size=25_000):
            osm = batch.to_pandas()
            result = match_batch(
                osm, nctcog, geometry, tree, parent_codes
            )

            if len(result) != len(osm):
                raise ValueError("Matching changed the number of OSM pieces.")

            matched = result["matched_nctcog_key"].notna()
            within_limits = (
                result.loc[matched, "match_distance_m"].le(MAX_DISTANCE_M)
                & result.loc[matched, "heading_difference_deg"].le(
                    MAX_HEADING_DIFF_DEG
                )
            )
            if not within_limits.all():
                raise ValueError("A selected match exceeds the fixed limits.")
            if not matched.eq(result["candidate_parent_count"].gt(0)).all():
                raise ValueError("Candidate counts and assignments disagree.")

            for status, count in result["match_status"].value_counts().items():
                totals[status] += int(count)

            table = pa.Table.from_pandas(result, preserve_index=False)
            output_metadata = dict(table.schema.metadata or {})
            output_metadata.update({
                b"distance_crs": METRIC_CRS.encode(),
                b"max_distance_m": str(MAX_DISTANCE_M).encode(),
                b"max_heading_difference_deg": str(
                    MAX_HEADING_DIFF_DEG
                ).encode(),
                b"selection_rule": (
                    b"distance,heading_difference,parent_key,piece_index"
                ),
            })
            table = table.replace_schema_metadata(output_metadata)

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="snappy"
                )
            writer.write_table(table)
            processed += len(result)

            now = time.perf_counter()
            if now - last_report >= 5:
                elapsed = now - started
                cpu_used = sum(process.cpu_times()[:2]) - cpu_start
                cpu_pct = 100 * cpu_used / elapsed / psutil.cpu_count()
                ram = process.memory_info().rss / 1024**3
                print(
                    f"{processed:,}/{expected_rows:,} pieces "
                    f"({100 * processed / expected_rows:.1f}%) | "
                    f"{elapsed:.1f}s | {processed / elapsed:,.0f} pieces/s | "
                    f"CPU: {cpu_pct:.1f}% | RAM: {ram:.2f} GB",
                    flush=True,
                )
                last_report = now
    finally:
        if writer is not None:
            writer.close()

    if processed != expected_rows or sum(totals.values()) != expected_rows:
        raise ValueError("Final output counts do not agree.")
    if pq.read_metadata(temporary).num_rows != expected_rows:
        raise ValueError("Saved Parquet row count does not agree.")

    temporary.replace(output_file)

    print("\nMATCHING COMPLETE")
    print(f"OSM pieces: {processed:,}")
    for status, count in totals.items():
        print(f"{status}: {count:,} ({100 * count / processed:.2f}%)")
    print(f"Total elapsed: {time.perf_counter() - started:.1f} seconds")
    print(f"File size: {output_file.stat().st_size / 1024**3:.3f} GB")
    print(f"Saved: {output_file}", flush=True)

if __name__ == "__main__":
    run_matching()
