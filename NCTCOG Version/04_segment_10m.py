import numpy as np
import shapely
from shapely.ops import substring
import pandas as pd
from pyproj import Transformer
import time
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from paths import OSM_INPUT, INTERMEDIATE_DIR, check_inputs

SEGMENT_LENGTH_M = 10.0

METRIC_CRS = "EPSG:32614"
TO_METRIC = Transformer.from_crs(
    "EPSG:4326", METRIC_CRS, always_xy=True
)


def split_linestring(line, segment_length=SEGMENT_LENGTH_M):
    """Split a projected, metre-based LineString in travel order."""
    if not np.isfinite(segment_length) or segment_length <= 0:
        raise ValueError("Segment length must be finite and positive.")

    if line is None or line.geom_type != "LineString" or line.is_empty:
        raise ValueError("Expected a nonempty LineString.")

    length = line.length
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Geometry length must be finite and positive.")

    if length <= segment_length:
        return np.asarray([line], dtype=object)

    distances = np.arange(0.0, length, segment_length)
    distances = np.append(distances, length)
    coordinates = np.asarray(line.coords)

    # Fast path: build all pieces of a straight link together.
    if len(coordinates) == 2:
        fractions = distances / length
        points = (
            coordinates[0]
            + fractions[:, None] * (coordinates[-1] - coordinates[0])
        )
        points[0] = coordinates[0]
        points[-1] = coordinates[-1]

        return shapely.linestrings(
            np.stack([points[:-1], points[1:]], axis=1)
        )

    # Preserve intermediate vertices on links containing bends.
    return np.asarray(
        [
            substring(line, start, end)
            for start, end in zip(distances[:-1], distances[1:])
        ],
        dtype=object,
    )

def segment_batch(frame, key_column, geometry_column):
    """Prepare matching pieces; output geometry remains in METRIC_CRS."""
    if frame.empty:
        raise ValueError("Cannot segment an empty batch.")

    keys = frame[key_column].astype("string")
    if keys.isna().any() or keys.str.strip().eq("").any():
        raise ValueError("Missing parent keys.")
    if keys.duplicated().any():
        raise ValueError("Duplicate parent keys within batch.")

    geometry = shapely.from_wkt(frame[geometry_column].to_numpy())
    projected = shapely.transform(
        geometry, TO_METRIC.transform, interleaved=False
    )

    groups = [split_linestring(line) for line in projected]
    counts = np.asarray([len(group) for group in groups])
    pieces = np.concatenate(groups)
    parent_rows = np.repeat(np.arange(len(frame)), counts)
    offsets = np.cumsum(counts) - counts

    lengths = shapely.length(pieces)
    totals = np.add.reduceat(lengths, offsets)
    if not np.allclose(
        totals, shapely.length(projected), rtol=1e-9, atol=1e-6
    ):
        raise ValueError("Splitting changed a parent geometry's length.")
    if not np.all(
        np.isfinite(lengths)
        & (lengths > 0)
        & (lengths <= SEGMENT_LENGTH_M + 1e-6)
    ):
        raise ValueError("Invalid piece lengths.")

    starts = shapely.get_point(pieces, 0)
    ends = shapely.get_point(pieces, -1)
    dx = shapely.get_x(ends) - shapely.get_x(starts)
    dy = shapely.get_y(ends) - shapely.get_y(starts)

    # Clockwise from grid north, calculated separately for each piece.
    headings = np.degrees(np.arctan2(dx, dy)) % 360
    headings[np.hypot(dx, dy) <= 1e-9] = np.nan

    result = pd.DataFrame({
        "parent_key": keys.to_numpy()[parent_rows],
        "piece_index": np.arange(len(pieces)) - np.repeat(offsets, counts),
        "length_m": lengths,
        "heading_grid_deg": headings,
        "geometry": pieces,
    })

    if "facility_type" in frame.columns:
        result["facility_type"] = (
            frame["facility_type"].to_numpy()[parent_rows]
        )

    return result

def process_file(
    label, input_file, output_file, key_column, geometry_column,
    extra_columns, chunk_size, expected_parents, expected_pieces,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".partial.parquet")
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])
    last_report = started
    parents = pieces = 0
    writer = None

    print(f"\n--- {label}: starting ---", flush=True)

    try:
        reader = pd.read_csv(
            input_file,
            usecols=[key_column, geometry_column] + extra_columns,
            dtype={key_column: "string"},
            chunksize=chunk_size,
        )

        for frame in reader:
            result = segment_batch(frame, key_column, geometry_column)

            missing = result["heading_grid_deg"].isna()
            if missing.any():
                examples = result.loc[
                    missing, ["parent_key", "piece_index"]
                ].head()
                raise ValueError(f"Undefined headings:\n{examples}")

            result["geometry_wkb"] = shapely.to_wkb(
                result.pop("geometry").to_numpy()
            )
            table = pa.Table.from_pandas(result, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update({
                b"geometry_encoding": b"WKB",
                b"geometry_column": b"geometry_wkb",
                b"crs": METRIC_CRS.encode(),
                b"heading_reference": b"clockwise_from_grid_north",
            })
            table = table.replace_schema_metadata(metadata)

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="snappy"
                )
            writer.write_table(table)

            parents += len(frame)
            pieces += len(result)
            now = time.perf_counter()

            if now - last_report >= 5:
                elapsed = now - started
                cpu_used = sum(process.cpu_times()[:2]) - cpu_start
                cpu_pct = 100 * cpu_used / elapsed / psutil.cpu_count()
                ram = process.memory_info().rss / 1024**3
                print(
                    f"{label}: {parents:,} parents | {pieces:,} pieces | "
                    f"{elapsed:.1f}s | {pieces / elapsed:,.0f} pieces/s | "
                    f"CPU: {cpu_pct:.1f}% | RAM: {ram:.2f} GB",
                    flush=True,
                )
                last_report = now
    finally:
        if writer is not None:
            writer.close()

    if parents == 0 or pieces == 0:
        raise ValueError(f"{label}: empty segmented output.")
    if ((expected_parents is not None and parents != expected_parents)
            or (expected_pieces is not None and pieces != expected_pieces)):
        raise ValueError(
            f"{label}: unexpected counts: {parents:,} parents, "
            f"{pieces:,} pieces. Output remains a partial file."
        )

    saved_rows = pq.read_metadata(temporary).num_rows
    if saved_rows != pieces:
        raise ValueError(f"Saved row count differs: {saved_rows:,}")

    temporary.replace(output_file)
    elapsed = time.perf_counter() - started

    print(f"\n{label} COMPLETE", flush=True)
    print(f"Parents: {parents:,}; pieces: {pieces:,}")
    print(f"Elapsed including writing: {elapsed:.1f} seconds")
    print(f"File size: {output_file.stat().st_size / 1024**3:.3f} GB")
    print(f"Saved: {output_file}", flush=True)

def main():
    check_inputs()

    process_file(
        label="OSM",
        input_file=OSM_INPUT,
        output_file=INTERMEDIATE_DIR / "OSM_segmented_10m.parquet",
        key_column="link_key",
        geometry_column="geometry",
        extra_columns=["facility_type"],
        chunk_size=50_000,
        expected_parents=None,
        expected_pieces=None,
    )

    process_file(
        label="NCTCOG",
        input_file=INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv",
        output_file=INTERMEDIATE_DIR / "NCTCOG_segmented_10m.parquet",
        key_column="nctcog_direction_key",
        geometry_column="travel_geometry",
        extra_columns=[],
        chunk_size=2_000,
        expected_parents=69_477,
        expected_pieces=5_522_572,
    )


if __name__ == "__main__":
    main()
