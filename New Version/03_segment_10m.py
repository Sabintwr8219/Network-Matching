import os

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely import wkt
from shapely.geometry import LineString

from paths import OSM_DIR, TXDOT_DIR


OSM_INPUT = OSM_DIR / "link_short_sequence_complete.csv"
TXDOT_INPUT = TXDOT_DIR / "link_list_with_speed_NOL_FC.csv"

OSM_OUTPUT = OSM_DIR / "OSM_segmented_10m.csv"
TXDOT_OUTPUT = TXDOT_DIR / "TxDOT_segmented_10m.csv"

SEGMENT_LENGTH_M = 10
CHUNK_SIZE = 50_000
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


# Split one projected LineString into approximately 10-meter segments.
def split_linestring(line, segment_length=SEGMENT_LENGTH_M):
    length = line.length

    if length <= segment_length:
        return [line]

    distances = np.arange(0, length, segment_length)

    if distances[-1] != length:
        distances = np.append(distances, length)

    points = [line.interpolate(distance) for distance in distances]

    return [
        LineString([points[i], points[i + 1]])
        for i in range(len(points) - 1)
    ]


# Segment one OSM short-link record and keep all source attributes.
def segment_osm_record(record):
    geometry = record.pop("geometry")
    segments = split_linestring(geometry)
    link_id = record.get("link_id")

    return [
        {
            **record,
            "segment_id": f"{link_id}_{i}",
            "segment_length_m": segment.length,
            "geometry": segment,
        }
        for i, segment in enumerate(segments)
    ]


# Segment one TxDOT link record and keep all source attributes.
def segment_txdot_record(record):
    geometry = record.pop("geometry")
    segments = split_linestring(geometry)
    link_id = record.get("LinkID")

    return [
        {
            **record,
            "segment_id": f"{link_id}_{i}",
            "segment_length_m": segment.length,
            "geometry": segment,
        }
        for i, segment in enumerate(segments)
    ]


# Convert one projected result batch back to WGS84 and append it to CSV.
def save_segment_batch(rows, output_file, geometry_name, first_write):
    if not rows:
        return first_write, 0

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857").to_crs("EPSG:4326")
    output = pd.DataFrame(gdf.drop(columns="geometry"))
    output[geometry_name] = gdf.geometry.map(lambda geom: geom.wkt if geom is not None else None)

    output.to_csv(
        output_file,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )

    return False, len(output)


# Segment motorway and trunk OSM short links in chunks.
def segment_osm():
    if OSM_OUTPUT.exists():
        OSM_OUTPUT.unlink()

    first_write = True
    input_rows = 0
    output_rows = 0

    reader = pd.read_csv(
        OSM_INPUT,
        chunksize=CHUNK_SIZE,
        dtype={"osm_short_a": "object", "osm_short_b": "object"},
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk[
            chunk["facility_type"].astype("string").str.lower().isin(["motorway", "trunk"])
        ].copy()

        if chunk.empty:
            continue

        input_rows += len(chunk)
        chunk["geometry"] = chunk["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(chunk, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

        records = gdf.to_dict("records")
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(segment_osm_record)(record.copy()) for record in records
        )
        rows = [item for group in results for item in group]

        first_write, saved = save_segment_batch(rows, OSM_OUTPUT, "geometry", first_write)
        output_rows += saved

    print(f"OSM links: {input_rows:,}")
    print(f"OSM 10-m segments: {output_rows:,}")


# Segment TxDOT links except Single Roadbed links in chunks.
def segment_txdot():
    if TXDOT_OUTPUT.exists():
        TXDOT_OUTPUT.unlink()

    first_write = True
    input_rows = 0
    output_rows = 0

    reader = pd.read_csv(TXDOT_INPUT, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in reader:
        chunk = chunk[chunk["RDBD_TYPE"] != "Single Roadbed"].copy()

        if chunk.empty:
            continue

        input_rows += len(chunk)
        chunk["geometry"] = chunk["Geometry"].apply(wkt.loads)
        chunk = chunk.drop(columns=["Geometry"])
        gdf = gpd.GeoDataFrame(chunk, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

        records = gdf.to_dict("records")
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(segment_txdot_record)(record.copy()) for record in records
        )
        rows = [item for group in results for item in group]

        first_write, saved = save_segment_batch(rows, TXDOT_OUTPUT, "Geometry", first_write)
        output_rows += saved

    print(f"TxDOT links: {input_rows:,}")
    print(f"TxDOT 10-m segments: {output_rows:,}")


# Run OSM and TxDOT 10-meter segmentation.
def main():
    print("Stage 03 - 10 m segmentation")
    segment_osm()
    segment_txdot()
    print("Saved:", OSM_OUTPUT)
    print("Saved:", TXDOT_OUTPUT)


if __name__ == "__main__":
    main()
