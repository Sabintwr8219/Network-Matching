import os

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely import wkt
from shapely import distance as shapely_distance

from paths import MOTORWAY_DIR, OSM_DIR, TRUNK_DIR, TXDOT_DIR


TXDOT_INPUT = TXDOT_DIR / "TxDOT_segmented_10m.csv"
OSM_INPUT = OSM_DIR / "OSM_segmented_10m.csv"

MOTORWAY_OUTPUT = MOTORWAY_DIR / "matched_network_basic_motorway.csv"
TRUNK_OUTPUT = TRUNK_DIR / "matched_network_basic_trunk.csv"

DISTANCE_THRESHOLD_M = 6
HEADING_THRESHOLD_DEG = 15
CHUNK_SIZE = 100_000
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 1))


# Return the smallest circular difference between two headings.
def heading_difference(values, heading):
    diff = np.abs(values - heading)
    return np.minimum(diff, 360 - diff)


# Convert directional values from CSV into Boolean values.
def directional_false_mask(series):
    return series.astype("string").str.strip().str.lower().isin(["false", "0"])


# Load and project the TxDOT 10-meter network used for candidate matching.
def load_txdot_targets():
    df = pd.read_csv(TXDOT_INPUT, low_memory=False)
    df["geometry"] = df["Geometry"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
    gdf["Heading"] = pd.to_numeric(gdf["Heading"], errors="coerce")

    motorway = gdf[
        ~gdf["RDBD_TYPE"].isin(
            [
                "Single Roadbed",
                "Other (used for weigh station approach lanes",
            ]
        )
    ].copy()

    trunk = gdf.copy()

    return motorway.reset_index(drop=True), trunk.reset_index(drop=True)


# Match one projected OSM segment to the closest eligible TxDOT segment.
def match_one(geometry, heading, target_geometries, target_headings, target_gids, spatial_index):
    if geometry is None or pd.isna(heading):
        return None, None, None, None

    minx, miny, maxx, maxy = geometry.bounds
    bounds = (
        minx - DISTANCE_THRESHOLD_M,
        miny - DISTANCE_THRESHOLD_M,
        maxx + DISTANCE_THRESHOLD_M,
        maxy + DISTANCE_THRESHOLD_M,
    )

    candidate_idx = np.asarray(list(spatial_index.intersection(bounds)), dtype=int)

    if candidate_idx.size == 0:
        return None, None, None, None

    candidate_headings = target_headings[candidate_idx]
    heading_diff = heading_difference(candidate_headings, heading)
    candidate_idx = candidate_idx[heading_diff <= HEADING_THRESHOLD_DEG]

    if candidate_idx.size == 0:
        return None, None, None, None

    candidate_geometries = target_geometries[candidate_idx]
    distances = np.asarray(shapely_distance(candidate_geometries, geometry), dtype=float)
    keep = distances <= DISTANCE_THRESHOLD_M

    candidate_idx = candidate_idx[keep]
    distances = distances[keep]

    if candidate_idx.size == 0:
        return None, None, None, None

    best_position = int(np.argmin(distances))
    best_idx = candidate_idx[best_position]

    gids = target_gids[candidate_idx].tolist()
    candidate_distances = np.round(distances, 2).tolist()

    return (
        target_gids[best_idx],
        float(distances[best_position]),
        gids,
        candidate_distances,
    )


# Match a batch of projected OSM rows while sharing the TxDOT spatial index.
def match_batch(rows, target_geometries, target_headings, target_gids, spatial_index):
    results = []

    for row in rows:
        results.append(
            match_one(
                row.geometry,
                row.heading,
                target_geometries,
                target_headings,
                target_gids,
                spatial_index,
            )
        )

    return results


# Match one OSM facility type and write results incrementally to CSV.
def match_facility(facility_type, target, output_file, require_non_directional=False):
    if output_file.exists():
        output_file.unlink()

    target_geometries = np.asarray(target.geometry.values, dtype=object)
    target_headings = target["Heading"].to_numpy(dtype=float)
    target_gids = target["GID"].to_numpy()
    spatial_index = target.sindex

    first_write = True
    processed = 0
    matched = 0

    reader = pd.read_csv(
        OSM_INPUT,
        chunksize=CHUNK_SIZE,
        dtype={"osm_short_a": "object", "osm_short_b": "object"},
        low_memory=False,
    )

    for chunk in reader:
        mask = chunk["facility_type"].astype("string").str.lower().eq(facility_type)

        if require_non_directional:
            mask &= directional_false_mask(chunk["directional"])

        chunk = chunk.loc[mask].copy().reset_index(drop=True)

        if chunk.empty:
            continue

        original = chunk.copy()
        chunk["geometry"] = chunk["geometry"].apply(wkt.loads)
        projected = gpd.GeoDataFrame(chunk, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
        projected["heading"] = pd.to_numeric(projected["heading"], errors="coerce")

        rows = list(projected[["geometry", "heading"]].itertuples(index=False))
        batch_size = max(1, int(np.ceil(len(rows) / max(1, N_JOBS * 4))))
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

        batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(match_batch)(
                batch,
                target_geometries,
                target_headings,
                target_gids,
                spatial_index,
            )
            for batch in batches
        )

        results = [result for batch in batch_results for result in batch]

        original["Matched_GID"] = [result[0] for result in results]
        original["Matched_Distance"] = [result[1] for result in results]
        original["Candidate_GIDs"] = [result[2] for result in results]
        original["Candidate_Distances"] = [result[3] for result in results]

        original.to_csv(
            output_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        processed += len(original)
        matched += original["Matched_GID"].notna().sum()
        first_write = False

    print(f"{facility_type.title()} segments: {processed:,}")
    print(f"{facility_type.title()} matched: {matched:,}")
    print(f"{facility_type.title()} unmatched: {processed - matched:,}")


# Run motorway and trunk spatial matching.
def main():
    print("Stage 04 - Freeway spatial matching")

    motorway_target, trunk_target = load_txdot_targets()

    match_facility(
        "motorway",
        motorway_target,
        MOTORWAY_OUTPUT,
        require_non_directional=True,
    )

    match_facility(
        "trunk",
        trunk_target,
        TRUNK_OUTPUT,
        require_non_directional=False,
    )

    print("Saved:", MOTORWAY_OUTPUT)
    print("Saved:", TRUNK_OUTPUT)


if __name__ == "__main__":
    main()
