import os

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely import wkt
from shapely import distance as shapely_distance
from shapely.wkt import dumps as wkt_dumps

from paths import INTERMEDIATE_DIR, OSM_DIR, TXDOT_DIR


TXDOT_INPUT = TXDOT_DIR / "link_list_with_speed_NOL_FC.csv"
OSM_INPUT = OSM_DIR / "link_short_sequence_complete.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "TxDOT Network - Arterial.csv"
TEMP_FILE = INTERMEDIATE_DIR / "TxDOT Network - Arterial_raw.csv"

DISTANCE_THRESHOLD_M = 6
HEADING_THRESHOLD_DEG = 15
CHUNK_SIZE = 100_000
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 1))

UNWANTED_ROADBEDS = [
    "Left Roadbed",
    "Right Roadbed",
    "Left Supplemental Main Lane",
    "Right Supplemental Main Lane",
    "Other (used for weigh station approach lanes",
    "Left Managed Lane",
    "Left Supplemental Managed Lane",
    "Right Supplemental Managed Lane",
    "Right Managed Lane",
]


# Return the smallest circular difference between candidate and OSM headings.
def heading_difference(values, heading):
    diff = np.abs(values - heading)
    return np.minimum(diff, 360 - diff)


# Load and project the TxDOT links eligible for arterial matching.
def load_txdot_targets():
    df = pd.read_csv(TXDOT_INPUT, low_memory=False)
    df = df[~df["RDBD_TYPE"].isin(UNWANTED_ROADBEDS)].copy()
    df["geometry"] = df["Geometry"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
    gdf["Heading"] = pd.to_numeric(gdf["Heading"], errors="coerce")

    return gdf.reset_index(drop=True)


# Match one projected OSM short link to the closest eligible TxDOT link.
def match_one(geometry, heading, target_geometries, target_headings, target_gids, target_roadbeds, target_fc, spatial_index):
    if geometry is None or pd.isna(heading):
        return None, None, None, None, None, None

    minx, miny, maxx, maxy = geometry.bounds
    bounds = (
        minx - DISTANCE_THRESHOLD_M,
        miny - DISTANCE_THRESHOLD_M,
        maxx + DISTANCE_THRESHOLD_M,
        maxy + DISTANCE_THRESHOLD_M,
    )

    candidate_idx = np.asarray(list(spatial_index.intersection(bounds)), dtype=int)

    if candidate_idx.size == 0:
        return None, None, None, None, None, None

    candidate_headings = target_headings[candidate_idx]
    heading_diff = heading_difference(candidate_headings, heading)
    candidate_idx = candidate_idx[heading_diff <= HEADING_THRESHOLD_DEG]

    if candidate_idx.size == 0:
        return None, None, None, None, None, None

    candidate_geometries = target_geometries[candidate_idx]
    distances = np.asarray(shapely_distance(candidate_geometries, geometry), dtype=float)
    keep = distances <= DISTANCE_THRESHOLD_M

    candidate_idx = candidate_idx[keep]
    distances = distances[keep]

    if candidate_idx.size == 0:
        return None, None, None, None, None, None

    best_position = int(np.argmin(distances))
    best_idx = candidate_idx[best_position]

    return (
        target_gids[best_idx],
        float(distances[best_position]),
        target_gids[candidate_idx].tolist(),
        np.round(distances, 2).tolist(),
        target_roadbeds[best_idx],
        target_fc[best_idx],
    )


# Match one batch of projected arterial OSM rows.
def match_batch(rows, target_geometries, target_headings, target_gids, target_roadbeds, target_fc, spatial_index):
    results = []

    for row in rows:
        results.append(
            match_one(
                row.geometry,
                row.heading,
                target_geometries,
                target_headings,
                target_gids,
                target_roadbeds,
                target_fc,
                spatial_index,
            )
        )

    return results


# Match all non-motorway and non-trunk OSM short links in chunks.
def run_matching(target):
    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    target_geometries = np.asarray(target.geometry.values, dtype=object)
    target_headings = target["Heading"].to_numpy(dtype=float)
    target_gids = target["GID"].to_numpy()
    target_roadbeds = target["RDBD_TYPE"].to_numpy(dtype=object)
    target_fc = target["Functional Class"].to_numpy(dtype=object)
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
        mask = ~chunk["facility_type"].astype("string").str.lower().isin(["motorway", "trunk"])
        chunk = chunk.loc[mask].copy().reset_index(drop=True)

        if chunk.empty:
            continue

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
                target_roadbeds,
                target_fc,
                spatial_index,
            )
            for batch in batches
        )

        results = [result for batch in batch_results for result in batch]

        output = projected.to_crs("EPSG:4326")
        output["Matched_GID"] = [result[0] for result in results]
        output["Matched_Distance"] = [result[1] for result in results]
        output["Candidate_GIDs"] = [result[2] for result in results]
        output["Candidate_Distances"] = [result[3] for result in results]
        output["RDBD_TYPE"] = [result[4] for result in results]
        output["Functional Class"] = [result[5] for result in results]
        output["geometry"] = output.geometry.apply(wkt_dumps)

        pd.DataFrame(output).to_csv(
            TEMP_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        processed += len(output)
        matched += output["Matched_GID"].notna().sum()
        first_write = False

    return processed, matched


# Build corrections for exactly two flipped links when only one direction matched.
def build_pair_corrections():
    pair_df = pd.read_csv(
        TEMP_FILE,
        usecols=[
            "link_id",
            "from_node_id",
            "to_node_id",
            "Matched_GID",
            "RDBD_TYPE",
            "Functional Class",
        ],
        low_memory=False,
    )

    from_text = pair_df["from_node_id"].astype(str)
    to_text = pair_df["to_node_id"].astype(str)

    pair_df["pair_key"] = np.where(
        from_text <= to_text,
        from_text + "_" + to_text,
        to_text + "_" + from_text,
    )

    group_size = pair_df.groupby("pair_key")["link_id"].transform("size")
    valid_gid_count = pair_df.groupby("pair_key")["Matched_GID"].transform("count")
    eligible = pair_df[(group_size == 2) & (valid_gid_count == 1)].copy()

    if eligible.empty:
        return {}, {}, {}

    matched_gid = eligible.groupby("pair_key")["Matched_GID"].transform("first")
    matched_roadbed = eligible.groupby("pair_key")["RDBD_TYPE"].transform("first")
    matched_fc = eligible.groupby("pair_key")["Functional Class"].transform("first")

    unmatched = eligible[eligible["Matched_GID"].isna()].copy()
    unmatched["new_gid"] = matched_gid.loc[unmatched.index].map(lambda gid: f"{gid}_1")
    unmatched["new_roadbed"] = matched_roadbed.loc[unmatched.index]
    unmatched["new_fc"] = matched_fc.loc[unmatched.index]

    gid_map = dict(zip(unmatched["link_id"], unmatched["new_gid"]))
    roadbed_map = dict(zip(unmatched["link_id"], unmatched["new_roadbed"]))
    fc_map = dict(zip(unmatched["link_id"], unmatched["new_fc"]))

    return gid_map, roadbed_map, fc_map


# Apply flipped-link corrections and write the final arterial file.
def apply_pair_corrections(gid_map, roadbed_map, fc_map):
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    first_write = True
    corrected = 0

    for chunk in pd.read_csv(TEMP_FILE, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["Matched_GID"] = chunk["Matched_GID"].astype("object")
        mask = chunk["link_id"].isin(gid_map)

        if mask.any():
            link_ids = chunk.loc[mask, "link_id"]
            chunk.loc[mask, "Matched_GID"] = link_ids.map(gid_map).to_numpy()
            chunk.loc[mask, "RDBD_TYPE"] = link_ids.map(roadbed_map).to_numpy()
            chunk.loc[mask, "Functional Class"] = link_ids.map(fc_map).to_numpy()
            corrected += int(mask.sum())

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

    TEMP_FILE.unlink(missing_ok=True)
    return corrected


# Run arterial matching and flipped-link correction.
def main():
    print("Stage 08 - Arterial matching")

    target = load_txdot_targets()
    processed, matched = run_matching(target)
    gid_map, roadbed_map, fc_map = build_pair_corrections()
    corrected = apply_pair_corrections(gid_map, roadbed_map, fc_map)

    print(f"Arterial links: {processed:,}")
    print(f"Matched: {matched:,}")
    print(f"Unmatched before pair correction: {processed - matched:,}")
    print(f"Synthetic reverse matches: {corrected:,}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
