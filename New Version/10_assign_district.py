import os

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from shapely import wkt
from shapely.strtree import STRtree

from paths import INTERMEDIATE_DIR, TXDOT_DISTRICTS


INPUT_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT_with_Districts.csv"

GEOMETRY_COLUMN = "geometry"
DISTRICT_COLUMN_1 = "TXDOT_DI_1"
DISTRICT_COLUMN_2 = "TXDOT_DI_2"
CHUNK_SIZE = 500_000
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 1))


# Load district polygons in WGS84 and build their spatial index.
def load_districts():
    districts = gpd.read_file(TXDOT_DISTRICTS)

    if districts.crs is None:
        districts = districts.set_crs("EPSG:4326")

    districts = districts.to_crs("EPSG:4326").reset_index(drop=True)

    missing = [
        column
        for column in [DISTRICT_COLUMN_1, DISTRICT_COLUMN_2]
        if column not in districts.columns
    ]

    if missing:
        raise ValueError(
            f"Missing district columns: {missing}. "
            f"Available columns: {districts.columns.tolist()}"
        )

    geometries = list(districts.geometry)
    tree = STRtree(geometries)

    return districts, geometries, tree


# Return the first district containing or touching one roadway midpoint.
def find_district(point, districts, geometries, tree):
    candidate_indexes = tree.query(point)

    for district_idx in candidate_indexes:
        district_idx = int(district_idx)
        district_geometry = geometries[district_idx]

        if district_geometry.contains(point) or district_geometry.touches(point):
            return (
                districts.loc[district_idx, DISTRICT_COLUMN_1],
                districts.loc[district_idx, DISTRICT_COLUMN_2],
            )

    return None, None


# Assign districts to one batch of roadway midpoint geometries.
def assign_batch(points, districts, geometries, tree):
    return [find_district(point, districts, geometries, tree) for point in points]


# Assign TxDOT districts to the complete network in large CSV chunks.
def main():
    print("Stage 10 - District assignment")

    districts, district_geometries, district_tree = load_districts()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    first_write = True
    total_input = 0
    total_saved = 0
    total_assigned = 0

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk_number, chunk in enumerate(reader, start=1):
        total_input += len(chunk)
        chunk = chunk[chunk[GEOMETRY_COLUMN].notna()].copy()

        if chunk.empty:
            continue

        chunk["geometry_obj"] = chunk[GEOMETRY_COLUMN].apply(wkt.loads)
        roads = gpd.GeoDataFrame(chunk, geometry="geometry_obj", crs="EPSG:4326")
        points = roads.geometry.interpolate(0.5, normalized=True).tolist()

        batch_size = max(1, int(len(points) / max(1, N_JOBS * 4)))
        batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]

        batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(assign_batch)(
                batch,
                districts,
                district_geometries,
                district_tree,
            )
            for batch in batches
        )

        results = [result for batch in batch_results for result in batch]

        roads[DISTRICT_COLUMN_1] = [result[0] for result in results]
        roads[DISTRICT_COLUMN_2] = [result[1] for result in results]

        assigned = int(roads[DISTRICT_COLUMN_2].notna().sum())

        if chunk_number == 1 and assigned == 0:
            raise RuntimeError("No districts were assigned in the first chunk.")

        roads = roads.drop(columns=["geometry_obj"])
        roads.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        total_saved += len(roads)
        total_assigned += assigned
        first_write = False

        print(
            f"Chunk {chunk_number}: "
            f"{len(roads):,} links, {assigned:,} assigned"
        )

    print(f"Input links: {total_input:,}")
    print(f"Saved links: {total_saved:,}")
    print(f"District assigned: {total_assigned:,}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
