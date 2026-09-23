import time
from itertools import zip_longest

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

from paths import INTERMEDIATE_DIR


def covered_lengths(intervals, parent_count):
    if intervals.empty:
        return np.zeros(parent_count)

    ordered = intervals.sort_values(["parent", "start", "end"])
    running_end = ordered.groupby("parent", sort=False)["end"].cummax()
    previous_end = running_end.groupby(
        ordered["parent"], sort=False
    ).shift(fill_value=0.0)

    added = np.maximum(
        0.0,
        ordered["end"].to_numpy()
        - np.maximum(ordered["start"].to_numpy(), previous_end.to_numpy()),
    )
    totals = pd.Series(added, index=ordered.index).groupby(
        ordered["parent"], sort=False
    ).sum()

    return totals.reindex(range(parent_count), fill_value=0.0).to_numpy()

def collect_inferred_intervals(parent_index, parent_geometry):
    assignments = pd.read_parquet(
        INTERMEDIATE_DIR / "OSM_link_matches_after_reverse_pairs.parquet",
        columns=["link_key", "assigned_nctcog_key", "assignment_method"],
    )

    repair_methods = [
        "single_gap_topology",
        "multi_gap_topology",
        "reverse_pair_inferred",
    ]
    repairs = assignments.loc[
        assignments["assignment_method"].isin(repair_methods)
    ].set_index("link_key")

    if not repairs.index.is_unique:
        raise ValueError("Duplicate repaired OSM link keys.")
    if repairs["assigned_nctcog_key"].isna().any():
        raise ValueError("A repaired link has no NCTCOG assignment.")

    print(f"\nMeasuring {len(repairs):,} inferred OSM assignments...", flush=True)

    columns = ["parent", "start", "end", "method"]
    if repairs.empty:
        return pd.DataFrame(columns=columns)

    parquet = pq.ParquetFile(
        INTERMEDIATE_DIR / "OSM_segmented_10m.parquet"
    )
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"crs") != b"EPSG:32614":
        raise ValueError("Unexpected OSM coordinate system.")

    collected = []
    seen = set()
    backward = zero_progress = 0

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["parent_key", "geometry_wkb"],
    ):
        source = batch.to_pandas()
        selected = source.loc[source["parent_key"].isin(repairs.index)]
        if selected.empty:
            continue

        seen.update(selected["parent_key"])
        proposed = selected["parent_key"].map(repairs["assigned_nctcog_key"])
        parent_rows = parent_index.get_indexer(proposed)
        if (parent_rows < 0).any():
            raise ValueError("A repair references an unknown NCTCOG parent.")

        geometry = shapely.from_wkb(selected["geometry_wkb"].to_numpy())
        target = parent_geometry[parent_rows]

        start = shapely.line_locate_point(
            target, shapely.get_point(geometry, 0)
        )
        end = shapely.line_locate_point(
            target, shapely.get_point(geometry, -1)
        )
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            raise ValueError("Nonfinite projected positions.")

        progress = end - start
        backward += int((progress < -1e-6).sum())
        zero_progress += int((np.abs(progress) <= 1e-6).sum())
        forward = progress > 1e-6

        methods = selected["parent_key"].map(
            repairs["assignment_method"]
        ).to_numpy()

        collected.append(pd.DataFrame({
            "parent": parent_rows[forward],
            "start": start[forward],
            "end": end[forward],
            "method": methods[forward],
        }))

    if seen != set(repairs.index):
        raise ValueError("Some repaired links have no segmented geometry.")

    print(f"Inferred pieces excluded for backward projection: {backward:,}")
    print(f"Inferred pieces excluded for zero progress: {zero_progress:,}")

    return (
        pd.concat(collected, ignore_index=True)
        if collected else pd.DataFrame(columns=columns)
    )

def measure_direct_coverage():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    nct = pd.read_csv(
        INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv",
        usecols=[
            "ID", "nctcog_direction_key", "nctcog_direction",
            "FUNCL", "travel_geometry",
        ],
        dtype={"ID": "string", "nctcog_direction_key": "string"},
    )
    parent_index = pd.Index(nct["nctcog_direction_key"])
    if not parent_index.is_unique or parent_index.hasnans:
        raise ValueError("Invalid NCTCOG directional keys.")

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32614", always_xy=True
    )
    parent_geometry = shapely.transform(
        shapely.from_wkt(nct.pop("travel_geometry").to_numpy()),
        transformer.transform,
        interleaved=False,
    )
    nct["length_m"] = shapely.length(parent_geometry)

    assignments = pd.read_parquet(
        INTERMEDIATE_DIR / "OSM_link_matches_after_reverse_pairs.parquet",
        columns=["link_key", "assigned_nctcog_key"],
    ).set_index("link_key")["assigned_nctcog_key"]
    if not assignments.index.is_unique:
        raise ValueError("Duplicate OSM assignment keys.")

    source_file = pq.ParquetFile(
        INTERMEDIATE_DIR / "OSM_segmented_10m.parquet"
    )
    match_file = pq.ParquetFile(
        INTERMEDIATE_DIR / "OSM_NCTCOG_piece_matches.parquet"
    )
    if source_file.metadata.num_rows != match_file.metadata.num_rows:
        raise ValueError("Source and match row counts differ.")
    metadata = source_file.schema_arrow.metadata or {}
    if metadata.get(b"crs") != b"EPSG:32614":
        raise ValueError("Unexpected OSM piece coordinate system.")

    sources = source_file.iter_batches(
        batch_size=100_000,
        columns=["parent_key", "piece_index", "geometry_wkb"],
    )
    matches = match_file.iter_batches(
        batch_size=100_000,
        columns=["parent_key", "piece_index", "matched_nctcog_key"],
    )

    collected = []
    processed = backward = zero_progress = 0
    last_report = time.perf_counter()

    for source_batch, match_batch in zip_longest(sources, matches):
        if source_batch is None or match_batch is None:
            raise ValueError("Source and match batches differ.")

        source = source_batch.to_pandas()
        match = match_batch.to_pandas()

        for column in ["parent_key", "piece_index"]:
            if not np.array_equal(
                source[column].to_numpy(), match[column].to_numpy()
            ):
                raise ValueError("Source and match piece keys are misaligned.")

        valid = match["matched_nctcog_key"].notna()
        selected = match.loc[valid]
        if len(selected):
            parent_rows = parent_index.get_indexer(
                selected["matched_nctcog_key"]
            )
            if (parent_rows < 0).any():
                raise ValueError("Unknown NCTCOG parent in piece matches.")

            geometry = shapely.from_wkb(
                source.loc[valid, "geometry_wkb"].to_numpy()
            )
            target = parent_geometry[parent_rows]

            start = shapely.line_locate_point(
                target, shapely.get_point(geometry, 0)
            )
            end = shapely.line_locate_point(
                target, shapely.get_point(geometry, -1)
            )
            if not (np.isfinite(start).all() and np.isfinite(end).all()):
                raise ValueError("Nonfinite projected positions.")

            progress = end - start
            backward += int((progress < -1e-6).sum())
            zero_progress += int((np.abs(progress) <= 1e-6).sum())
            forward = progress > 1e-6

            # Keep track of support retained by final parent selection.
            retained = (
                selected["parent_key"].map(assignments)
                .eq(selected["matched_nctcog_key"]).fillna(False)
                .to_numpy(dtype=bool)
            )

            collected.append(pd.DataFrame({
                "parent": parent_rows[forward],
                "start": start[forward],
                "end": end[forward],
                "retained": retained[forward],
            }))

        processed += len(source)
        now = time.perf_counter()
        if now - last_report >= 5:
            elapsed = now - started
            cpu_used = sum(process.cpu_times()[:2]) - cpu_start
            print(
                f"Checked {processed:,} pieces | {elapsed:.1f}s | "
                f"CPU: {100*cpu_used/elapsed/psutil.cpu_count():.1f}% | "
                f"RAM: {process.memory_info().rss/1024**3:.2f} GB",
                flush=True,
            )
            last_report = now

    if not collected:
        raise ValueError("No matched pieces found.")

    intervals = pd.concat(collected, ignore_index=True)
    print("Merging overlapping supported intervals...", flush=True)

    nct["piece_support_m"] = covered_lengths(intervals, len(nct))
    nct["retained_direct_m"] = covered_lengths(
        intervals.loc[intervals["retained"]], len(nct)
    )

    for column in ["piece_support_m", "retained_direct_m"]:
        if (nct[column] > nct["length_m"] + 1e-6).any():
            raise ValueError("Covered length exceeds parent length.")

    nct["retained_direct_pct"] = (
        100 * nct["retained_direct_m"] / nct["length_m"]
    )

    print("\nNCTCOG DIRECT LENGTH COVERAGE")
    for column in ["piece_support_m", "retained_direct_m"]:
        percentage = 100 * nct[column].sum() / nct["length_m"].sum()
        print(f"{column}: {percentage:.2f}% of NCTCOG directional length")

    print(f"Excluded backward projections: {backward:,}")
    print(f"Excluded zero-progress projections: {zero_progress:,}")

    percentage = nct["retained_direct_pct"]
    nct["coverage_band"] = pd.cut(
        percentage,
        bins=[-np.inf, 0, 25, 50, 75, 99, np.inf],
        labels=["0%", ">0–25%", ">25–50%", ">50–75%", ">75–99%", ">99%"],
    )

    print("\nDIRECTIONAL RECORDS BY RETAINED DIRECT COVERAGE")
    counts = nct["coverage_band"].value_counts(sort=False)
    print(pd.DataFrame({
        "records": counts,
        "records_%": (100 * counts / len(nct)).round(2),
    }).to_string())

    print("\nBY FUNCL")
    report = nct.groupby("FUNCL", dropna=False).agg(
        records=("nctcog_direction_key", "size"),
        total_length_m=("length_m", "sum"),
        piece_support_m=("piece_support_m", "sum"),
        retained_direct_m=("retained_direct_m", "sum"),
    )
    report["piece_support_%"] = (
        100 * report["piece_support_m"] / report["total_length_m"]
    ).round(2)
    report["retained_direct_%"] = (
        100 * report["retained_direct_m"] / report["total_length_m"]
    ).round(2)
    print(report[["records", "piece_support_%", "retained_direct_%"]].to_string())

    destination = INTERMEDIATE_DIR / "NCTCOG_direct_length_coverage.parquet"
    temporary = destination.with_suffix(".partial.parquet")
    nct.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(destination)

    print(f"\nElapsed: {time.perf_counter()-started:.1f}s")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    measure_direct_coverage()