import argparse
import time

import numpy as np
import pandas as pd
import psutil
import shapely

from paths import INTERMEDIATE_DIR, OSM_INPUT


CURRENT_LINKS = INTERMEDIATE_DIR / "OSM_link_matches_after_multi_gap.parquet"
NCTCOG_INPUT = INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv"
REVERSE_CANDIDATES = INTERMEDIATE_DIR / "OSM_reverse_pair_candidates.parquet"

REVERSE_REPAIRED_OUTPUT = (
    INTERMEDIATE_DIR / "OSM_link_matches_after_reverse_pairs.parquet"
)

JUNCTION_CANDIDATES = (
    INTERMEDIATE_DIR / "OSM_junction_candidates.parquet"
)

def check_reverse_geometry(pairs):
    if pairs.empty:
        return np.zeros(0, dtype=bool)

    needed = set(pairs["link_key"]) | set(pairs["source_link_key"])
    selected = []

    for chunk in pd.read_csv(
        OSM_INPUT,
        usecols=["link_key", "geometry"],
        dtype={"link_key": "string"},
        chunksize=100_000,
    ):
        keep = chunk["link_key"].isin(needed)
        if keep.any():
            selected.append(chunk.loc[keep])

    geometry = pd.concat(selected, ignore_index=True).set_index("link_key")
    if not geometry.index.is_unique or set(geometry.index) != needed:
        raise ValueError("Missing or duplicate OSM geometries.")

    source = shapely.from_wkt(
        geometry.loc[pairs["source_link_key"], "geometry"].to_numpy()
    )
    reverse = shapely.from_wkt(
        geometry.loc[pairs["link_key"], "geometry"].to_numpy()
    )

    # Tiny coordinate-rounding tolerance in degrees, not a search radius.
    return shapely.equals_exact(
        shapely.reverse(source), reverse, tolerance=1e-10
    )

def evaluate_reverse_pairs():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    print("Reading current assignments...", flush=True)
    links = pd.read_parquet(CURRENT_LINKS)
    if not links["link_key"].is_unique:
        raise ValueError("Duplicate OSM link keys.")

    nodes = ["from_node_id", "to_node_id"]
    for column in nodes:
        links[column] = links[column].astype("string")
        if links[column].isna().any() or links[column].str.strip().eq("").any():
            raise ValueError(f"Missing node identifiers: {column}")

    pair_count = links.groupby(nodes)["link_key"].transform("size")
    unique_pair = pair_count.eq(1)
    non_loop = links["from_node_id"].ne(links["to_node_id"])

    reliable = (
        links["assignment_method"].eq("direct_spatial_dominant")
        & links["consolidation_status"].eq("single_parent")
        & links["ambiguous_length_m"].eq(0)
        & links["assigned_nctcog_key"].notna()
        & np.isclose(
            links["matched_length_m"], links["total_length_m"],
            rtol=1e-9, atol=1e-6,
        )
    )

    sources = links.loc[
        reliable & unique_pair & non_loop,
        ["link_key", "from_node_id", "to_node_id", "assigned_nctcog_key"],
    ].rename(columns={
        "link_key": "source_link_key",
        "from_node_id": "source_from",
        "to_node_id": "source_to",
        "assigned_nctcog_key": "source_nctcog_key",
    })

    nct = pd.read_csv(
        NCTCOG_INPUT,
        usecols=["ID", "Dir", "nctcog_direction_key", "nctcog_direction", "FUNCL"],
        dtype={"ID": "string", "nctcog_direction_key": "string"},
    )

    # Only records prepared as two-way can supply an opposite direction.
    two_way = nct.loc[nct["Dir"].eq(0)]

    sources = sources.merge(
        two_way[["ID", "nctcog_direction_key", "nctcog_direction"]]
        .rename(columns={"nctcog_direction_key": "source_nctcog_key"}),
        on="source_nctcog_key",
        how="inner",
        validate="many_to_one",
    )
    sources["opposite_direction"] = np.where(
        sources["nctcog_direction"].eq("AB"), "BA", "AB"
    )

    sources = sources.merge(
        two_way[["ID", "nctcog_direction", "nctcog_direction_key", "FUNCL"]]
        .rename(columns={
            "nctcog_direction": "opposite_direction",
            "nctcog_direction_key": "proposed_nctcog_key",
        }),
        on=["ID", "opposite_direction"],
        how="inner",
        validate="many_to_one",
    )

    targets = links.loc[
        links["assigned_nctcog_key"].isna() & unique_pair & non_loop,
        ["link_key", "from_node_id", "to_node_id", "facility_type"],
    ]

    pairs = sources.merge(
        targets,
        left_on=["source_to", "source_from"],
        right_on=["from_node_id", "to_node_id"],
        how="inner",
        validate="one_to_one",
    )

    print(f"Node-reversed candidates: {len(pairs):,}", flush=True)
    print("Checking reversed geometry...", flush=True)
    pairs["reverse_geometry_ok"] = check_reverse_geometry(pairs)
    pairs["repair_status"] = "candidate_not_applied"

    temporary = REVERSE_CANDIDATES.with_suffix(".partial.parquet")
    pairs.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(REVERSE_CANDIDATES)

    eligible = pairs.loc[pairs["reverse_geometry_ok"]]
    represented = set(links["assigned_nctcog_key"].dropna())
    new_parents = set(eligible["proposed_nctcog_key"]) - represented

    print("\nREVERSE-PAIR EVALUATION")
    print(f"Candidates with matching reverse geometry: {len(eligible):,}")
    print(f"Geometry disagreements: {(~pairs['reverse_geometry_ok']).sum():,}")
    print(f"Potential newly represented NCTCOG directions: {len(new_parents):,}")

    if len(eligible):
        print("\nCandidate OSM links by facility:")
        print(eligible["facility_type"].value_counts().to_string())
        print("\nCandidate OSM links by proposed NCTCOG FUNCL:")
        print(eligible["FUNCL"].value_counts().sort_index().to_string())

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {REVERSE_CANDIDATES}")

def apply_reverse_pairs():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    links = pd.read_parquet(CURRENT_LINKS).set_index("link_key")
    pairs = pd.read_parquet(REVERSE_CANDIDATES)
    pairs = pairs.loc[pairs["reverse_geometry_ok"].eq(True)].copy()

    if not links.index.is_unique or pairs["link_key"].duplicated().any():
        raise ValueError("Duplicate link keys.")
    if not pairs["link_key"].isin(links.index).all():
        raise ValueError("A target link is missing.")
    if not pairs["source_link_key"].isin(links.index).all():
        raise ValueError("A supporting source link is missing.")
    if pairs["proposed_nctcog_key"].isna().any():
        raise ValueError("Missing proposed NCTCOG direction.")

    targets = links.loc[pairs["link_key"]].reset_index(drop=True)
    sources = links.loc[pairs["source_link_key"]].reset_index(drop=True)
    pairs = pairs.reset_index(drop=True)

    if targets["assigned_nctcog_key"].notna().any():
        raise ValueError("A repair would overwrite an existing assignment.")

    reliable = (
        sources["assignment_method"].eq("direct_spatial_dominant")
        & sources["consolidation_status"].eq("single_parent")
        & sources["ambiguous_length_m"].eq(0)
        & np.isclose(
            sources["matched_length_m"], sources["total_length_m"],
            rtol=1e-9, atol=1e-6,
        )
    )
    same_assignment = sources["assigned_nctcog_key"].eq(
        pairs["source_nctcog_key"]
    ).fillna(False)
    reversed_nodes = (
        sources["from_node_id"].astype("string")
        .eq(targets["to_node_id"].astype("string"))
        & sources["to_node_id"].astype("string")
        .eq(targets["from_node_id"].astype("string"))
    )
    if not reliable.all() or not same_assignment.all() or not reversed_nodes.all():
        raise ValueError("Supporting assignments or reverse endpoints changed.")

    nct = pd.read_csv(
        NCTCOG_INPUT,
        usecols=["ID", "Dir", "nctcog_direction_key", "nctcog_direction"],
        dtype={"ID": "string", "nctcog_direction_key": "string"},
    ).set_index("nctcog_direction_key")

    for column in ["source_nctcog_key", "proposed_nctcog_key"]:
        if not pairs[column].isin(nct.index).all():
            raise ValueError("A referenced NCTCOG direction is missing.")

    source_nct = nct.loc[pairs["source_nctcog_key"]].reset_index(drop=True)
    target_nct = nct.loc[pairs["proposed_nctcog_key"]].reset_index(drop=True)
    opposite = (
        source_nct["ID"].eq(target_nct["ID"])
        & source_nct["Dir"].eq(0)
        & target_nct["Dir"].eq(0)
        & (
            (source_nct["nctcog_direction"].eq("AB")
             & target_nct["nctcog_direction"].eq("BA"))
            | (source_nct["nctcog_direction"].eq("BA")
               & target_nct["nctcog_direction"].eq("AB"))
        )
    )
    if not opposite.all():
        raise ValueError("Proposed records are not valid opposite directions.")

    before = links["assigned_nctcog_key"].notna().sum()
    repairs = pairs.set_index("link_key")

    links.loc[repairs.index, "assigned_nctcog_key"] = (
        repairs["proposed_nctcog_key"].astype("string")
    )
    links.loc[repairs.index, "assignment_method"] = "reverse_pair_inferred"
    links["reverse_pair_repaired"] = links.index.isin(repairs.index)
    links["repair_reverse_source_link"] = (
        repairs["source_link_key"].reindex(links.index).astype("string")
    )

    after = links["assigned_nctcog_key"].notna().sum()
    if after != before + len(repairs):
        raise ValueError("Unexpected assignment count.")

    output = links.reset_index()
    temporary = REVERSE_REPAIRED_OUTPUT.with_suffix(".partial.parquet")
    output.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(REVERSE_REPAIRED_OUTPUT)

    print("\nREVERSE-PAIR REPAIRS COMPLETE")
    print(f"New assignments: {len(repairs):,}")
    print(output["assignment_method"].value_counts().to_string())

    for label, group in [
        ("All", output),
        *list(output.groupby("facility_type", sort=True)),
    ]:
        assigned = group["assigned_nctcog_key"].notna().sum()
        print(
            f"{label}: {assigned:,}/{len(group):,} assigned "
            f"({100 * assigned / len(group):.2f}%)"
        )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {REVERSE_REPAIRED_OUTPUT}")


def junction_side_support(gaps, links, incoming):
    if incoming:
        gap_node, gap_other = "from_node_id", "to_node_id"
        neighbor_node, neighbor_other = "to_node_id", "from_node_id"
    else:
        gap_node, gap_other = "to_node_id", "from_node_id"
        neighbor_node, neighbor_other = "from_node_id", "to_node_id"

    left = gaps[["link_key", gap_node, gap_other]].rename(columns={
        "link_key": "gap_key",
        gap_node: "join_node",
        gap_other: "gap_other",
    })
    right = links[
        ["link_key", neighbor_node, neighbor_other, "anchor_key"]
    ].rename(columns={
        "link_key": "neighbor_key",
        neighbor_node: "join_node",
        neighbor_other: "neighbor_other",
    })

    neighbors = left.merge(right, on="join_node", how="inner")
    neighbors = neighbors.loc[
        neighbors["gap_key"].ne(neighbors["neighbor_key"])
        & neighbors["gap_other"].ne(neighbors["neighbor_other"])
    ]

    stats = neighbors.groupby("gap_key", sort=False).agg(
        neighbor_count=("neighbor_key", "size"),
        reliable_count=("anchor_key", "count"),
        parent_count=("anchor_key", "nunique"),
        parent_key=("anchor_key", "first"),
    )

    stats = stats.reindex(pd.Index(gaps["link_key"], name="link_key"))
    for column in ["neighbor_count", "reliable_count", "parent_count"]:
        stats[column] = stats[column].fillna(0).astype("int32")

    prefix = "upstream_" if incoming else "downstream_"
    return stats.add_prefix(prefix)

def evaluate_junctions():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    columns = [
        "link_key", "from_node_id", "to_node_id", "facility_type",
        "total_length_m", "matched_length_m", "ambiguous_length_m",
        "consolidation_status", "assigned_nctcog_key", "assignment_method",
    ]
    links = pd.read_parquet(REVERSE_REPAIRED_OUTPUT, columns=columns)

    if not links["link_key"].is_unique:
        raise ValueError("Duplicate OSM link keys.")

    for column in ["from_node_id", "to_node_id"]:
        links[column] = links[column].astype("string")
        if links[column].isna().any() or links[column].str.strip().eq("").any():
            raise ValueError(f"Missing node identifiers: {column}")

    reliable = (
        links["assignment_method"].eq("direct_spatial_dominant")
        & links["consolidation_status"].eq("single_parent")
        & links["ambiguous_length_m"].eq(0)
        & links["assigned_nctcog_key"].notna()
        & np.isclose(
            links["matched_length_m"], links["total_length_m"],
            rtol=1e-9, atol=1e-6,
        )
    )
    links["anchor_key"] = links["assigned_nctcog_key"].where(reliable)
    gaps = links.loc[links["assigned_nctcog_key"].isna()].copy()

    print("Checking incoming junction support...", flush=True)
    upstream = junction_side_support(gaps, links, incoming=True)
    print("Checking outgoing junction support...", flush=True)
    downstream = junction_side_support(gaps, links, incoming=False)

    evaluated = (
        gaps.set_index("link_key")
        .join(upstream, validate="one_to_one")
        .join(downstream, validate="one_to_one")
    )

    at_branch = (
        evaluated["upstream_neighbor_count"].gt(1)
        | evaluated["downstream_neighbor_count"].gt(1)
    )
    both_sides = (
        evaluated["upstream_neighbor_count"].gt(0)
        & evaluated["downstream_neighbor_count"].gt(0)
    )
    all_reliable = (
        both_sides
        & evaluated["upstream_reliable_count"].eq(
            evaluated["upstream_neighbor_count"]
        )
        & evaluated["downstream_reliable_count"].eq(
            evaluated["downstream_neighbor_count"]
        )
    )
    agreement = (
        evaluated["upstream_parent_count"].eq(1)
        & evaluated["downstream_parent_count"].eq(1)
        & evaluated["upstream_parent_key"].eq(
            evaluated["downstream_parent_key"]
        ).fillna(False)
    )

    eligible = at_branch & all_reliable & agreement
    candidates = evaluated.loc[eligible].copy()
    candidates["proposed_nctcog_key"] = candidates["upstream_parent_key"]
    candidates["repair_status"] = "candidate_not_applied"

    keep = [
        "from_node_id", "to_node_id", "facility_type", "total_length_m",
        "upstream_neighbor_count", "downstream_neighbor_count",
        "proposed_nctcog_key", "repair_status",
    ]
    candidates = candidates[keep].reset_index()

    temporary = JUNCTION_CANDIDATES.with_suffix(".partial.parquet")
    candidates.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(JUNCTION_CANDIDATES)

    print("\nJUNCTION EVALUATION")
    print(f"Unassigned links touching a branch: {at_branch.sum():,}")
    print(f"With neighbors on both sides: {(at_branch & both_sides).sum():,}")
    print(f"All neighbors reliably assigned: {(at_branch & all_reliable).sum():,}")
    print(f"Same-parent junction candidates: {len(candidates):,}")
    print(
        "Reliable but conflicting parent evidence: "
        f"{(at_branch & all_reliable & ~agreement).sum():,}"
    )

    if len(candidates):
        print("\nCandidates by facility:")
        print(candidates["facility_type"].value_counts().to_string())
        print("\nCandidate lengths, metres:")
        print(
            candidates["total_length_m"]
            .describe(percentiles=[0.5, 0.9, 0.99]).to_string()
        )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {JUNCTION_CANDIDATES}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["reverse_candidates", "reverse_apply", "junction_candidates"],
    )
    args = parser.parse_args()

    if args.stage == "reverse_candidates":
        evaluate_reverse_pairs()
    elif args.stage == "reverse_apply":
        apply_reverse_pairs()
    elif args.stage == "junction_candidates":
        evaluate_junctions()