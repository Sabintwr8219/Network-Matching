import argparse
import time

import numpy as np
import pandas as pd
import psutil

from paths import INTERMEDIATE_DIR
import shapely
from pyproj import Transformer
from paths import OSM_INPUT


SUMMARY_INPUT = INTERMEDIATE_DIR / "OSM_link_match_summary.parquet"
SINGLE_GAP_OUTPUT = INTERMEDIATE_DIR / "OSM_single_gap_candidates.parquet"

SINGLE_REPAIRED_OUTPUT = (
    INTERMEDIATE_DIR / "OSM_link_matches_after_single_gap.parquet"
)

MULTI_GAP_OUTPUT = (
    INTERMEDIATE_DIR / "OSM_multi_gap_candidates.parquet"
)

MULTI_REPAIRED_OUTPUT = (
    INTERMEDIATE_DIR / "OSM_link_matches_after_multi_gap.parquet"
)

def find_gap_neighbors(gaps, links, incoming):
    if incoming:
        gap_node, other_gap_node = "from_node_id", "to_node_id"
        neighbor_node, other_neighbor_node = "to_node_id", "from_node_id"
    else:
        gap_node, other_gap_node = "to_node_id", "from_node_id"
        neighbor_node, other_neighbor_node = "from_node_id", "to_node_id"

    left = gaps[["link_key", gap_node, other_gap_node]].rename(columns={
        "link_key": "gap_key",
        gap_node: "join_node",
        other_gap_node: "other_gap_node",
    })
    right = links[
        ["link_key", neighbor_node, other_neighbor_node, "anchor_key"]
    ].rename(columns={
        "link_key": "neighbor_key",
        neighbor_node: "join_node",
        other_neighbor_node: "other_neighbor_node",
    })

    neighbors = left.merge(right, on="join_node", how="inner")

    # Exclude the same link and immediate return along the gap's endpoints.
    neighbors = neighbors.loc[
        neighbors["gap_key"].ne(neighbors["neighbor_key"])
        & neighbors["other_gap_node"].ne(neighbors["other_neighbor_node"])
    ]

    counts = neighbors.groupby("gap_key", sort=False).size()
    unique = neighbors.loc[
        neighbors["gap_key"].map(counts).eq(1)
    ].set_index("gap_key")

    output = pd.DataFrame(index=pd.Index(gaps["link_key"], name="link_key"))
    output["neighbor_count"] = counts.reindex(output.index).fillna(0).astype("int32")
    output["neighbor_key"] = unique["neighbor_key"].reindex(output.index)
    output["anchor_key"] = unique["anchor_key"].reindex(output.index)

    prefix = "upstream_" if incoming else "downstream_"
    return output.add_prefix(prefix)

def evaluate_single_gaps():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    links = pd.read_parquet(SUMMARY_INPUT)
    node_columns = ["from_node_id", "to_node_id"]

    for column in node_columns:
        links[column] = links[column].astype("string")
        if links[column].isna().any() or links[column].str.strip().eq("").any():
            raise ValueError(f"Missing node identifiers in {column}.")

    # Use fully supported, unambiguous direct matches as anchors.
    anchor = (
        links["consolidation_status"].eq("single_parent")
        & np.isclose(
            links["matched_length_m"], links["total_length_m"],
            rtol=1e-9, atol=1e-6,
        )
        & links["ambiguous_length_m"].eq(0)
        & links["dominant_nctcog_key"].notna()
    )
    links["anchor_key"] = links["dominant_nctcog_key"].where(anchor)

    gaps = links.loc[links["consolidation_status"].eq("unmatched")].copy()

    print(f"Reliable supporting links: {anchor.sum():,}", flush=True)
    print(f"Unmatched links to evaluate: {len(gaps):,}", flush=True)

    print("Finding upstream neighbors...", flush=True)
    upstream = find_gap_neighbors(gaps, links, incoming=True)

    print("Finding downstream neighbors...", flush=True)
    downstream = find_gap_neighbors(gaps, links, incoming=False)

    evaluated = (
        gaps.set_index("link_key")
        .join(upstream, validate="one_to_one")
        .join(downstream, validate="one_to_one")
    )

    unique_path = (
        evaluated["upstream_neighbor_count"].eq(1)
        & evaluated["downstream_neighbor_count"].eq(1)
    )
    anchored = (
        unique_path
        & evaluated["upstream_anchor_key"].notna()
        & evaluated["downstream_anchor_key"].notna()
    )
    same_parent = (
        anchored
        & evaluated["upstream_anchor_key"]
        .eq(evaluated["downstream_anchor_key"]).fillna(False)
    )

    candidates = evaluated.loc[same_parent, [
        "from_node_id", "to_node_id", "facility_type", "total_length_m",
        "upstream_neighbor_key", "downstream_neighbor_key",
        "upstream_anchor_key",
    ]].rename(columns={
        "upstream_anchor_key": "proposed_nctcog_key",
    }).reset_index()

    candidates["repair_status"] = "candidate_not_applied"
    temporary = SINGLE_GAP_OUTPUT.with_suffix(".partial.parquet")
    candidates.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(SINGLE_GAP_OUTPUT)

    print("\nSINGLE-GAP EVALUATION")
    print(f"Unmatched links: {len(gaps):,}")
    print(f"Unique continuation on both sides: {unique_path.sum():,}")
    print(f"Reliable anchors on both sides: {anchored.sum():,}")
    print(f"Same-parent candidates: {len(candidates):,}")
    print(f"Different-parent anchored gaps: {(anchored & ~same_parent).sum():,}")

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
    print(f"Saved: {SINGLE_GAP_OUTPUT}")

def apply_single_gap_repairs():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    links = pd.read_parquet(SUMMARY_INPUT).set_index("link_key")
    candidates = pd.read_parquet(SINGLE_GAP_OUTPUT).set_index("link_key")

    if not links.index.is_unique or not candidates.index.is_unique:
        raise ValueError("Duplicate OSM link keys.")
    if not candidates.index.isin(links.index).all():
        raise ValueError("Candidate links are absent from the summary.")
    if candidates["proposed_nctcog_key"].isna().any():
        raise ValueError("Candidate has no proposed NCTCOG parent.")
    if not links.loc[
        candidates.index, "consolidation_status"
    ].eq("unmatched").all():
        raise ValueError("A repair would overwrite an existing direct match.")

    # Confirm that both recorded neighbors still support the proposed key.
    for column in ["upstream_neighbor_key", "downstream_neighbor_key"]:
        neighbor_keys = candidates[column]
        if not neighbor_keys.isin(links.index).all():
            raise ValueError("A supporting neighbor is missing.")

        neighbors = links.loc[neighbor_keys.to_numpy()]
        same_parent = (
            neighbors["dominant_nctcog_key"].to_numpy()
            == candidates["proposed_nctcog_key"].to_numpy()
        )
        reliable = (
            neighbors["consolidation_status"].eq("single_parent")
            & neighbors["ambiguous_length_m"].eq(0)
            & np.isclose(
                neighbors["matched_length_m"],
                neighbors["total_length_m"],
                rtol=1e-9,
                atol=1e-6,
            )
        )
        if not same_parent.all() or not reliable.all():
            raise ValueError("Supporting assignments changed; reevaluate gaps.")

    links["assigned_nctcog_key"] = links["dominant_nctcog_key"].astype("string")
    direct = links["assigned_nctcog_key"].notna()

    links["assignment_method"] = np.where(
        direct, "direct_spatial_dominant", "unmatched"
    )
    links["gap_repaired"] = False

    links.loc[candidates.index, "assigned_nctcog_key"] = (
        candidates["proposed_nctcog_key"].astype("string")
    )
    links.loc[candidates.index, "assignment_method"] = "single_gap_topology"
    links.loc[candidates.index, "gap_repaired"] = True

    # Keep the supporting OSM links with each repair.
    links["repair_upstream_link"] = candidates[
        "upstream_neighbor_key"
    ].reindex(links.index).astype("string")
    links["repair_downstream_link"] = candidates[
        "downstream_neighbor_key"
    ].reindex(links.index).astype("string")

    assigned = links["assigned_nctcog_key"].notna()
    if assigned.sum() != direct.sum() + len(candidates):
        raise ValueError("Assignment count changed unexpectedly.")

    output = links.reset_index()
    temporary = SINGLE_REPAIRED_OUTPUT.with_suffix(".partial.parquet")
    output.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(SINGLE_REPAIRED_OUTPUT)

    print("\nSINGLE-GAP REPAIRS COMPLETE")
    print(f"Directly supported links: {direct.sum():,}")
    print(f"Topology-inferred repairs: {len(candidates):,}")
    print(f"Links with an assignment: {assigned.sum():,}")
    print(f"Remaining unassigned links: {(~assigned).sum():,}")

    for label, group in [
        ("All", output),
        *list(output.groupby("facility_type", sort=True)),
    ]:
        count = group["assigned_nctcog_key"].notna().sum()
        direct_length_pct = (
            100 * group["matched_length_m"].sum()
            / group["total_length_m"].sum()
        )
        print(
            f"{label}: {count:,}/{len(group):,} assigned "
            f"({100 * count / len(group):.2f}%) | "
            f"Directly matched length: {direct_length_pct:.2f}%"
        )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {SINGLE_REPAIRED_OUTPUT}")

def evaluate_multi_gaps():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    columns = [
        "link_key", "from_node_id", "to_node_id", "facility_type",
        "total_length_m", "matched_length_m", "ambiguous_length_m",
        "consolidation_status", "dominant_nctcog_key",
        "assigned_nctcog_key", "assignment_method",
    ]
    links = pd.read_parquet(SINGLE_REPAIRED_OUTPUT, columns=columns)

    if not links["link_key"].is_unique:
        raise ValueError("Duplicate OSM link keys.")
    for column in ["from_node_id", "to_node_id"]:
        links[column] = links[column].astype("string")
        if links[column].isna().any() or links[column].str.strip().eq("").any():
            raise ValueError(f"Missing node identifiers: {column}")

    anchor = (
        links["assignment_method"].eq("direct_spatial_dominant")
        & links["consolidation_status"].eq("single_parent")
        & links["ambiguous_length_m"].eq(0)
        & links["dominant_nctcog_key"].notna()
        & np.isclose(
            links["matched_length_m"], links["total_length_m"],
            rtol=1e-9, atol=1e-6,
        )
    )
    links["anchor_key"] = links["dominant_nctcog_key"].where(anchor)
    gaps = links.loc[links["assigned_nctcog_key"].isna()].copy()

    print(f"Remaining unmatched links: {len(gaps):,}", flush=True)
    print("Finding upstream neighbors...", flush=True)
    upstream = find_gap_neighbors(gaps, links, incoming=True)
    print("Finding downstream neighbors...", flush=True)
    downstream = find_gap_neighbors(gaps, links, incoming=False)

    evaluated = (
        gaps.set_index("link_key")
        .join(upstream, validate="one_to_one")
        .join(downstream, validate="one_to_one")
    )

    keys = evaluated.index.to_numpy()
    upstream_keys = evaluated["upstream_neighbor_key"].to_numpy()
    downstream_keys = evaluated["downstream_neighbor_key"].to_numpy()
    upstream_anchors = evaluated["upstream_anchor_key"].to_numpy()
    downstream_anchors = evaluated["downstream_anchor_key"].to_numpy()
    lengths = evaluated["total_length_m"].to_numpy()
    facilities = evaluated["facility_type"].to_numpy()

    unique_path = (
        evaluated["upstream_neighbor_count"].eq(1)
        & evaluated["downstream_neighbor_count"].eq(1)
    ).to_numpy()

    # -1 means the next neighbor is not an unmatched link.
    next_gap = evaluated.index.get_indexer(
        evaluated["downstream_neighbor_key"].fillna("")
    )
    starts = np.flatnonzero(
        unique_path & evaluated["upstream_anchor_key"].notna().to_numpy()
    )

    print(f"Tracing from {len(starts):,} anchored starts...", flush=True)
    records = []
    chain_count = 0
    stopped = {}

    for start in starts:
        current = int(start)
        previous_key = upstream_keys[start]
        proposed_parent = upstream_anchors[start]
        chain = []
        visited = set()

        while True:
            if current in visited:
                reason = "cycle"
                break
            if not unique_path[current]:
                reason = "branch_or_dead_end"
                break
            if upstream_keys[current] != previous_key:
                reason = "inconsistent_predecessor"
                break

            visited.add(current)
            chain.append(current)
            following = int(next_gap[current])

            if following >= 0:
                previous_key = keys[current]
                current = following
                continue

            ending_parent = downstream_anchors[current]
            if pd.isna(ending_parent):
                reason = "no_reliable_downstream_anchor"
            elif ending_parent != proposed_parent:
                reason = "different_parent_endpoints"
            elif len(chain) < 2:
                reason = "single_link_only"
            else:
                reason = "candidate"
                chain_count += 1
                chain_length = float(lengths[chain].sum())

                for position, row in enumerate(chain):
                    records.append({
                        "chain_id": chain_count,
                        "chain_position": position,
                        "link_key": keys[row],
                        "facility_type": facilities[row],
                        "link_length_m": lengths[row],
                        "chain_link_count": len(chain),
                        "chain_length_m": chain_length,
                        "proposed_nctcog_key": proposed_parent,
                        "upstream_anchor_link": upstream_keys[start],
                        "downstream_anchor_link": downstream_keys[current],
                        "repair_status": "candidate_not_applied",
                    })
            break

        stopped[reason] = stopped.get(reason, 0) + 1

    output_columns = [
        "chain_id", "chain_position", "link_key", "facility_type",
        "link_length_m", "chain_link_count", "chain_length_m",
        "proposed_nctcog_key", "upstream_anchor_link",
        "downstream_anchor_link", "repair_status",
    ]
    candidates = pd.DataFrame(records, columns=output_columns)

    if candidates["link_key"].duplicated().any():
        raise ValueError("An unmatched link belongs to competing chains.")

    temporary = MULTI_GAP_OUTPUT.with_suffix(".partial.parquet")
    candidates.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(MULTI_GAP_OUTPUT)

    print("\nMULTI-GAP EVALUATION")
    print(f"Candidate chains: {chain_count:,}")
    print(f"Candidate links: {len(candidates):,}")
    print("\nTrace outcomes:")
    print(pd.Series(stopped, dtype="int64").to_string())

    if len(candidates):
        chains = candidates.drop_duplicates("chain_id")
        print("\nCandidate links by facility:")
        print(candidates["facility_type"].value_counts().to_string())
        print("\nChain sizes and lengths:")
        print(
            chains[["chain_link_count", "chain_length_m"]]
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
    print(f"Saved: {MULTI_GAP_OUTPUT}")


def check_multi_gap_geometry():
    started = time.perf_counter()
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])

    candidates = pd.read_parquet(MULTI_GAP_OUTPUT)
    if candidates.empty:
        print("No multi-gap candidates to check.")
        return

    needed_keys = set(candidates["link_key"])
    selected = []

    print("Reading candidate OSM geometries...", flush=True)
    for chunk in pd.read_csv(
        OSM_INPUT,
        usecols=["link_key", "geometry"],
        dtype={"link_key": "string"},
        chunksize=100_000,
    ):
        keep = chunk["link_key"].isin(needed_keys)
        if keep.any():
            selected.append(chunk.loc[keep])

    osm = pd.concat(selected, ignore_index=True).set_index("link_key")
    if not osm.index.is_unique or set(osm.index) != needed_keys:
        raise ValueError("Candidate OSM geometries are missing or duplicated.")

    nctcog = pd.read_csv(
        INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv",
        usecols=["nctcog_direction_key", "travel_geometry"],
        dtype={"nctcog_direction_key": "string"},
    ).set_index("nctcog_direction_key")

    parent_keys = candidates["proposed_nctcog_key"].unique()
    if not pd.Index(parent_keys).isin(nctcog.index).all():
        raise ValueError("A proposed NCTCOG parent is missing.")

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32614", always_xy=True
    )

    def project_wkt(values):
        return shapely.transform(
            shapely.from_wkt(values),
            transformer.transform,
            interleaved=False,
        )

    osm_geometry = pd.Series(
        project_wkt(osm["geometry"].to_numpy()), index=osm.index
    )
    parent_geometry = pd.Series(
        project_wkt(nctcog.loc[parent_keys, "travel_geometry"].to_numpy()),
        index=parent_keys,
    )

    print("Checking travel progression along proposed parents...", flush=True)
    records = []

    for chain_id, group in candidates.groupby("chain_id", sort=False):
        group = group.sort_values("chain_position")
        if group["proposed_nctcog_key"].nunique() != 1:
            raise ValueError("A chain has conflicting proposed parents.")

        parent_key = group["proposed_nctcog_key"].iloc[0]
        parent = parent_geometry.loc[parent_key]
        lines = osm_geometry.loc[group["link_key"]].to_numpy()

        # OSM short links have two vertices; retain travel order.
        coordinates = shapely.get_coordinates(lines)
        points = shapely.points(coordinates)
        positions = shapely.line_locate_point(parent, points)
        steps = np.diff(positions)
        offsets = shapely.distance(points, parent)

        records.append({
            "chain_id": chain_id,
            "proposed_nctcog_key": parent_key,
            "chain_link_count": len(group),
            "chain_length_m": group["link_length_m"].sum(),
            "parent_length_m": parent.length,
            "projected_progress_m": positions[-1] - positions[0],
            "backward_steps": int((steps < -1e-6).sum()),
            "max_endpoint_offset_m": offsets.max(),
        })

    checks = pd.DataFrame(records)
    checks["needs_progression_review"] = (
        checks["backward_steps"].gt(0)
        | checks["projected_progress_m"].le(0)
    )

    destination = INTERMEDIATE_DIR / "OSM_multi_gap_geometry_checks.parquet"
    temporary = destination.with_suffix(".partial.parquet")
    checks.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(destination)

    print("\nMULTI-GAP GEOMETRY CHECK")
    print(f"Chains checked: {len(checks):,}")
    print(
        "Chains needing progression review: "
        f"{checks['needs_progression_review'].sum():,}"
    )
    print("\nTen longest candidate chains:")
    print(
        checks.nlargest(10, "chain_length_m").to_string(index=False)
    )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {destination}")

def apply_multi_gap_repairs():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    links = pd.read_parquet(SINGLE_REPAIRED_OUTPUT).set_index("link_key")
    candidates = pd.read_parquet(MULTI_GAP_OUTPUT)
    checks = pd.read_parquet(
        INTERMEDIATE_DIR / "OSM_multi_gap_geometry_checks.parquet"
    )

    if not links.index.is_unique or not checks["chain_id"].is_unique:
        raise ValueError("Duplicate link keys or chain-check IDs.")
    if candidates["link_key"].duplicated().any():
        raise ValueError("A link appears in multiple candidate chains.")
    if set(candidates["chain_id"]) != set(checks["chain_id"]):
        raise ValueError("Candidates and geometry checks do not agree.")
    if checks["needs_progression_review"].isna().any():
        raise ValueError("Missing geometry-check results.")

    # Verify that geometry checks describe the current candidate chains.
    actual = candidates.groupby("chain_id").agg(
        parent_count=("proposed_nctcog_key", "nunique"),
        proposed_nctcog_key=("proposed_nctcog_key", "first"),
        chain_link_count=("link_key", "size"),
        chain_length_m=("link_length_m", "sum"),
    )
    recorded = checks.set_index("chain_id").reindex(actual.index)

    if not actual["parent_count"].eq(1).all():
        raise ValueError("Conflicting parents within a chain.")
    for column in ["proposed_nctcog_key", "chain_link_count"]:
        if not actual[column].eq(recorded[column]).all():
            raise ValueError(f"Geometry checks differ on {column}.")
    if not np.allclose(
        actual["chain_length_m"], recorded["chain_length_m"],
        rtol=1e-9, atol=1e-6,
    ):
        raise ValueError("Chain lengths differ from geometry checks.")

    accepted_ids = checks.loc[
        ~checks["needs_progression_review"], "chain_id"
    ]
    accepted = candidates.loc[
        candidates["chain_id"].isin(accepted_ids)
    ].set_index("link_key")

    if not accepted.index.isin(links.index).all():
        raise ValueError("Candidate links are absent from the current network.")
    if links.loc[accepted.index, "assigned_nctcog_key"].notna().any():
        raise ValueError("A repair would overwrite an existing assignment.")

    for column in ["upstream_anchor_link", "downstream_anchor_link"]:
        anchor_keys = accepted[column]
        if not anchor_keys.isin(links.index).all():
            raise ValueError("Supporting anchor is missing.")

        anchors = links.loc[anchor_keys.to_numpy()]
        reliable = (
            anchors["assignment_method"].eq("direct_spatial_dominant")
            & anchors["consolidation_status"].eq("single_parent")
            & anchors["ambiguous_length_m"].eq(0)
            & np.isclose(
                anchors["matched_length_m"], anchors["total_length_m"],
                rtol=1e-9, atol=1e-6,
            )
        )
        same_parent = (
            anchors["assigned_nctcog_key"].astype("string")
            .reset_index(drop=True)
            .eq(
                accepted["proposed_nctcog_key"].astype("string")
                .reset_index(drop=True)
            ).fillna(False)
        )
        if not reliable.all() or not same_parent.all():
            raise ValueError("Supporting anchor assignments have changed.")

    before = links["assigned_nctcog_key"].notna().sum()

    links.loc[accepted.index, "assigned_nctcog_key"] = (
        accepted["proposed_nctcog_key"].astype("string")
    )
    links.loc[accepted.index, "assignment_method"] = "multi_gap_topology"
    links.loc[accepted.index, "gap_repaired"] = True
    links.loc[accepted.index, "repair_upstream_link"] = (
        accepted["upstream_anchor_link"].astype("string")
    )
    links.loc[accepted.index, "repair_downstream_link"] = (
        accepted["downstream_anchor_link"].astype("string")
    )
    links["repair_chain_id"] = (
        accepted["chain_id"].reindex(links.index).astype("Int64")
    )

    after = links["assigned_nctcog_key"].notna().sum()
    if after != before + len(accepted):
        raise ValueError("Assignment count changed unexpectedly.")

    output = links.reset_index()
    temporary = MULTI_REPAIRED_OUTPUT.with_suffix(".partial.parquet")
    output.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(MULTI_REPAIRED_OUTPUT)

    print("\nMULTI-GAP REPAIRS COMPLETE")
    print(f"Applied chains: {len(accepted_ids):,}")
    print(f"Newly assigned links: {len(accepted):,}")
    print(f"Deferred chains: {checks['needs_progression_review'].sum():,}")
    print(f"Deferred candidate links: {len(candidates) - len(accepted):,}")
    print("\nAssignment methods:")
    print(output["assignment_method"].value_counts().to_string())

    for label, group in [
        ("All", output),
        *list(output.groupby("facility_type", sort=True)),
    ]:
        assigned = group["assigned_nctcog_key"].notna().sum()
        direct_length_pct = (
            100 * group["matched_length_m"].sum()
            / group["total_length_m"].sum()
        )
        print(
            f"{label}: {assigned:,}/{len(group):,} assigned "
            f"({100 * assigned / len(group):.2f}%) | "
            f"Directly matched length: {direct_length_pct:.2f}%"
        )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {MULTI_REPAIRED_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and apply conservative topology gap repairs."
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "single_candidates",
            "single_apply",
            "multi_candidates",
            "multi_geometry",
            "multi_apply",
            "all",
        ],
    )
    args = parser.parse_args()

    actions = {
        "single_candidates": evaluate_single_gaps,
        "single_apply": apply_single_gap_repairs,
        "multi_candidates": evaluate_multi_gaps,
        "multi_geometry": check_multi_gap_geometry,
        "multi_apply": apply_multi_gap_repairs,
    }

    if args.stage == "all":
        for action in actions.values():
            action()
    else:
        actions[args.stage]()


if __name__ == "__main__":
    main()
