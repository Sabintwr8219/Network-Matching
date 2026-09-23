import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

from paths import INTERMEDIATE_DIR, MOTORWAY_DIR, TRUNK_DIR


MOTORWAY_INPUT = MOTORWAY_DIR / "matched_network_basic_motorway.csv"
TRUNK_INPUT = TRUNK_DIR / "matched_network_basic_trunk.csv"

MOTORWAY_DOMINANT = MOTORWAY_DIR / "matched_network_basic_motorway_with_dominant_gid.csv"
TRUNK_DOMINANT = TRUNK_DIR / "matched_network_basic_trunk_with_dominant_gid.csv"

MOTORWAY_SHORT = MOTORWAY_DIR / "matched_network_basic_motorway_short_link.csv"
TRUNK_SHORT = TRUNK_DIR / "matched_network_basic_trunk_short_link.csv"
FREEWAY_OUTPUT = INTERMEDIATE_DIR / "matched_network_basic_short_link.csv"


# Assign one dominant GID to every 10-meter segment of an OSM way.
def assign_dominant_gid(group):
    gid_counts = group["Matched_GID"].value_counts(dropna=False)
    top_count = gid_counts.max()
    top_gids = gid_counts[gid_counts == top_count].index.tolist()

    if len(top_gids) == 1:
        dominant_gid = top_gids[0]
    else:
        length_sums = {}

        for gid in top_gids:
            if pd.isna(gid):
                total = group.loc[group["Matched_GID"].isna(), "segment_length_m"].sum()
            else:
                total = group.loc[group["Matched_GID"] == gid, "segment_length_m"].sum()

            length_sums[gid] = total

        dominant_gid = max(length_sums, key=length_sums.get)

    group["Matched_GID"] = dominant_gid
    return group


# Synchronize flipped trunk segments using the first non-null GID of each physical node pair.
def synchronize_trunk_pairs(df):
    node_a = np.minimum(df["from_node_id"].to_numpy(), df["to_node_id"].to_numpy())
    node_b = np.maximum(df["from_node_id"].to_numpy(), df["to_node_id"].to_numpy())

    df = df.copy()
    df["pair_a"] = node_a
    df["pair_b"] = node_b

    synced = (
        df.groupby(["pair_a", "pair_b"], observed=True, sort=True)["Matched_GID"]
        .first()
        .rename("sync_gid")
        .reset_index()
    )

    df = df.drop(columns=["Matched_GID"]).merge(
        synced,
        on=["pair_a", "pair_b"],
        how="left",
        sort=False,
    )

    df = df.rename(columns={"sync_gid": "Matched_GID"})
    return df.drop(columns=["pair_a", "pair_b"])


# Apply the dominant-GID rule to one segmented freeway dataset.
def apply_dominant_gid(df):
    return (
        df.groupby("osm_way_id", group_keys=True, sort=True)
        .apply(assign_dominant_gid, include_groups=False)
        .reset_index(level=0)
    )


# Rebuild one original OSM short link from its ordered 10-meter pieces.
def restore_short_links(df, facility_type):
    df = df.copy()
    df["segment_order"] = pd.to_numeric(
        df["segment_id"].astype(str).str.extract(r"_(\d+)$")[0],
        errors="coerce",
    ).astype("Int64")

    df = df.sort_values(["link_id", "segment_order"], kind="stable")

    keep_columns = [
        "link_id",
        "name",
        "osm_way_id",
        "from_node_id",
        "to_node_id",
        "directed",
        "geometry",
        "dir_flag",
        "length",
        "facility_type",
        "link_type",
        "free_speed",
        "lanes",
        "capacity",
        "allowed_uses",
        "notes",
        "heading",
        "directional",
        "Matched_GID",
        "osm_short_a",
        "osm_short_b",
    ]

    first_rows = df.drop_duplicates("link_id", keep="first").set_index("link_id")
    last_rows = df.drop_duplicates("link_id", keep="last").set_index("link_id")

    rows = []

    for link_id, first_row in first_rows.iterrows():
        first_geometry = wkt.loads(first_row["geometry"])
        last_geometry = wkt.loads(last_rows.loc[link_id, "geometry"])
        new_geometry = LineString([first_geometry.coords[0], last_geometry.coords[-1]])

        row = first_row[keep_columns[1:]].copy()
        row["link_id"] = link_id
        row["geometry"] = new_geometry.wkt
        rows.append(row)

    output = pd.DataFrame(rows)[keep_columns]
    facility = output[output["facility_type"].astype("string").str.lower() == facility_type].copy()

    if facility_type == "motorway":
        topology_source = facility[facility["directional"] == False]
    else:
        topology_source = facility

    from_counts = topology_source["from_node_id"].value_counts()
    to_counts = topology_source["to_node_id"].value_counts()

    diverging_nodes = set(from_counts[from_counts == 2].index)
    converging_nodes = set(to_counts[to_counts == 2].index)

    def get_merge_type(row):
        is_diverging = row["from_node_id"] in diverging_nodes
        is_converging = row["to_node_id"] in converging_nodes

        if bool(row["directional"]):
            return "None"
        if is_diverging and is_converging:
            return "Diverging-Converging"
        if is_diverging:
            return "Diverging"
        if is_converging:
            return "Converging"
        return "None"

    output["merge"] = output.apply(get_merge_type, axis=1)
    output["Matched_GID"] = pd.to_numeric(output["Matched_GID"], errors="coerce").fillna(-1)

    return output


# Run dominant-GID assignment and restore motorway and trunk short links.
def main():
    print("Stage 05 - Dominant GID and short-link restore")

    motorway = pd.read_csv(MOTORWAY_INPUT, low_memory=False)
    trunk = pd.read_csv(TRUNK_INPUT, low_memory=False)

    motorway = apply_dominant_gid(motorway)
    trunk = synchronize_trunk_pairs(trunk)
    trunk = apply_dominant_gid(trunk)

    motorway.to_csv(MOTORWAY_DOMINANT, index=False)
    trunk.to_csv(TRUNK_DOMINANT, index=False)

    motorway_short = restore_short_links(motorway, "motorway")
    trunk_short = restore_short_links(trunk, "trunk")

    motorway_short.to_csv(MOTORWAY_SHORT, index=False)
    trunk_short.to_csv(TRUNK_SHORT, index=False)

    combined = pd.concat([motorway_short, trunk_short], ignore_index=True)
    combined.to_csv(FREEWAY_OUTPUT, index=False)

    print(f"Motorway short links: {len(motorway_short):,}")
    print(f"Trunk short links: {len(trunk_short):,}")
    print(f"Combined freeway links: {len(combined):,}")
    print("Saved:", FREEWAY_OUTPUT)


if __name__ == "__main__":
    main()
