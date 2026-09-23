import re

import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

from paths import FINAL_DIR, INTERMEDIATE_DIR


INPUT_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT_with_Districts_Clean.csv"

OUTPUT_DIR = FINAL_DIR / "OSM_Level_CSV"
LINK_OUTPUT = OUTPUT_DIR / "Complete_OSM_Level_Link_List.csv"
NODE_OUTPUT = OUTPUT_DIR / "Complete_OSM_Level_Node_List.csv"
DISTRICT_LINK_DIR = OUTPUT_DIR / "District_Link_List"
DISTRICT_NODE_DIR = OUTPUT_DIR / "District_Node_List"


# Convert a district name into a safe CSV filename.
def clean_filename(name):
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return re.sub(r"\s+", "_", name)


# Remove only the final short-link sequence number from a link key.
def get_consolidated_link_key(link_key):
    parts = str(link_key).split("_")

    if len(parts) < 3:
        return str(link_key)

    return f"{parts[0]}_{parts[1]}"


# Extract the final short-link sequence number for ordering.
def get_link_number(link_key):
    parts = str(link_key).split("_")

    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            return 0

    return 0


# Combine ordered short-link geometries into one continuous OSM-level LineString.
def combine_ordered_lines(group):
    group = group.sort_values("segment_number")
    ordered_coordinates = []

    for row in group.itertuples():
        coordinates = list(row.geometry.coords)

        if not ordered_coordinates:
            ordered_coordinates.extend(coordinates)
            continue

        previous_end = ordered_coordinates[-1]
        current_start = coordinates[0]
        current_end = coordinates[-1]

        if previous_end == current_start:
            ordered_coordinates.extend(coordinates[1:])
        elif previous_end == current_end:
            coordinates = list(reversed(coordinates))
            ordered_coordinates.extend(coordinates[1:])
        else:
            ordered_coordinates.extend(coordinates)

    return LineString(ordered_coordinates)


# Consolidate all short links sharing one directional OSM-level key.
def consolidate_links(df):
    rows = []

    for consolidated_key, group in df.groupby("consolidated_link_key", sort=False):
        group = group.sort_values("segment_number")
        row = group.iloc[0].copy()
        row["link_key"] = consolidated_key
        row["geometry"] = combine_ordered_lines(group).wkt
        rows.append(row)

    output = pd.DataFrame(rows)
    return output.drop(columns=["consolidated_link_key", "segment_number"], errors="ignore")


# Assign global node numbers from consolidated link endpoints.
def assign_nodes(df):
    node_dict = {}
    node_rows = []
    from_nodes = []
    to_nodes = []
    node_number = 1

    for geometry_text in df["geometry"]:
        geometry = wkt.loads(geometry_text)
        start = geometry.coords[0]
        end = geometry.coords[-1]

        start_key = (round(start[0], 10), round(start[1], 10))
        end_key = (round(end[0], 10), round(end[1], 10))

        if start_key not in node_dict:
            node_dict[start_key] = node_number
            node_rows.append(
                {
                    "Node Number": node_number,
                    "Latitude": start_key[1],
                    "Longitude": start_key[0],
                }
            )
            node_number += 1

        if end_key not in node_dict:
            node_dict[end_key] = node_number
            node_rows.append(
                {
                    "Node Number": node_number,
                    "Latitude": end_key[1],
                    "Longitude": end_key[0],
                }
            )
            node_number += 1

        from_nodes.append(node_dict[start_key])
        to_nodes.append(node_dict[end_key])

    links = df.copy()
    links["from_node_id"] = from_nodes
    links["to_node_id"] = to_nodes

    return links, pd.DataFrame(node_rows)


# Reclassify consolidated facilities into final freeway and arterial link types.
def classify_final_links(df):
    df = df.copy()
    facility = df["facility_type"].astype(str).str.strip().str.lower()
    df["facility_type"] = facility.where(~facility.isin(["motorway", "trunk"]), "freeway")
    df.loc[df["facility_type"] != "freeway", "facility_type"] = "arterial"
    df["link_type"] = df["facility_type"].map({"freeway": 1, "arterial": 2})
    return df


# Save complete and district-specific consolidated OSM-level files.
def export_district_files(links, nodes):
    for district_name, district_links in links.groupby("TXDOT_DI_2", sort=True):
        district_file = clean_filename(district_name)
        used_nodes = set(district_links["from_node_id"]).union(district_links["to_node_id"])
        district_nodes = nodes[nodes["Node Number"].isin(used_nodes)].copy()

        district_links.to_csv(
            DISTRICT_LINK_DIR / f"{district_file}_Link_List.csv",
            index=False,
        )
        district_nodes.to_csv(
            DISTRICT_NODE_DIR / f"{district_file}_Node_List.csv",
            index=False,
        )


# Create the final consolidated OSM-level network and district exports.
def main():
    print("Stage 12 - OSM-level export")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISTRICT_LINK_DIR.mkdir(parents=True, exist_ok=True)
    DISTRICT_NODE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df[df["TXDOT_DI_2"].notna()].copy()
    df["TXDOT_DI_2"] = df["TXDOT_DI_2"].astype(str).str.strip()
    df = df[df["TXDOT_DI_2"] != ""].copy()
    df["geometry"] = df["geometry"].apply(wkt.loads)

    df["consolidated_link_key"] = df["link_key"].apply(get_consolidated_link_key)
    df["segment_number"] = df["link_key"].apply(get_link_number)

    consolidated = consolidate_links(df)
    links, nodes = assign_nodes(consolidated)
    links = classify_final_links(links)

    links.to_csv(LINK_OUTPUT, index=False)
    nodes.to_csv(NODE_OUTPUT, index=False)
    export_district_files(links, nodes)

    print(f"Consolidated links: {len(links):,}")
    print(f"Consolidated nodes: {len(nodes):,}")
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
