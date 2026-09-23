import re

import pandas as pd
from shapely import wkt

from paths import FINAL_DIR, INTERMEDIATE_DIR


INPUT_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT_with_Districts.csv"
CLEAN_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT_with_Districts_Clean.csv"

OUTPUT_DIR = FINAL_DIR / "OSM_Short_Level_CSV"
LINK_OUTPUT = OUTPUT_DIR / "Complete_OSM_Short_Level_Link_List.csv"
NODE_OUTPUT = OUTPUT_DIR / "Complete_OSM_Short_Level_Node_List.csv"
DISTRICT_LINK_DIR = OUTPUT_DIR / "District_Link_List"
DISTRICT_NODE_DIR = OUTPUT_DIR / "District_Node_List"


# Convert a district name into a safe CSV filename.
def clean_filename(name):
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return re.sub(r"\s+", "_", name)


# Keep the database fields and standardize their data types.
def clean_network(df):
    df = df.copy()
    df["osm_short_b"] = df["osm_short_b"].astype("string").str.strip()
    df = df[df["osm_short_b"].notna() & df["osm_short_b"].ne("")].copy()

    columns = [
        "osm_short_b",
        "name",
        "osm_way_id",
        "from_node_id",
        "to_node_id",
        "geometry",
        "facility_type",
        "link_type",
        "free_speed",
        "lanes",
        "capacity",
        "allowed_uses",
        "heading",
        "directional",
        "Matched_GID",
        "RDBD_TYPE",
        "Functional Class",
        "TXDOT_DI_1",
        "TXDOT_DI_2",
        "length",
    ]

    df = df[columns].copy()
    df = df.rename(columns={"osm_short_b": "link_key"})

    string_columns = [
        "link_key",
        "name",
        "osm_way_id",
        "facility_type",
        "allowed_uses",
        "Matched_GID",
        "RDBD_TYPE",
        "Functional Class",
        "TXDOT_DI_2",
    ]

    integer_columns = [
        "from_node_id",
        "to_node_id",
        "link_type",
        "free_speed",
        "lanes",
        "capacity",
        "heading",
        "TXDOT_DI_1",
    ]

    df[string_columns] = df[string_columns].astype("string")
    df[integer_columns] = df[integer_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce").astype("Int64")
    )

    df["directional"] = df["directional"].astype("boolean")
    df["geometry"] = df["geometry"].astype(object)
    df["length"] = pd.to_numeric(df["length"], errors="coerce").round(1)

    return df


# Assign global node numbers from short-link start and end coordinates.
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

    output = df.copy()
    output["from_node_id"] = from_nodes
    output["to_node_id"] = to_nodes

    return output, pd.DataFrame(node_rows)


# Reclassify OSM facilities into final freeway and arterial link types.
def classify_final_links(df):
    df = df.copy()
    facility = df["facility_type"].astype(str).str.strip().str.lower()
    df["facility_type"] = facility.where(~facility.isin(["motorway", "trunk"]), "freeway")
    df.loc[df["facility_type"] != "freeway", "facility_type"] = "arterial"
    df["link_type"] = df["facility_type"].map({"freeway": 1, "arterial": 2})
    return df


# Save complete and district-specific OSM short-level link and node files.
def export_short_network(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISTRICT_LINK_DIR.mkdir(parents=True, exist_ok=True)
    DISTRICT_NODE_DIR.mkdir(parents=True, exist_ok=True)

    df = df[df["TXDOT_DI_2"].notna()].copy()
    df["TXDOT_DI_2"] = df["TXDOT_DI_2"].astype(str).str.strip()
    df = df[df["TXDOT_DI_2"] != ""].copy()

    links, nodes = assign_nodes(df)
    links = classify_final_links(links)

    links.to_csv(LINK_OUTPUT, index=False)
    nodes.to_csv(NODE_OUTPUT, index=False)

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

    return links, nodes


# Clean the network and create the final OSM short-level exports.
def main():
    print("Stage 11 - Short network export")

    raw = pd.read_csv(INPUT_FILE, low_memory=False)
    before = len(raw)
    clean = clean_network(raw)
    removed = before - len(clean)
    duplicates = int(clean["link_key"].duplicated().sum())

    clean.to_csv(CLEAN_FILE, index=False)
    links, nodes = export_short_network(clean)

    print(f"Rows removed without short-link ID: {removed:,}")
    print(f"Duplicate short-link IDs: {duplicates:,}")
    print(f"Final short links: {len(links):,}")
    print(f"Final short nodes: {len(nodes):,}")
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
