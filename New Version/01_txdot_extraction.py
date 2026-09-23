import ast
import json
import math
import re

import numpy as np
import pandas as pd
from pyproj import Transformer
from rtree import index as rtree_index
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree

from paths import (
    TXDOT_ROADWAY,
    TXDOT_SPEED,
    TXDOT_LANES,
    TXDOT_FC,
    TXDOT_DIR,
)


# Output files
NODE_OUTPUT = TXDOT_DIR / "NOL_Speed_Roadway_nodes_data.csv"
LINK_OUTPUT = TXDOT_DIR / "link_list_with_speed_NOL.csv"
LINK_FC_OUTPUT = TXDOT_DIR / "link_list_with_speed_NOL_FC.csv"


# Coordinate transformer used by the inherited TxDOT processing logic
transformer = Transformer.from_crs(
    "epsg:3857",
    "epsg:4326",
    always_xy=True,
)


# Read one TxDOT Feature Collection text file into a DataFrame.
def read_feature_collection(file_path):

    def extract_records(data):
        rows = []

        if isinstance(data, dict):
            if "attributes" in data and "geometry" in data:
                row = dict(data["attributes"])

                for key, value in data["geometry"].items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            row[f"{key}.{sub_key}"] = sub_value
                    else:
                        row[key] = value

                rows.append(row)

            for value in data.values():
                rows.extend(extract_records(value))

        elif isinstance(data, list):
            for item in data:
                rows.extend(extract_records(item))

        return rows

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        content = content.replace('\\"', '"').replace("\\\\", "\\")
        data = json.loads(content)

    df = pd.json_normalize(extract_records(data))
    df = df.dropna(how="all")

    df.columns = [
        column.replace("geometry.", "")
        for column in df.columns
    ]

    if "paths" in df.columns:
        df["paths"] = df["paths"].apply(
            lambda value: str(value)
            if isinstance(value, list)
            else value
        )

    return df


# Convert Web Mercator coordinates to the inherited longitude/latitude tuple format.
def convert_to_latlon(coords):
    try:
        return [
            transformer.transform(x, y)
            for x, y in coords
        ]
    except Exception:
        return []


# Calculate heading using the effective heading function from the Sabin script.
def calculate_heading(coords):
    if len(coords) < 2:
        return None

    (lat1, lon1), (lat2, lon2) = coords[:2]

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    heading = math.degrees(
        math.atan2(delta_lon, delta_lat)
    )

    return round(
        (heading + 360) % 360
    )


# Convert a heading to the inherited cardinal direction.
def determine_direction(heading):
    if heading is None:
        return None

    if 315 <= heading <= 360 or 0 <= heading < 45:
        return "Northbound"

    if 45 <= heading < 135:
        return "Eastbound"

    if 135 <= heading < 225:
        return "Southbound"

    if 225 <= heading < 315:
        return "Westbound"

    return None


# Convert a final link heading to the inherited pseudo-direction.
def get_pseudo_direction(heading):
    if 0 <= heading <= 45 or 315 < heading <= 360:
        return "Northbound"

    if 45 < heading < 135:
        return "Eastbound"

    if 135 <= heading < 225:
        return "Southbound"

    if 225 <= heading <= 315:
        return "Westbound"

    return "Unknown"


# Infer missing DES_DRCT values from the beginning of each roadway geometry.
def fill_unknown_directions(df):
    if "DES_DRCT" not in df.columns:
        df["DES_DRCT"] = "Unknown"

    for idx, row in df.iterrows():

        if row.get("DES_DRCT") != "Unknown":
            continue

        try:
            paths = ast.literal_eval(
                row.get("paths", "[]")
            )

            if paths and len(paths[0]) >= 2:
                coords = convert_to_latlon(
                    paths[0][:2]
                )

                heading = calculate_heading(coords)

                df.at[idx, "DES_DRCT"] = (
                    determine_direction(heading)
                )

        except Exception:
            continue

    return df


# Apply the inherited left/right roadbed direction rule.
def update_direction(rdbd_type, direction):

    right_types = {
        "Right Roadbed",
        "Right Frontage",
        "Right Managed Lane",
        "Right Supplemental Frontage",
        "Right Supplemental Main Lane",
        "Right Supplemental Managed Lane",
    }

    left_types = {
        "Left Roadbed",
        "Left Frontage",
        "Left Managed Lane",
        "Left Supplemental Frontage",
        "Left Supplemental Main Lane",
        "Left Supplemental Managed Lane",
    }

    if rdbd_type in right_types:
        return direction

    if rdbd_type not in left_types:
        return direction

    reverse_direction = {
        "Northbound": "Southbound",
        "Southbound": "Northbound",
        "Eastbound": "Westbound",
        "Westbound": "Eastbound",
    }

    return reverse_direction.get(
        direction,
        direction,
    )


# Apply roadbed direction correction and reverse geometry when required.
def prepare_roadway(df):

    required_columns = [
        "paths",
        "spatialReference.wkid",
        "spatialReference.latestWkid",
        "EXT_DATE",
        "GID",
        "BEGIN_DFO",
        "END_DFO",
        "RTE_NM",
        "RTE_GRID",
        "RTE_PRFX",
        "RTE_NBR",
        "RTE_SFX",
        "RDBD_TYPE",
        "DES_DRCT",
        "COUNTY",
        "MAP_LBL",
        "ZOOM",
        "OBJECTID",
        "Shape__Length",
    ]

    updated_paths = []
    updated_directions = []

    for _, row in df.iterrows():

        try:
            paths = ast.literal_eval(
                row["paths"]
            )

            if not isinstance(paths, list):
                paths = [paths]

            if not all(
                isinstance(path, list)
                for path in paths
            ):
                paths = [paths]

            current_direction = row["DES_DRCT"]
            rdbd_type = row["RDBD_TYPE"]

            new_direction = update_direction(
                rdbd_type,
                current_direction,
            )

            if new_direction != current_direction:
                paths = [paths[0][::-1]]

            else:
                paths = [paths[0]]

            # Preserve the inherited ramp-direction calculation.
            if rdbd_type == "Ramp":
                coords = convert_to_latlon(
                    paths[0]
                )

                heading = calculate_heading(coords)

                if heading is not None:
                    if (
                        0 <= heading <= 45
                        or heading >= 315
                    ):
                        new_direction = "Northbound"

                    elif 45 < heading <= 135:
                        new_direction = "Eastbound"

                    elif 135 < heading <= 225:
                        new_direction = "Southbound"

                    elif 225 < heading < 315:
                        new_direction = "Westbound"

            updated_paths.append(paths)
            updated_directions.append(
                new_direction
            )

        except Exception:
            updated_paths.append(
                row["paths"]
            )

            updated_directions.append(
                row["DES_DRCT"]
            )

    df["paths"] = updated_paths
    df["DES_DRCT"] = updated_directions

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    return df[required_columns]


# Convert TxDOT feature paths into sequential node rows.
def paths_to_nodes(df, value_column=None):

    nodes = []
    node_number = 1

    for _, row in df.iterrows():

        path = ast.literal_eval(
            row["paths"]
        )

        gid = row["GID"]

        value = (
            row[value_column]
            if value_column
            else None
        )

        for segment in path:
            for x, y in segment:

                # Preserve the inherited column assignment.
                lat, lon = transformer.transform(
                    x,
                    y,
                )

                node = {
                    "NodeNumber": node_number,
                    "Latitude": lat,
                    "Longitude": lon,
                    "GID": gid,
                }

                if value_column == "SPD_LMT":
                    node["SpeedLimit"] = value

                elif value_column == "LANE_CNT":
                    node["NumberOfLanes"] = value

                nodes.append(node)

                node_number += 1

    return pd.DataFrame(nodes)


# Perform the inherited exact coordinate match.
def exact_coordinate_match(
    roadway_nodes,
    reference_nodes,
    value_column,
):

    roadway_nodes = roadway_nodes.copy()
    reference_nodes = reference_nodes.copy()

    roadway_nodes["Latitude"] = (
        roadway_nodes["Latitude"].astype(float)
    )

    roadway_nodes["Longitude"] = (
        roadway_nodes["Longitude"].astype(float)
    )

    reference_nodes["Latitude"] = (
        reference_nodes["Latitude"].astype(float)
    )

    reference_nodes["Longitude"] = (
        reference_nodes["Longitude"].astype(float)
    )

    lookup = (
        reference_nodes
        .set_index(
            ["Latitude", "Longitude"]
        )[value_column]
        .to_dict()
    )

    keys = list(
        zip(
            roadway_nodes["Latitude"],
            roadway_nodes["Longitude"],
        )
    )

    roadway_nodes[value_column] = [
        lookup.get(key)
        for key in keys
    ]

    matched = roadway_nodes[
        roadway_nodes[value_column].notna()
    ].copy()

    unmatched = roadway_nodes[
        roadway_nodes[value_column].isna()
    ][
        [
            "NodeNumber",
            "Latitude",
            "Longitude",
            "GID",
        ]
    ].copy()

    return matched, unmatched


# Match remaining nodes to the nearest reference node within one meter.
def nearest_node_match(
    reference_nodes,
    matched,
    unmatched,
    value_column,
):

    if unmatched.empty:
        return matched.copy()

    reference_coords = reference_nodes[
        ["Latitude", "Longitude"]
    ].astype(float).to_numpy()

    unmatched_coords = unmatched[
        ["Latitude", "Longitude"]
    ].astype(float).to_numpy()

    tree = BallTree(
        np.radians(reference_coords),
        metric="haversine",
    )

    distances, indices = tree.query(
        np.radians(unmatched_coords),
        k=1,
    )

    distances_m = (
        distances[:, 0] * 6371000
    )

    nearest_indices = indices[:, 0]

    values = reference_nodes.iloc[
        nearest_indices
    ][value_column].to_numpy()

    unmatched = unmatched.copy()

    unmatched[value_column] = np.where(
        distances_m < 1.0,
        values,
        np.nan,
    )

    return pd.concat(
        [
            matched,
            unmatched,
        ],
        ignore_index=True,
    )


# Create a WKT LineString from two node coordinates.
def create_linestring(points):
    return (
        "LINESTRING ("
        + ", ".join(
            f"{lon} {lat}"
            for lat, lon in points
        )
        + ")"
    )


# Extract latitude/longitude tuples from a WKT LineString.
def extract_coordinates(geometry):

    match = re.match(
        r"LINESTRING \(([^)]+)\)",
        geometry,
    )

    if not match:
        raise ValueError(
            f"Invalid LINESTRING: {geometry}"
        )

    coords = match.group(1).split(", ")

    return [
        (
            float(coord.split()[1]),
            float(coord.split()[0]),
        )
        for coord in coords
    ]


# Build TxDOT links between consecutive nodes within each GID.
def build_links(nodes_df):

    links = []
    link_id = 1

    for gid, group in nodes_df.groupby(
        "GID",
        sort=True,
    ):

        group = group.sort_values(
            "NodeNumber"
        )

        rows = list(
            group.itertuples(index=False)
        )

        for i in range(
            len(rows) - 1
        ):

            from_node = rows[i]
            to_node = rows[i + 1]

            geometry = create_linestring(
                [
                    (
                        from_node.Longitude,
                        from_node.Latitude,
                    ),
                    (
                        to_node.Longitude,
                        to_node.Latitude,
                    ),
                ]
            )

            coords = extract_coordinates(
                geometry
            )

            heading = calculate_heading(
                coords
            )

            heading1 = None
            heading2 = None

            if from_node.RDBD_TYPE == "Single Roadbed":
                heading1 = heading

                heading2 = calculate_heading(
                    coords[::-1]
                )

            links.append(
                {
                    "LinkID": link_id,
                    "FromNode": from_node.NodeNumber,
                    "ToNode": to_node.NodeNumber,
                    "GID": gid,
                    "Geometry": geometry,
                    "RDBD_TYPE": from_node.RDBD_TYPE,
                    "DES_DRCT": from_node.DES_DRCT,
                    "COUNTY": from_node.COUNTY,
                    "MAP_LBL": from_node.MAP_LBL,
                    "OBJECTID": from_node.OBJECTID,
                    "Heading": heading,
                    "Heading1": heading1,
                    "Heading2": heading2,
                    "SpeedLimit": from_node.SpeedLimit,
                    "NumberOfLanes": from_node.NumberOfLanes,
                }
            )

            link_id += 1

    links_df = pd.DataFrame(links)

    links_df["pseudo_direction"] = (
        links_df["Heading"].apply(
            get_pseudo_direction
        )
    )

    return links_df


# Find the nearest eligible mainline LineString to one ramp endpoint.
def find_nearest_line(
    point,
    filtered_lines,
    spatial_index,
):

    point_geom = Point(point)

    nearest_distance = float("inf")
    closest_link_id = None

    candidate_indices = list(
        spatial_index.nearest(
            point_geom.bounds,
            10,
        )
    )

    for line_idx in candidate_indices:

        line, link_id = (
            filtered_lines[line_idx]
        )

        distance = point_geom.distance(
            line
        )

        if distance < nearest_distance:
            nearest_distance = distance
            closest_link_id = link_id

    return (
        nearest_distance,
        closest_link_id,
    )


# Convert the inherited degree-based ramp distance to feet.
def degrees_to_feet(distance_degrees):

    earth_radius_ft = 20902230

    return (
        distance_degrees
        * earth_radius_ft
        * math.pi
        / 180
    )


# Assign On Ramp or Off Ramp using the inherited endpoint-distance rule.
def classify_ramps(links_df):

    ramps = links_df[
        links_df["RDBD_TYPE"] == "Ramp"
    ]

    if ramps.empty:
        links_df["Ramp_type"] = np.nan
        return links_df

    ramp_endpoints = []

    for gid, group in ramps.groupby(
        "GID"
    ):

        first_geometry = (
            group.iloc[0]["Geometry"]
        )

        last_geometry = (
            group.iloc[-1]["Geometry"]
        )

        first_numbers = re.findall(
            r"-?\d+\.\d+",
            first_geometry,
        )

        last_numbers = re.findall(
            r"-?\d+\.\d+",
            last_geometry,
        )

        ramp_endpoints.append(
            {
                "GID": gid,
                "First_Point": (
                    float(first_numbers[0]),
                    float(first_numbers[1]),
                ),
                "Last_Point": (
                    float(last_numbers[-2]),
                    float(last_numbers[-1]),
                ),
            }
        )

    endpoints = pd.DataFrame(
        ramp_endpoints
    )

    mainline = links_df[
        links_df["RDBD_TYPE"].isin(
            [
                "Left Roadbed",
                "Right Roadbed",
            ]
        )
    ]

    filtered_lines = []

    for row in mainline.itertuples(
        index=False
    ):

        numbers = re.findall(
            r"-?\d+\.\d+",
            row.Geometry,
        )

        coords = [
            (
                float(numbers[i]),
                float(numbers[i + 1]),
            )
            for i in range(
                0,
                len(numbers),
                2,
            )
        ]

        filtered_lines.append(
            (
                LineString(coords),
                row.LinkID,
            )
        )

    spatial_index = rtree_index.Index()

    for idx, (line, _) in enumerate(
        filtered_lines
    ):
        spatial_index.insert(
            idx,
            line.bounds,
        )

    ramp_types = {}

    for row in endpoints.itertuples(
        index=False
    ):

        first_distance, _ = (
            find_nearest_line(
                row.First_Point,
                filtered_lines,
                spatial_index,
            )
        )

        last_distance, _ = (
            find_nearest_line(
                row.Last_Point,
                filtered_lines,
                spatial_index,
            )
        )

        first_distance = degrees_to_feet(
            first_distance
        )

        last_distance = degrees_to_feet(
            last_distance
        )

        ramp_types[row.GID] = (
            "Off Ramp"
            if last_distance
            > first_distance
            else "On Ramp"
        )

    links_df["Ramp_type"] = (
        links_df["GID"].map(
            ramp_types
        )
    )

    return links_df


# Calculate link length in feet with the inherited haversine formula.
def haversine_feet(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return 20925646.3 * c


# Add link lengths from the final two-point geometries.
def add_link_lengths(links_df):

    coordinates = links_df[
        "Geometry"
    ].apply(extract_coordinates)

    lat1 = np.array(
        [coords[0][0] for coords in coordinates]
    )

    lon1 = np.array(
        [coords[0][1] for coords in coordinates]
    )

    lat2 = np.array(
        [coords[1][0] for coords in coordinates]
    )

    lon2 = np.array(
        [coords[1][1] for coords in coordinates]
    )

    links_df["length"] = haversine_feet(
        lat1,
        lon1,
        lat2,
        lon2,
    )

    return links_df


# Reverse one ramp GID using the inherited node/link reversal behavior.
def flip_ramp_gid(
    gid,
    roadway_nodes,
    links_df,
):

    node_mask = (
        roadway_nodes["GID"] == gid
    )

    nodes = roadway_nodes[
        node_mask
    ].copy()

    reversed_nodes = nodes.sort_values(
        "NodeNumber",
        ascending=False,
    )

    roadway_nodes.loc[
        node_mask,
        "NodeNumber",
    ] = reversed_nodes[
        "NodeNumber"
    ].to_numpy()

    link_mask = (
        links_df["GID"] == gid
    )

    links = links_df[
        link_mask
    ].copy()

    links_df.loc[
        links.index,
        "FromNode",
    ] = links["FromNode"].to_numpy()[::-1]

    links_df.loc[
        links.index,
        "ToNode",
    ] = links["ToNode"].to_numpy()[::-1]

    links_df.loc[
        links.index,
        "LinkID",
    ] = links["LinkID"].to_numpy()[::-1]

    for idx in links.index:

        coords = extract_coordinates(
            links_df.at[
                idx,
                "Geometry",
            ]
        )

        reversed_coords = coords[::-1]

        links_df.at[
            idx,
            "Geometry",
        ] = (
            "LINESTRING ("
            + ", ".join(
                f"{lon} {lat}"
                for lat, lon
                in reversed_coords
            )
            + ")"
        )

        if (
            links_df.at[
                idx,
                "Ramp_type",
            ]
            == "On Ramp"
        ):
            links_df.at[
                idx,
                "Ramp_type",
            ] = "Off Ramp"

        elif (
            links_df.at[
                idx,
                "Ramp_type",
            ]
            == "Off Ramp"
        ):
            links_df.at[
                idx,
                "Ramp_type",
            ] = "On Ramp"

        heading = calculate_heading(
            reversed_coords
        )

        links_df.at[
            idx,
            "Heading",
        ] = heading

        links_df.at[
            idx,
            "pseudo_direction",
        ] = get_pseudo_direction(
            heading
        )


# Apply the inherited ramp/mainline connectivity direction repair.
def fix_ramp_directions(
    roadway_nodes,
    links_df,
):

    roadway_nodes = (
        roadway_nodes
        .sort_values(
            ["GID", "NodeNumber"]
        )
        .copy()
    )

    duplicates = (
        roadway_nodes
        .groupby(
            ["Latitude", "Longitude"]
        )
        .filter(
            lambda group:
            len(group) > 1
        )
    )

    roadbed_types = {
        "Left Roadbed",
        "Right Roadbed",
    }

    valid_direction_pairs = [
        {"Northbound", "Southbound"},
        {"Eastbound", "Westbound"},
    ]

    for _, group in duplicates.groupby(
        ["Latitude", "Longitude"]
    ):

        pairs = (
            group[
                ["GID", "NodeNumber"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        for i in range(len(pairs)):
            for j in range(
                i + 1,
                len(pairs),
            ):

                gid_1 = pairs.iloc[i]["GID"]
                node_1 = pairs.iloc[i]["NodeNumber"]

                gid_2 = pairs.iloc[j]["GID"]
                node_2 = pairs.iloc[j]["NodeNumber"]

                link_1 = links_df[
                    (links_df["GID"] == gid_1)
                    & (
                        links_df["FromNode"]
                        == node_1
                    )
                ]

                link_2 = links_df[
                    (links_df["GID"] == gid_2)
                    & (
                        links_df["FromNode"]
                        == node_2
                    )
                ]

                type_1 = (
                    link_1.iloc[0]["RDBD_TYPE"]
                    if not link_1.empty
                    else None
                )

                type_2 = (
                    link_2.iloc[0]["RDBD_TYPE"]
                    if not link_2.empty
                    else None
                )

                if not (
                    (
                        type_1 == "Ramp"
                        and type_2
                        in roadbed_types
                    )
                    or (
                        type_2 == "Ramp"
                        and type_1
                        in roadbed_types
                    )
                ):
                    continue

                if type_1 == "Ramp":
                    ramp_gid = gid_1
                    ramp_link = link_1
                else:
                    ramp_gid = gid_2
                    ramp_link = link_2

                ramp_links = links_df[
                    links_df["GID"]
                    == ramp_gid
                ]

                first_row = ramp_links.iloc[0]
                last_row = ramp_links.iloc[-1]

                current_from = (
                    ramp_link.iloc[0]["FromNode"]
                )

                if (
                    current_from
                    != first_row["FromNode"]
                    and current_from
                    != last_row["FromNode"]
                ):
                    continue

                direction_1 = (
                    link_1.iloc[0][
                        "pseudo_direction"
                    ]
                    if not link_1.empty
                    else None
                )

                direction_2 = (
                    link_2.iloc[0][
                        "pseudo_direction"
                    ]
                    if not link_2.empty
                    else None
                )

                if (
                    direction_1
                    == direction_2
                ):
                    continue

                if {
                    direction_1,
                    direction_2,
                } not in valid_direction_pairs:
                    continue

                if type_1 in roadbed_types:
                    roadbed_heading = (
                        link_1.iloc[0][
                            "Heading"
                        ]
                    )

                else:
                    roadbed_heading = (
                        link_2.iloc[0][
                            "Heading"
                        ]
                    )

                # Preserve the inherited heading-window calculation.
                if (
                    45
                    <= roadbed_heading
                    <= 315
                ):
                    min_heading = (
                        roadbed_heading - 45
                    )

                    max_heading = (
                        roadbed_heading + 45
                    )

                elif roadbed_heading < 45:
                    min_heading = (
                        roadbed_heading
                        - 45
                        + 360
                    )

                    max_heading = (
                        roadbed_heading
                        + 45
                    )

                    if max_heading > 360:
                        max_heading -= 360

                else:
                    min_heading = (
                        roadbed_heading
                        - 45
                    )

                    max_heading = (
                        roadbed_heading
                        + 45
                    )

                    if max_heading > 360:
                        max_heading -= 360

                ramp_heading = (
                    ramp_link.iloc[0][
                        "Heading"
                    ]
                )

                # Preserve the inherited range test exactly.
                if not (
                    min_heading
                    <= ramp_heading
                    <= max_heading
                ):
                    flip_ramp_gid(
                        ramp_gid,
                        roadway_nodes,
                        links_df,
                    )

    return (
        roadway_nodes,
        links_df,
    )


# Add Functional Class to the final TxDOT links using GID.
def add_functional_class(
    links_df,
    fc_df,
):

    fc_df = fc_df.copy()

    fc_df["GID"] = pd.to_numeric(
        fc_df["GID"],
        errors="coerce",
    )

    links_df["GID"] = pd.to_numeric(
        links_df["GID"],
        errors="coerce",
    )

    fc_map = (
        fc_df
        .dropna(
            subset=[
                "GID",
                "FC_DESC",
            ]
        )
        .drop_duplicates(
            subset="GID"
        )
        .set_index("GID")[
            "FC_DESC"
        ]
    )

    links_df[
        "Functional Class"
    ] = links_df["GID"].map(
        fc_map
    )

    return links_df


# Run the complete rewritten TxDOT extraction stage.
def main():

    print(
        "Stage 01 - TxDOT extraction"
    )

    roadway = read_feature_collection(
        TXDOT_ROADWAY
    )

    speed = read_feature_collection(
        TXDOT_SPEED
    )

    lanes = read_feature_collection(
        TXDOT_LANES
    )

    functional_class = (
        read_feature_collection(
            TXDOT_FC
        )
    )

    roadway = fill_unknown_directions(
        roadway
    )

    roadway = prepare_roadway(
        roadway
    )

    roadway["paths"] = roadway[
        "paths"
    ].apply(
        lambda value:
        str(value)
        if isinstance(value, list)
        else value
    )

    roadway_attributes = (
        roadway.copy()
    )

    roadway_nodes = paths_to_nodes(
        roadway
    )

    speed_nodes = paths_to_nodes(
        speed,
        "SPD_LMT",
    )

    lane_nodes = paths_to_nodes(
        lanes,
        "LANE_CNT",
    )

    matched_speed, unmatched_speed = (
        exact_coordinate_match(
            roadway_nodes,
            speed_nodes,
            "SpeedLimit",
        )
    )

    speed_result = (
        nearest_node_match(
            speed_nodes,
            matched_speed,
            unmatched_speed,
            "SpeedLimit",
        )
    )

    matched_lanes, unmatched_lanes = (
        exact_coordinate_match(
            roadway_nodes,
            lane_nodes,
            "NumberOfLanes",
        )
    )

    lane_result = (
        nearest_node_match(
            lane_nodes,
            matched_lanes,
            unmatched_lanes,
            "NumberOfLanes",
        )
    )

    merged_nodes = pd.merge(
        speed_result,
        lane_result,
        on=[
            "NodeNumber",
            "Latitude",
            "Longitude",
            "GID",
        ],
    )

    nodes_for_links = merged_nodes.merge(
        roadway_attributes[
            [
                "GID",
                "RDBD_TYPE",
                "DES_DRCT",
                "COUNTY",
                "MAP_LBL",
                "OBJECTID",
            ]
        ],
        on="GID",
    )

    links = build_links(
        nodes_for_links
    )

    links = classify_ramps(
        links
    )

    links = add_link_lengths(
        links
    )

    corrected_nodes, links = (
        fix_ramp_directions(
            merged_nodes,
            links,
        )
    )

    links = links.sort_values(
        "LinkID"
    )

    links_with_fc = (
        add_functional_class(
            links.copy(),
            functional_class,
        )
    )

    corrected_nodes.to_csv(
        NODE_OUTPUT,
        index=False,
    )

    links.to_csv(
        LINK_OUTPUT,
        index=False,
    )

    links_with_fc.to_csv(
        LINK_FC_OUTPUT,
        index=False,
    )

    print(
        f"Roadway nodes: {len(corrected_nodes):,}"
    )

    print(
        f"TxDOT links: {len(links):,}"
    )

    print(
        "Saved:",
        LINK_FC_OUTPUT,
    )


if __name__ == "__main__":
    main()