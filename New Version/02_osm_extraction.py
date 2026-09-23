import math
from collections import Counter, defaultdict

import osm2gmns as og
import pandas as pd

from paths import OSM_PBF, OSM_DIR


# Output files
RAW_LINK_FILE = OSM_DIR / "link.csv"
LINK_SHORT_FILE = OSM_DIR / "link_short.csv"
NODE_SHORT_FILE = OSM_DIR / "node_short.csv"
FINAL_LINK_FILE = OSM_DIR / "link_short_sequence_complete.csv"
DUPLICATE_FILE = OSM_DIR / "osm_short_b_duplicates.csv"


# Calculate link length in feet using the haversine formula.
def haversine_ft(lon1, lat1, lon2, lat2):
    radius_m = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return round(
        radius_m * c * 3.28084,
        2,
    )


# Calculate link heading clockwise from North.
def calculate_heading(lon1, lat1, lon2, lat2):
    d_lon = math.radians(
        lon2 - lon1
    )

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    x = (
        math.sin(d_lon)
        * math.cos(lat2_rad)
    )

    y = (
        math.cos(lat1_rad)
        * math.sin(lat2_rad)
        - math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(d_lon)
    )

    bearing = math.atan2(x, y)

    return round(
        (math.degrees(bearing) + 360) % 360
    )


# Parse one WKT LINESTRING into longitude-latitude coordinate pairs.
def parse_linestring(text):
    text = text[12:-1]

    points = []

    for point in text.split(","):
        lon, lat = point.strip().split()

        points.append(
            (
                float(lon),
                float(lat),
            )
        )

    return points


# Split OSM LineStrings into consecutive two-point short links.
def split_osm_links(df):
    rows = []

    for row in df.to_dict("records"):

        try:
            coords = parse_linestring(
                row["geometry"]
            )
        except Exception:
            continue

        for i in range(
            len(coords) - 1
        ):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            new_row = row.copy()

            new_row["geometry"] = (
                f"LINESTRING ({lon1} {lat1}, "
                f"{lon2} {lat2})"
            )

            new_row["heading"] = (
                calculate_heading(
                    lon1,
                    lat1,
                    lon2,
                    lat2,
                )
            )

            new_row["length"] = (
                haversine_ft(
                    lon1,
                    lat1,
                    lon2,
                    lat2,
                )
            )

            new_row["from_node"] = (
                f"{lon1:.8f} {lat1:.8f}"
            )

            new_row["to_node"] = (
                f"{lon2:.8f} {lat2:.8f}"
            )

            rows.append(new_row)

    segment_df = pd.DataFrame(rows)

    segment_df.drop(
        columns=["link_id"],
        inplace=True,
        errors="ignore",
    )

    segment_df.insert(
        0,
        "link_id",
        range(
            1,
            len(segment_df) + 1,
        ),
    )

    return segment_df


# Build unique OSM nodes and assign node IDs to every short link.
def build_node_table(segment_df):
    all_nodes = pd.concat(
        [
            segment_df["from_node"],
            segment_df["to_node"],
        ],
        ignore_index=True,
    ).drop_duplicates().reset_index(
        drop=True
    )

    coordinates = all_nodes.str.split(
        " ",
        expand=True,
    )

    node_table = pd.DataFrame(
        {
            "node_id": range(
                1,
                len(all_nodes) + 1,
            ),
            "longitude": coordinates[
                0
            ].astype(float),
            "latitude": coordinates[
                1
            ].astype(float),
        }
    )

    node_lookup = dict(
        zip(
            all_nodes,
            node_table["node_id"],
        )
    )

    segment_df["from_node_id"] = (
        segment_df["from_node"].map(
            node_lookup
        )
    )

    segment_df["to_node_id"] = (
        segment_df["to_node"].map(
            node_lookup
        )
    )

    return segment_df, node_table


# Mark links that have an existing reverse-direction counterpart.
def identify_directional_links(segment_df):
    link_pairs = set(
        zip(
            segment_df["from_node_id"],
            segment_df["to_node_id"],
        )
    )

    segment_df["directional"] = [
        (to_node, from_node)
        in link_pairs
        for from_node, to_node in zip(
            segment_df["from_node_id"],
            segment_df["to_node_id"],
        )
    ]

    return segment_df


# Order connected links within one non-directional OSM way.
def order_links(group):
    from_to = {}

    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for idx, row in group.iterrows():
        from_node = row["from_node_id"]
        to_node = row["to_node_id"]

        from_to[from_node] = (
            to_node,
            idx,
        )

        out_degree[from_node] += 1
        in_degree[to_node] += 1

        if from_node not in in_degree:
            in_degree[from_node] += 0

        if to_node not in out_degree:
            out_degree[to_node] += 0

    start_nodes = [
        node
        for node in from_to
        if in_degree[node] == 0
    ]

    if not start_nodes:
        start_nodes = [
            list(from_to.keys())[0]
        ]

    current = start_nodes[0]

    ordered = []
    visited = set()

    while current in from_to:
        next_node, idx = (
            from_to[current]
        )

        if idx in visited:
            break

        ordered.append(idx)
        visited.add(idx)

        current = next_node

    return ordered


# Assign short-link sequence IDs to non-directional OSM ways.
def sequence_non_directional_links(
    segment_df,
):
    subset = segment_df[
        segment_df["directional"] == False
    ].copy()

    subset["osm_short_a"] = (
        subset["osm_way_id"]
        .astype(str)
        + "_0"
    )

    for _, group in subset.groupby(
        "osm_way_id"
    ):
        ordered_idx = order_links(
            group
        )

        if len(group) == 0:
            continue

        base = group[
            "osm_short_a"
        ].iloc[0]

        for sequence, idx in enumerate(
            ordered_idx,
            start=1,
        ):
            subset.loc[
                idx,
                "osm_short_b",
            ] = (
                f"{base}_{sequence}"
            )

    segment_df.loc[
        subset.index,
        "osm_short_a",
    ] = subset["osm_short_a"]

    segment_df.loc[
        subset.index,
        "osm_short_b",
    ] = subset["osm_short_b"]

    return segment_df


# Find a valid starting link for one directional OSM way.
def find_valid_start_link(links):
    edge_lookup = defaultdict(list)

    for idx, (u, v) in links.items():
        edge_lookup[(u, v)].append(
            idx
        )

    reverse_of = {
        idx: next(
            (
                reverse_idx
                for reverse_idx
                in edge_lookup.get(
                    (v, u),
                    [],
                )
                if reverse_idx != idx
            ),
            None,
        )
        for idx, (u, v)
        in links.items()
    }

    forward_map = defaultdict(list)

    for idx, (u, v) in links.items():
        forward_map[u].append(
            (
                v,
                idx,
            )
        )

    node_frequency = defaultdict(int)

    for u, v in links.values():
        node_frequency[u] += 1
        node_frequency[v] += 1

    edge_nodes = {
        node
        for node, count
        in node_frequency.items()
        if count == 2
    }

    if len(edge_nodes) < 2:
        return None

    candidates = []

    for idx, (u, v) in links.items():

        if (
            u not in edge_nodes
            and v not in edge_nodes
        ):
            continue

        forward_options = [
            (next_node, next_idx)
            for next_node, next_idx
            in forward_map[v]
            if next_idx
            != reverse_of[idx]
        ]

        if forward_options:
            candidates.append(idx)

    return (
        candidates[0]
        if candidates
        else None
    )


# Assign short-link sequence IDs to normal directional OSM ways.
def sequence_directional_links(
    segment_df,
):
    subset = segment_df[
        segment_df["directional"] == True
    ].copy()

    subset["osm_short_a"] = pd.Series(
        pd.NA,
        index=subset.index,
        dtype="object",
    )

    subset["osm_short_b"] = pd.Series(
        pd.NA,
        index=subset.index,
        dtype="object",
    )

    for osm_way, group in subset.groupby(
        "osm_way_id"
    ):

        if len(group) <= 1:
            continue

        if len(group) == 2:
            idx1, idx2 = (
                group.index.tolist()
            )

            u1 = group.loc[
                idx1,
                "from_node_id",
            ]

            v1 = group.loc[
                idx1,
                "to_node_id",
            ]

            u2 = group.loc[
                idx2,
                "from_node_id",
            ]

            v2 = group.loc[
                idx2,
                "to_node_id",
            ]

            if (
                u1 == v2
                and v1 == u2
            ):
                if u1 < u2:
                    subset.loc[
                        idx1,
                        "osm_short_a",
                    ] = 1

                    subset.loc[
                        idx1,
                        "osm_short_b",
                    ] = (
                        f"{osm_way}_1_1"
                    )

                    subset.loc[
                        idx2,
                        "osm_short_a",
                    ] = 2

                    subset.loc[
                        idx2,
                        "osm_short_b",
                    ] = (
                        f"{osm_way}_2_1"
                    )

                else:
                    subset.loc[
                        idx2,
                        "osm_short_a",
                    ] = 1

                    subset.loc[
                        idx2,
                        "osm_short_b",
                    ] = (
                        f"{osm_way}_1_1"
                    )

                    subset.loc[
                        idx1,
                        "osm_short_a",
                    ] = 2

                    subset.loc[
                        idx1,
                        "osm_short_b",
                    ] = (
                        f"{osm_way}_2_1"
                    )

                continue

        links = {
            idx: (
                row["from_node_id"],
                row["to_node_id"],
            )
            for idx, row
            in group.iterrows()
        }

        start_idx = (
            find_valid_start_link(
                links
            )
        )

        if start_idx is None:
            continue

        start_u, _ = links[start_idx]

        visited = set()

        current = start_idx

        segment_id = 1
        sequence = 1

        while True:

            if current in visited:
                break

            visited.add(current)

            subset.loc[
                current,
                "osm_short_a",
            ] = segment_id

            subset.loc[
                current,
                "osm_short_b",
            ] = (
                f"{osm_way}_"
                f"{segment_id}_"
                f"{sequence}"
            )

            sequence += 1

            u, v = links[current]

            forward_links = []

            for next_idx, (
                next_from,
                next_to,
            ) in links.items():

                if next_idx in visited:
                    continue

                if next_from == v:
                    forward_links.append(
                        next_idx
                    )

            next_link = None

            if len(forward_links) == 2:

                for next_idx in (
                    forward_links
                ):
                    if (
                        links[next_idx]
                        != (v, u)
                    ):
                        next_link = (
                            next_idx
                        )
                        break

            elif len(forward_links) == 1:
                next_link = (
                    forward_links[0]
                )

                if (
                    links[next_link]
                    == (v, u)
                ):
                    segment_id = 2
                    sequence = 1

            else:
                break

            if next_link is None:
                break

            if (
                links[next_link][1]
                == start_u
            ):
                subset.loc[
                    next_link,
                    "osm_short_a",
                ] = segment_id

                subset.loc[
                    next_link,
                    "osm_short_b",
                ] = (
                    f"{osm_way}_"
                    f"{segment_id}_"
                    f"{sequence}"
                )

                break

            current = next_link

    segment_df.loc[
        subset.index,
        "osm_short_a",
    ] = subset["osm_short_a"]

    segment_df.loc[
        subset.index,
        "osm_short_b",
    ] = subset["osm_short_b"]

    return segment_df


# Check whether an unresolved directional OSM way is a normal circular case.
def is_circular(group):
    frequency = Counter(
        list(group["from_node_id"])
        + list(group["to_node_id"])
    )

    return (
        len(frequency) > 0
        and all(
            count == 4
            for count
            in frequency.values()
        )
    )


# Follow one direction around a circular OSM way.
def follow_circle(
    start_idx,
    group,
):
    chain = []
    visited = set()

    current = start_idx

    start_node = group.loc[
        start_idx,
        "from_node_id",
    ]

    while current not in visited:
        visited.add(current)
        chain.append(current)

        current_to = group.loc[
            current,
            "to_node_id",
        ]

        current_from = group.loc[
            current,
            "from_node_id",
        ]

        next_link = None

        for idx, row in group.iterrows():

            if (
                idx not in visited
                and row["from_node_id"]
                == current_to
                and row["to_node_id"]
                != current_from
            ):
                next_link = idx
                break

        if next_link is None:
            break

        if (
            group.loc[
                next_link,
                "from_node_id",
            ]
            == start_node
        ):
            break

        current = next_link

    return chain


# Complete sequence IDs for normal circular OSM ways.
def sequence_normal_circles(df):
    unresolved = df[
        (df["directional"] == True)
        & df["osm_short_b"].isna()
    ].copy()

    circular_ids = [
        osm_way
        for osm_way, group
        in unresolved.groupby(
            "osm_way_id"
        )
        if is_circular(group)
    ]

    df["circular"] = "no"

    df.loc[
        df["osm_way_id"].isin(
            circular_ids
        ),
        "circular",
    ] = "yes"

    processed = 0

    for osm_way, group in df[
        df["circular"] == "yes"
    ].groupby("osm_way_id"):

        reference = group.index[0]

        ref_from = group.loc[
            reference,
            "from_node_id",
        ]

        ref_to = group.loc[
            reference,
            "to_node_id",
        ]

        reverse = next(
            (
                idx
                for idx, row
                in group.iterrows()
                if (
                    row["from_node_id"]
                    == ref_to
                    and row["to_node_id"]
                    == ref_from
                )
            ),
            None,
        )

        if reverse is None:
            continue

        forward_chain = (
            follow_circle(
                reference,
                group,
            )
        )

        reverse_chain = (
            follow_circle(
                reverse,
                group,
            )
        )

        if (
            len(forward_chain)
            != len(reverse_chain)
        ):
            continue

        for sequence, idx in enumerate(
            forward_chain,
            start=1,
        ):
            df.loc[
                idx,
                "osm_short_a",
            ] = 1

            df.loc[
                idx,
                "osm_short_b",
            ] = (
                f"{osm_way}_1_"
                f"{sequence}"
            )

        for sequence, idx in enumerate(
            reverse_chain,
            start=1,
        ):
            df.loc[
                idx,
                "osm_short_a",
            ] = 2

            df.loc[
                idx,
                "osm_short_b",
            ] = (
                f"{osm_way}_2_"
                f"{sequence}"
            )

        processed += 1

    return df, processed


# Count endpoint occurrences within one OSM way.
def get_node_frequency(group):
    return Counter(
        list(group["from_node_id"])
        + list(group["to_node_id"])
    )


# Check for the special circular 2-6-4 node-frequency pattern.
def is_target_case(frequency):
    values = list(
        frequency.values()
    )

    return (
        values.count(2) == 1
        and values.count(6) == 1
        and all(
            value in [2, 4, 6]
            for value in values
        )
    )


# Build the forward chain for one special circular OSM way.
def build_special_chain(group):
    lookup = {
        idx: row
        for idx, row
        in group.iterrows()
    }

    frequency = (
        get_node_frequency(group)
    )

    start_node = [
        node
        for node, count
        in frequency.items()
        if count == 2
    ][0]

    reference = None

    for idx, row in group.iterrows():

        if (
            row["from_node_id"]
            != start_node
        ):
            continue

        next_links = group[
            group["from_node_id"]
            == row["to_node_id"]
        ]

        reverse_exists = any(
            next_row["from_node_id"]
            == row["to_node_id"]
            and next_row["to_node_id"]
            == row["from_node_id"]
            for _, next_row
            in next_links.iterrows()
        )

        if (
            len(next_links) > 0
            and reverse_exists
        ):
            reference = idx
            break

    if reference is None:
        return None

    chain = []
    visited_nodes = set()

    current = reference

    while True:
        row = lookup[current]

        chain.append(current)

        visited_nodes.update(
            [
                row["from_node_id"],
                row["to_node_id"],
            ]
        )

        possible = group[
            group["from_node_id"]
            == row["to_node_id"]
        ]

        possible = possible[
            ~(
                (
                    possible["to_node_id"]
                    == row["from_node_id"]
                )
                & (
                    possible[
                        "from_node_id"
                    ]
                    == row["to_node_id"]
                )
            )
        ]

        if len(possible) == 0:
            break

        next_link = next(
            (
                idx
                for idx, next_row
                in possible.iterrows()
                if (
                    next_row["to_node_id"]
                    in visited_nodes
                )
            ),
            possible.index[0],
        )

        current = next_link

        if (
            lookup[current][
                "to_node_id"
            ]
            in visited_nodes
        ):
            chain.append(current)
            break

    return chain


# Create forward and reverse sequence IDs for one special circular OSM way.
def process_special_group(group):
    frequency = (
        get_node_frequency(group)
    )

    if not is_target_case(
        frequency
    ):
        return None

    chain = build_special_chain(
        group
    )

    if chain is None:
        return None

    way_id = str(
        group.iloc[0]["osm_way_id"]
    )

    chain_length = len(chain)

    result = {}

    for sequence, idx in enumerate(
        chain,
        start=1,
    ):
        result[idx] = (
            1,
            f"{way_id}_1_{sequence}",
        )

        row = group.loc[idx]

        reverse = group[
            (
                group["from_node_id"]
                == row["to_node_id"]
            )
            & (
                group["to_node_id"]
                == row["from_node_id"]
            )
        ]

        if len(reverse) > 0:
            reverse_idx = (
                reverse.index[0]
            )

            result[reverse_idx] = (
                2,
                f"{way_id}_2_"
                f"{chain_length - sequence + 1}",
            )

    return result


# Complete sequence IDs for special unresolved circular OSM ways.
def sequence_special_circles(df):
    unresolved = df[
        df["osm_short_b"].isna()
    ].copy()

    processed = 0

    for _, group in unresolved.groupby(
        "osm_way_id"
    ):
        result = (
            process_special_group(
                group
            )
        )

        if result is None:
            continue

        for idx, (
            short_a,
            short_b,
        ) in result.items():

            df.loc[
                idx,
                "osm_short_a",
            ] = short_a

            df.loc[
                idx,
                "osm_short_b",
            ] = short_b

        processed += 1

    return df, processed


# Save duplicate short-link identifiers for inspection.
def check_duplicates(df):
    valid = df[
        df["osm_short_b"].notna()
    ]

    duplicates = valid[
        valid["osm_short_b"].duplicated(
            keep=False
        )
    ]

    if len(duplicates) > 0:
        duplicates.to_csv(
            DUPLICATE_FILE,
            index=False,
        )

    elif DUPLICATE_FILE.exists():
        DUPLICATE_FILE.unlink()

    return (
        duplicates[
            "osm_short_b"
        ].nunique(),
        len(duplicates),
    )


# Run the complete OSM extraction and sequencing stage.
def main():
    print(
        "Stage 02 - OSM extraction"
    )

    print(
        "Reading OSM PBF..."
    )

    network = og.getNetFromFile(
        str(OSM_PBF)
    )

    og.outputNetToCSV(
        network,
        str(OSM_DIR),
    )

    raw_links = pd.read_csv(
        RAW_LINK_FILE,
        low_memory=False,
    )

    print(
        f"Original OSM links: "
        f"{len(raw_links):,}"
    )

    segment_df = split_osm_links(
        raw_links
    )

    print(
        f"Short links created: "
        f"{len(segment_df):,}"
    )

    segment_df, node_table = (
        build_node_table(
            segment_df
        )
    )

    segment_df = (
        identify_directional_links(
            segment_df
        )
    )

    segment_df[
        "osm_short_a"
    ] = pd.Series(
        pd.NA,
        index=segment_df.index,
        dtype="object",
    )

    segment_df[
        "osm_short_b"
    ] = pd.Series(
        pd.NA,
        index=segment_df.index,
        dtype="object",
    )

    segment_df = (
        sequence_non_directional_links(
            segment_df
        )
    )

    segment_df = (
        sequence_directional_links(
            segment_df
        )
    )

    preferred = [
        "link_id",
        "from_node_id",
        "to_node_id",
        "from_node",
        "to_node",
        "heading",
        "length",
    ]

    other_columns = [
        column
        for column
        in segment_df.columns
        if column not in preferred
    ]

    segment_df = segment_df[
        preferred + other_columns
    ]

    column_order = [
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
        "osm_short_a",
        "osm_short_b",
    ]

    segment_df = segment_df[
        column_order
    ]

    segment_df.to_csv(
        LINK_SHORT_FILE,
        index=False,
    )

    node_table.to_csv(
        NODE_SHORT_FILE,
        index=False,
    )

    segment_df, normal_circular = (
        sequence_normal_circles(
            segment_df
        )
    )

    segment_df, special_circular = (
        sequence_special_circles(
            segment_df
        )
    )

    duplicate_ids, duplicate_rows = (
        check_duplicates(
            segment_df
        )
    )

    segment_df.to_csv(
        FINAL_LINK_FILE,
        index=False,
    )

    directional_count = int(
        segment_df[
            "directional"
        ].sum()
    )

    unresolved_count = int(
        segment_df[
            "osm_short_b"
        ].isna().sum()
    )

    print(
        f"Directional links: "
        f"{directional_count:,}"
    )

    print(
        f"Non-directional links: "
        f"{len(segment_df) - directional_count:,}"
    )

    print(
        f"Normal circular ways: "
        f"{normal_circular:,}"
    )

    print(
        f"Special circular ways: "
        f"{special_circular:,}"
    )

    print(
        f"Unresolved sequences: "
        f"{unresolved_count:,}"
    )

    print(
        f"Duplicate IDs: "
        f"{duplicate_ids:,}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Unique nodes: "
        f"{len(node_table):,}"
    )

    print(
        "Saved:",
        FINAL_LINK_FILE,
    )


if __name__ == "__main__":
    main()