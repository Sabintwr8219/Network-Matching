from collections import defaultdict

import pandas as pd

from paths import INTERMEDIATE_DIR


INPUT_FILE = INTERMEDIATE_DIR / "matched_network_basic_short_link.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "matched_network_basic_short_link_ConvergingFixed.csv"

MAX_TRACE_LINKS = 10
MERGE_TYPES = {"Diverging", "Converging", "Diverging-Converging"}


# Build node lookups and per-link records for topology tracing.
def build_lookups(df):
    from_node_map = defaultdict(list)
    to_node_map = defaultdict(list)
    link_data = {}

    for row in df.itertuples(index=True):
        link_data[row.link_id] = {
            "idx": row.Index,
            "from_node_id": row.from_node_id,
            "to_node_id": row.to_node_id,
            "Matched_GID": row.Matched_GID,
            "merge": row.merge,
        }
        from_node_map[row.from_node_id].append(row.link_id)
        to_node_map[row.to_node_id].append(row.link_id)

    return from_node_map, to_node_map, link_data


# Update a traced path with one GID and mark the affected links as fixed.
def update_path(df, link_data, path, new_gid):
    indexes = [link_data[link_id]["idx"] for link_id in path if link_id in link_data]

    if not indexes:
        return

    df.loc[indexes, "Matched_GID"] = new_gid
    df.loc[indexes, "Fixed"] = "Yes"

    for link_id in path:
        if link_id in link_data:
            link_data[link_id]["Matched_GID"] = new_gid


# Check whether a traced path contains a Diverging-Converging link.
def path_has_dc(link_data, path):
    return any(
        link_id in link_data and link_data[link_id]["merge"] == "Diverging-Converging"
        for link_id in path
    )


# Trace forward until topology changes, the network ends, or a different GID is found.
def trace_forward(link_id, original_gid, from_node_map, link_data):
    path = [link_id]
    current_link = link_id
    found_gid = None
    depth = 0

    while depth < MAX_TRACE_LINKS:
        to_node = link_data.get(current_link, {}).get("to_node_id")

        if to_node is None:
            break

        next_links = [lid for lid in from_node_map.get(to_node, []) if lid != current_link]

        if len(next_links) != 1:
            break

        next_link = next_links[0]
        next_data = link_data.get(next_link)

        if next_data is None:
            break

        if next_data["merge"] in MERGE_TYPES:
            break

        next_gid = next_data["Matched_GID"]

        if next_gid != original_gid:
            found_gid = next_gid
            path.append(next_link)
            break

        path.append(next_link)
        current_link = next_link
        depth += 1

    return path, found_gid if found_gid is not None else original_gid


# Apply the diverging-path correction rules.
def correct_diverging(df):
    df = df.copy()
    df["Fixed"] = "No"

    from_node_map, _, link_data = build_lookups(df)
    fix_count = 0

    for node, outgoing_links in from_node_map.items():
        diverging_links = [
            link_id
            for link_id in outgoing_links
            if link_data[link_id]["merge"] in {"Diverging", "Diverging-Converging"}
        ]

        if len(diverging_links) != 2:
            continue

        link1, link2 = diverging_links
        gid1 = link_data[link1]["Matched_GID"]
        gid2 = link_data[link2]["Matched_GID"]

        if gid1 != -1 and gid2 != -1 and gid1 != gid2:
            continue

        path1, ref_gid1 = trace_forward(link1, gid1, from_node_map, link_data)
        path2, ref_gid2 = trace_forward(link2, gid2, from_node_map, link_data)

        if ref_gid1 == ref_gid2:
            path1_has_dc = path_has_dc(link_data, path1)
            path2_has_dc = path_has_dc(link_data, path2)

            if ref_gid1 == -1:
                update_path(df, link_data, path1 + path2, -1)
            elif path1_has_dc and not path2_has_dc:
                update_path(df, link_data, path2, -1)
            elif path2_has_dc and not path1_has_dc:
                update_path(df, link_data, path1, -1)
            elif len(path1) > len(path2):
                update_path(df, link_data, path2, -1)
            elif len(path2) > len(path1):
                update_path(df, link_data, path1, -1)
            else:
                update_path(df, link_data, path1, -1)

            fix_count += 1
            continue

        if ref_gid1 != gid1:
            update_path(df, link_data, path1, ref_gid1)
            fix_count += 1

        if ref_gid2 != gid2:
            update_path(df, link_data, path2, ref_gid2)
            fix_count += 1

    return df, fix_count


# Trace backward until topology changes, the network ends, or a different GID is found.
def trace_backward(link_id, original_gid, to_node_map, link_data):
    path = [link_id]
    current_link = link_id
    found_gid = None
    depth = 0

    while depth < MAX_TRACE_LINKS:
        current_data = link_data.get(current_link)

        if current_data is None:
            break

        from_node = current_data.get("from_node_id")

        if from_node is None:
            break

        previous_links = [lid for lid in to_node_map.get(from_node, []) if lid != current_link]

        if len(previous_links) != 1:
            break

        previous_link = previous_links[0]
        previous_data = link_data.get(previous_link)

        if previous_data is None:
            break

        if previous_data["merge"] in MERGE_TYPES:
            break

        previous_gid = previous_data["Matched_GID"]

        if previous_gid != original_gid:
            found_gid = previous_gid
            path.append(previous_link)
            break

        path.append(previous_link)
        current_link = previous_link
        depth += 1

    return path, found_gid if found_gid is not None else original_gid


# Apply the converging-path correction rules.
def correct_converging(df):
    df = df.copy()
    df["Fixed"] = "No"

    _, to_node_map, link_data = build_lookups(df)
    fix_count = 0

    for node, incoming_links in to_node_map.items():
        converging_links = [
            link_id
            for link_id in incoming_links
            if link_data[link_id]["merge"] in MERGE_TYPES
        ]

        if len(converging_links) != 2:
            continue

        link1, link2 = converging_links
        gid1 = link_data[link1]["Matched_GID"]
        gid2 = link_data[link2]["Matched_GID"]

        if gid1 != -1 and gid2 != -1 and gid1 != gid2:
            continue

        path1, ref_gid1 = trace_backward(link1, gid1, to_node_map, link_data)
        path2, ref_gid2 = trace_backward(link2, gid2, to_node_map, link_data)

        if ref_gid1 == ref_gid2:
            path1_has_dc = path_has_dc(link_data, path1)
            path2_has_dc = path_has_dc(link_data, path2)

            if ref_gid1 == -1:
                update_path(df, link_data, path1 + path2, -1)
            elif path1_has_dc and not path2_has_dc:
                update_path(df, link_data, path2, -1)
            elif path2_has_dc and not path1_has_dc:
                update_path(df, link_data, path1, -1)
            elif len(path1) > len(path2):
                update_path(df, link_data, path2, -1)
            elif len(path2) > len(path1):
                update_path(df, link_data, path1, -1)
            else:
                update_path(df, link_data, path1, -1)

            fix_count += 1
            continue

        if ref_gid1 != gid1:
            update_path(df, link_data, path1, ref_gid1)
            fix_count += 1

        if ref_gid2 != gid2:
            update_path(df, link_data, path2, ref_gid2)
            fix_count += 1

    return df, fix_count


# Run diverging correction followed by converging correction.
def main():
    print("Stage 06 - Topology correction")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["Matched_GID"] = pd.to_numeric(df["Matched_GID"], errors="coerce").fillna(-1)

    df, diverging_count = correct_diverging(df)
    df, converging_count = correct_converging(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Diverging cases changed: {diverging_count:,}")
    print(f"Converging cases changed: {converging_count:,}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
