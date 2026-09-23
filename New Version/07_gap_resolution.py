from collections import defaultdict, deque

import pandas as pd

from paths import INTERMEDIATE_DIR, TXDOT_DIR


INPUT_FILE = INTERMEDIATE_DIR / "matched_network_basic_short_link_ConvergingFixed.csv"
TXDOT_INPUT = TXDOT_DIR / "link_list_with_speed_NOL_FC.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "matched_network_basic_with_group_chain_Multiple_Fixed.csv"


# Build forward, backward, and per-link lookup dictionaries.
def build_lookups(df):
    from_node_map = defaultdict(list)
    to_node_map = defaultdict(list)
    link_data = {}

    for row in df.itertuples():
        from_node_map[row.from_node_id].append(row.link_id)
        to_node_map[row.to_node_id].append(row.link_id)
        link_data[row.link_id] = row

    return from_node_map, to_node_map, link_data


# Find one connected component of unresolved links using both directions.
def bfs_group(start_id, unresolved_links, from_node_map, to_node_map, link_data):
    group = []
    queue = deque([start_id])
    visited = {start_id}

    while queue:
        current_id = queue.popleft()
        group.append(current_id)
        row = link_data[current_id]

        neighbors = (
            from_node_map.get(row.to_node_id, [])
            + to_node_map.get(row.from_node_id, [])
        )

        for neighbor in neighbors:
            if neighbor in unresolved_links and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return group


# Assign chain group IDs to all unresolved connected components.
def assign_chain_groups(df):
    df = df.copy()
    df["Chain_Group_ID"] = None

    from_node_map, to_node_map, link_data = build_lookups(df)
    group_id_counter = 1

    while True:
        unresolved = set(
            df.loc[
                (df["Matched_GID"] == -1) & df["Chain_Group_ID"].isna(),
                "link_id",
            ]
        )

        if not unresolved:
            break

        assigned = set()
        groups = []

        for link_id in unresolved:
            if link_id not in assigned:
                group = bfs_group(
                    link_id,
                    unresolved,
                    from_node_map,
                    to_node_map,
                    link_data,
                )
                assigned.update(group)
                groups.append(group)

        for group in groups:
            local_visited = set()

            def group_label(number):
                return f"group_{number}"

            start_link = None

            for link_id in group:
                row = link_data[link_id]
                upstream = [x for x in to_node_map[row.from_node_id] if x in group]

                if not upstream:
                    start_link = link_id
                    break

            if not start_link:
                for link_id in group:
                    if str(link_data[link_id].merge) in [
                        "Diverging",
                        "Converging",
                        "Diverging-Converging",
                    ]:
                        start_link = link_id
                        break

            if not start_link:
                start_link = group[0]

            stack = deque([(start_link, group_label(group_id_counter))])

            while stack:
                current_link, current_group = stack.popleft()

                if current_link in local_visited:
                    continue

                local_visited.add(current_link)
                df.loc[df["link_id"] == current_link, "Chain_Group_ID"] = current_group

                merge_type = str(link_data[current_link].merge)
                next_links = from_node_map.get(link_data[current_link].to_node_id, [])
                next_links = [
                    link_id
                    for link_id in next_links
                    if link_id in group and link_id not in local_visited
                ]

                if merge_type == "Converging":
                    next_group = group_label(group_id_counter + 1)

                    for next_link in next_links:
                        stack.append((next_link, next_group))

                    group_id_counter += 1

                elif merge_type == "Diverging":
                    next_group = group_label(group_id_counter + 1)
                    df.loc[df["link_id"] == current_link, "Chain_Group_ID"] = next_group

                    for next_link in next_links:
                        stack.append((next_link, next_group))

                    group_id_counter += 1

                else:
                    for next_link in next_links:
                        stack.append((next_link, current_group))

            group_id_counter += 1

    df["Chain"] = df["Matched_GID"] == -1

    group_sizes = (
        df.loc[df["Chain"] & df["Chain_Group_ID"].notna()]
        .groupby("Chain_Group_ID")["link_id"]
        .count()
        .to_dict()
    )

    df["Group_Type"] = [
        (
            "single"
            if row.Chain and pd.notna(row.Chain_Group_ID) and group_sizes.get(row.Chain_Group_ID, 0) == 1
            else "multiple"
            if row.Chain and pd.notna(row.Chain_Group_ID)
            else None
        )
        for row in df.itertuples()
    ]

    return df


# Fill unresolved groups containing exactly one short link.
def resolve_single_gaps(df):
    df = df.copy()
    from_node_map, to_node_map, link_data = build_lookups(df)
    single_df = df[df["Group_Type"] == "single"].copy()
    filled = 0

    for group_id, group_rows in single_df.groupby("Chain_Group_ID"):
        link_id = group_rows.iloc[0]["link_id"]
        merge_type = group_rows.iloc[0]["merge"]
        from_node = link_data[link_id].from_node_id
        to_node = link_data[link_id].to_node_id

        backward_links = to_node_map.get(from_node, [])
        forward_links = from_node_map.get(to_node, [])
        new_gid = None

        if pd.isna(merge_type):
            back_gid = None
            forward_gid = None

            if len(backward_links) == 1:
                back_gid = getattr(link_data[backward_links[0]], "Matched_GID", None)

            if len(forward_links) == 1:
                forward_gid = getattr(link_data[forward_links[0]], "Matched_GID", None)

            if pd.notna(back_gid) and pd.notna(forward_gid):
                if back_gid == forward_gid:
                    new_gid = back_gid
            elif pd.notna(back_gid):
                new_gid = back_gid
            elif pd.notna(forward_gid):
                new_gid = forward_gid

        elif merge_type == "Diverging-Converging":
            if len(forward_links) == 1:
                forward_row = link_data.get(forward_links[0])

                if forward_row is not None:
                    gid = getattr(forward_row, "Matched_GID", None)
                    next_merge = getattr(forward_row, "merge", None)

                    if pd.notna(gid) and (
                        pd.isna(next_merge)
                        or next_merge not in ["Converging", "Diverging", "Diverging-Converging"]
                    ):
                        new_gid = gid

        if new_gid is not None:
            df.loc[df["Chain_Group_ID"] == group_id, "Matched_GID"] = new_gid
            filled += 1

    return df, filled


# Fill unresolved groups containing multiple connected short links.
def resolve_multiple_gaps(df):
    df = df.copy()
    from_node_map, to_node_map, link_data = build_lookups(df)
    multiple_df = df[df["Group_Type"] == "multiple"].copy()
    filled = 0

    for group_id, group_links in multiple_df.groupby("Chain_Group_ID"):
        node_count = defaultdict(int)

        for row in group_links.itertuples():
            node_count[row.from_node_id] += 1
            node_count[row.to_node_id] += 1

        endpoints = [node for node, count in node_count.items() if count == 1]

        if len(endpoints) != 2:
            continue

        first_link = next(
            (row.link_id for row in group_links.itertuples() if row.from_node_id in endpoints),
            None,
        )
        last_link = next(
            (row.link_id for row in group_links.itertuples() if row.to_node_id in endpoints),
            None,
        )

        if not first_link or not last_link:
            continue

        first_merge = str(getattr(link_data[first_link], "merge", None))
        last_merge = str(getattr(link_data[last_link], "merge", None))

        if first_merge == "nan":
            first_merge = None

        if last_merge == "nan":
            last_merge = None

        backward_links = to_node_map.get(link_data[first_link].from_node_id, [])
        forward_links = from_node_map.get(link_data[last_link].to_node_id, [])

        if not backward_links:
            backward_gids = {"no_gid"}
        else:
            backward_gids = {
                getattr(link_data[link_id], "Matched_GID", -1)
                if pd.notna(getattr(link_data[link_id], "Matched_GID", None))
                else -1
                for link_id in backward_links
            }

        if not forward_links:
            forward_gids = {"no_gid"}
        else:
            forward_gids = {
                getattr(link_data[link_id], "Matched_GID", -1)
                if pd.notna(getattr(link_data[link_id], "Matched_GID", None))
                else -1
                for link_id in forward_links
            }

        valid_forward = {gid for gid in forward_gids if gid != "no_gid"}
        valid_backward = {gid for gid in backward_gids if gid != "no_gid"}
        common_gids = valid_forward & valid_backward
        positive_common = {gid for gid in common_gids if gid != -1}
        new_gid = None

        if len(positive_common) == 1:
            new_gid = next(iter(positive_common))
        elif len(positive_common) == 0 and -1 in common_gids:
            new_gid = -1
        elif "no_gid" in backward_gids and len(valid_forward) == 1 and last_merge in [None, "None", "nan"]:
            new_gid = next(iter(valid_forward))
        elif "no_gid" in forward_gids and len(valid_backward) == 1 and first_merge in [None, "None", "nan"]:
            new_gid = next(iter(valid_backward))
        elif not common_gids:
            if last_merge == "Converging" and first_merge in [None, "None", "nan"] and len(valid_backward) == 1:
                new_gid = next(iter(valid_backward))
            elif first_merge == "Converging" and last_merge in [None, "None", "nan"] and len(valid_forward) == 1:
                new_gid = next(iter(valid_forward))
            elif last_merge == "Diverging" and first_merge in [None, "None", "nan"] and len(valid_backward) == 1:
                new_gid = next(iter(valid_backward))
            elif first_merge == "Diverging" and last_merge in [None, "None", "nan"] and len(valid_forward) == 1:
                new_gid = next(iter(valid_forward))

        if new_gid is not None:
            df.loc[df["Chain_Group_ID"] == group_id, "Matched_GID"] = new_gid
            filled += 1

    return df, filled


# Map final freeway GIDs to TxDOT roadbed type and functional class.
def add_txdot_attributes(df):
    txdot = pd.read_csv(TXDOT_INPUT, low_memory=False)
    txdot["GID"] = pd.to_numeric(txdot["GID"], errors="coerce")
    df["Matched_GID"] = pd.to_numeric(df["Matched_GID"], errors="coerce")

    roadbed_map = (
        txdot.dropna(subset=["GID", "RDBD_TYPE"])
        .drop_duplicates("GID")
        .set_index("GID")["RDBD_TYPE"]
    )

    functional_class_map = (
        txdot.dropna(subset=["GID", "Functional Class"])
        .drop_duplicates("GID")
        .set_index("GID")["Functional Class"]
    )

    df["RDBD_TYPE"] = df["Matched_GID"].map(roadbed_map)
    df["Functional Class"] = df["Matched_GID"].map(functional_class_map)

    return df


# Run chain grouping, single-gap repair, and multiple-gap repair.
def main():
    print("Stage 07 - Gap resolution")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["Matched_GID"] = pd.to_numeric(df["Matched_GID"], errors="coerce")

    unresolved_before = int((df["Matched_GID"] == -1).sum())

    df = assign_chain_groups(df)
    single_groups = df.loc[df["Group_Type"] == "single", "Chain_Group_ID"].nunique()
    multiple_groups = df.loc[df["Group_Type"] == "multiple", "Chain_Group_ID"].nunique()

    df, single_filled = resolve_single_gaps(df)
    df, multiple_filled = resolve_multiple_gaps(df)
    df = add_txdot_attributes(df)

    unresolved_after = int((df["Matched_GID"] == -1).sum())
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Unresolved before: {unresolved_before:,}")
    print(f"Single groups: {single_groups:,} | filled: {single_filled:,}")
    print(f"Multiple groups: {multiple_groups:,} | filled: {multiple_filled:,}")
    print(f"Unresolved after: {unresolved_after:,}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
