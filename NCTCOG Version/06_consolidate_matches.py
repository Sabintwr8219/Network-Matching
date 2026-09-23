import time

import numpy as np
import pandas as pd
import psutil

from paths import INTERMEDIATE_DIR, OSM_INPUT


MATCH_INPUT = INTERMEDIATE_DIR / "OSM_NCTCOG_piece_matches.parquet"
SUMMARY_OUTPUT = INTERMEDIATE_DIR / "OSM_link_match_summary.parquet"
SUPPORT_OUTPUT = INTERMEDIATE_DIR / "OSM_NCTCOG_parent_support.parquet"

def consolidate_matches(pieces):
    pieces = pieces.copy()
    matched = pieces["matched_nctcog_key"].notna()

    pieces["matched_length_m"] = np.where(
        matched, pieces["length_m"], 0.0
    )
    pieces["ambiguous_length_m"] = np.where(
        pieces["candidate_parent_count"].gt(1),
        pieces["length_m"],
        0.0,
    )

    summary = pieces.groupby("parent_key", sort=False).agg(
        piece_count=("length_m", "size"),
        total_length_m=("length_m", "sum"),
        matched_length_m=("matched_length_m", "sum"),
        ambiguous_length_m=("ambiguous_length_m", "sum"),
    )

    support = (
        pieces.loc[matched]
        .groupby(["parent_key", "matched_nctcog_key"], sort=False)
        .agg(
            supporting_length_m=("length_m", "sum"),
            supporting_pieces=("length_m", "size"),
        )
        .reset_index()
    )

    # Largest supporting length wins; exact ties use the parent key.
    ranked = support.sort_values(
        ["parent_key", "supporting_length_m", "matched_nctcog_key"],
        ascending=[True, False, True],
    )
    winners = ranked.drop_duplicates("parent_key").set_index("parent_key")

    summary["dominant_nctcog_key"] = winners["matched_nctcog_key"]
    summary["dominant_length_m"] = (
        winners["supporting_length_m"].reindex(summary.index).fillna(0)
    )
    summary["selected_parent_count"] = (
        support.groupby("parent_key").size()
        .reindex(summary.index).fillna(0).astype("int32")
    )

    # Flag practically equal support; this is numerical tolerance,
    # not a change to the spatial matching thresholds.
    if not support.empty:
        best_lengths = support["parent_key"].map(
            winners["supporting_length_m"]
        )
        near_best = np.isclose(
            support["supporting_length_m"],
            best_lengths,
            rtol=1e-9,
            atol=1e-6,
        )
        tied_counts = support.loc[near_best].groupby("parent_key").size()
        summary["dominant_tie"] = (
            tied_counts.reindex(summary.index).fillna(0).gt(1)
        )
    else:
        summary["dominant_tie"] = False

    summary["matched_fraction"] = (
        summary["matched_length_m"] / summary["total_length_m"]
    )
    summary["dominant_fraction"] = (
        summary["dominant_length_m"] / summary["total_length_m"]
    )
    summary["dominant_share_of_matched"] = (
        summary["dominant_length_m"]
        / summary["matched_length_m"].replace(0, np.nan)
    )

    summary["consolidation_status"] = np.select(
        [
            summary["selected_parent_count"].eq(0),
            summary["dominant_tie"],
            summary["selected_parent_count"].eq(1),
        ],
        ["unmatched", "dominant_tie", "single_parent"],
        default="multiple_parents",
    )

    return summary.reset_index(), support

def main():
    process = psutil.Process()
    started = time.perf_counter()
    cpu_start = sum(process.cpu_times()[:2])

    print("Reading piece matches...", flush=True)
    pieces = pd.read_parquet(
        MATCH_INPUT,
        columns=[
            "parent_key",
            "length_m",
            "matched_nctcog_key",
            "candidate_parent_count",
        ],
    )
    if len(pieces) != 17_448_950:
        raise ValueError("Unexpected piece count.")

    expected_length = pieces["length_m"].sum()
    expected_matched_length = pieces.loc[
        pieces["matched_nctcog_key"].notna(), "length_m"
    ].sum()

    print("Consolidating by original OSM link...", flush=True)
    summary, support = consolidate_matches(pieces)
    del pieces

    print("Attaching original OSM endpoints...", flush=True)
    links = pd.read_csv(
        OSM_INPUT,
        usecols=["link_key", "from_node_id", "to_node_id", "facility_type"],
        dtype={
            "link_key": "string",
            "from_node_id": "string",
            "to_node_id": "string",
        },
    )

    summary = links.merge(
        summary.rename(columns={"parent_key": "link_key"}),
        on="link_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(summary) != 3_780_936 or not summary["_merge"].eq("both").all():
        raise ValueError("Consolidated links do not match original OSM links.")
    summary = summary.drop(columns="_merge")

    if not np.isclose(summary["total_length_m"].sum(), expected_length):
        raise ValueError("Total length changed during consolidation.")
    if not np.isclose(
        summary["matched_length_m"].sum(), expected_matched_length
    ):
        raise ValueError("Matched length changed during consolidation.")
    if (summary["matched_fraction"] > 1 + 1e-9).any():
        raise ValueError("Matched length exceeds total link length.")

    print("Saving summary and parent support...", flush=True)
    for frame, destination in [
        (summary, SUMMARY_OUTPUT),
        (support, SUPPORT_OUTPUT),
    ]:
        temporary = destination.with_suffix(".partial.parquet")
        frame.to_parquet(temporary, index=False, compression="snappy")
        temporary.replace(destination)

    print("\nCONSOLIDATION COMPLETE")
    print(f"Original OSM links: {len(summary):,}")
    print(summary["consolidation_status"].value_counts().to_string())

    print("\nDIRECT-MATCH BASELINE")
    for label, group in [
        ("All", summary),
        *list(summary.groupby("facility_type", sort=True)),
    ]:
        any_support = group["matched_length_m"].gt(0)
        link_pct = 100 * any_support.mean()
        length_pct = (
            100 * group["matched_length_m"].sum()
            / group["total_length_m"].sum()
        )
        print(
            f"{label}: {any_support.sum():,}/{len(group):,} links "
            f"with any support ({link_pct:.2f}%) | "
            f"Matched length: {length_pct:.2f}%"
        )

    elapsed = time.perf_counter() - started
    cpu_used = sum(process.cpu_times()[:2]) - cpu_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Average CPU share: "
        f"{100 * cpu_used / elapsed / psutil.cpu_count():.1f}%"
    )
    print(f"Current process RAM: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"Saved: {SUMMARY_OUTPUT}")
    print(f"Saved: {SUPPORT_OUTPUT}")


if __name__ == "__main__":
    main()