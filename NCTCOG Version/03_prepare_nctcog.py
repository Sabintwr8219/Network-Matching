import math

import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

from paths import NCTCOG_INPUT, INTERMEDIATE_DIR, check_inputs

OUTPUT_FILE = INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv"

def prepare_directional_links(df):
    """Represent existing links by travel direction; preserve source columns."""
    # Working interpretation: Dir=0 is two-way; Dir=1 is AB-only.
    if not df["Dir"].isin([0, 1]).all():
        raise ValueError("Unexpected NCTCOG Dir value.")

    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("NCTCOG IDs must be present and unique.")

    source = df.copy()
    source["ID"] = source["ID"].astype("string")

    ab = source.copy()
    ab["nctcog_direction"] = "AB"
    ab["travel_from_node"] = ab["FROM_NODE"]
    ab["travel_to_node"] = ab["TO_NODE"]
    ab["travel_geometry"] = ab["WKT"]
    ab["directional_volume"] = ab["DAYVOL_AB"]

    ba = source.loc[source["Dir"].eq(0)].copy()
    ba["nctcog_direction"] = "BA"
    ba["travel_from_node"] = ba["TO_NODE"]
    ba["travel_to_node"] = ba["FROM_NODE"]
    ba["travel_geometry"] = ba["WKT"].map(
        lambda text: LineString(
            list(wkt.loads(text).coords)[::-1]
        ).wkt
    )
    ba["directional_volume"] = ba["DAYVOL_BA"]

    prepared = pd.concat([ab, ba], ignore_index=True)
    prepared["nctcog_direction_key"] = (
        prepared["ID"] + "_" + prepared["nctcog_direction"]
    )

    return prepared

def main():
    check_inputs()

    source = pd.read_csv(NCTCOG_INPUT, dtype={"ID": "string"})
    prepared = prepare_directional_links(source)

    expected_rows = len(source) + int(source["Dir"].eq(0).sum())

    if len(prepared) != expected_rows:
        raise ValueError("Unexpected directional record count.")

    if prepared["nctcog_direction_key"].duplicated().any():
        raise ValueError("Duplicate directional keys.")

    # Check each source link retains its total volume across directions.
    totals = prepared.groupby("ID")["directional_volume"].sum()
    expected = source.set_index("ID")["DAYVOL"]
    differences = (totals.reindex(expected.index) - expected).abs()

    if differences.isna().any() or differences.gt(0.01).any():
        raise ValueError("Directional volumes do not preserve source totals.")

    # Check travel geometry and endpoints for both directions.
    for row in prepared.itertuples(index=False):
        coordinates = list(wkt.loads(row.WKT).coords)
        reverse = row.nctcog_direction == "BA"
        expected_coordinates = coordinates[::-1] if reverse else coordinates
        actual_coordinates = list(wkt.loads(row.travel_geometry).coords)

        expected_from = row.TO_NODE if reverse else row.FROM_NODE
        expected_to = row.FROM_NODE if reverse else row.TO_NODE

        if (
            actual_coordinates != expected_coordinates
            or row.travel_from_node != expected_from
            or row.travel_to_node != expected_to
        ):
            raise ValueError(
                f"Direction or geometry mismatch: {row.nctcog_direction_key}"
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_FILE, index=False)

    print(f"Source links: {len(source):,}")
    print(prepared["nctcog_direction"].value_counts().to_string())
    print(f"Prepared records: {len(prepared):,}")
    print("Checks passed: unique keys, volume totals, travel geometry/endpoints.")
    print("Direction assumption: Dir=0 two-way; Dir=1 AB-only.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()