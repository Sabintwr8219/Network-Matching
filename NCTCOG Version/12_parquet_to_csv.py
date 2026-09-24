"""Export a compact, checked CSV from the validated final NCTCOG Parquet."""

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from paths import FINAL_NETWORK_CSV, FINAL_NETWORK_FILE


BATCH_SIZE = 250_000
OUTPUT_COLUMNS = [
    "link_key", "name", "osm_way_id", "from_node_id", "to_node_id",
    "geometry", "facility_type", "heading", "length",
    "nctcog_key", "nctcog_volume",
]
INPUT_COLUMNS = [
    "link_key", "name", "osm_way_id", "from_node_id", "to_node_id",
    "geometry", "facility_type", "link_type", "heading", "length",
    "nctcog_id", "nctcog_direction", "nctcog_dir_code",
    "nctcog_dayvol_ab", "nctcog_dayvol_ba",
    "nctcog_directional_volume", "nctcog_direction_key",
]


def convert_parquet_to_csv(overwrite=False):
    if not FINAL_NETWORK_FILE.is_file():
        raise FileNotFoundError(FINAL_NETWORK_FILE)
    if FINAL_NETWORK_CSV.exists() and not overwrite:
        raise FileExistsError(f"CSV exists: {FINAL_NETWORK_CSV}; use --overwrite")

    source = pq.ParquetFile(FINAL_NETWORK_FILE)
    missing = sorted(set(INPUT_COLUMNS) - set(source.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    FINAL_NETWORK_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = FINAL_NETWORK_CSV.with_suffix(".partial.csv")
    temporary.unlink(missing_ok=True)
    start = time.perf_counter()
    rows = assigned_total = 0
    pairs = set()

    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            for batch in source.iter_batches(batch_size=BATCH_SIZE, columns=INPUT_COLUMNS):
                frame = batch.to_pandas()
                facility = frame["facility_type"].astype("string")
                link_type = frame["link_type"].astype("string")
                pairs.update(zip(facility.fillna("<NULL>"), link_type.fillna("<NULL>")))

                assigned = frame["nctcog_direction_key"].notna()
                cols = ["nctcog_id", "nctcog_direction", "nctcog_dir_code",
                        "nctcog_dayvol_ab", "nctcog_dayvol_ba",
                        "nctcog_directional_volume"]
                if frame.loc[~assigned, cols].notna().any().any():
                    raise ValueError("Unmatched row has NCTCOG attributes")

                selected = frame.loc[assigned]
                direction = selected["nctcog_direction"].astype("string")
                source_dir = pd.to_numeric(selected["nctcog_dir_code"], errors="coerce")
                if (not direction.isin(["AB", "BA"]).all()
                        or not source_dir.isin([0, 1]).all()
                        or (direction.eq("BA") & source_dir.ne(0)).any()):
                    raise ValueError("NCTCOG travel direction conflicts with source Dir")

                original_key = selected["nctcog_direction_key"].astype("string")
                expected_key = selected["nctcog_id"].astype("string") + "_" + direction
                if expected_key.isna().any() or not original_key.eq(expected_key).all():
                    raise ValueError("Directional key disagrees with NCTCOG ID/direction")

                ab = pd.to_numeric(selected["nctcog_dayvol_ab"], errors="coerce")
                ba = pd.to_numeric(selected["nctcog_dayvol_ba"], errors="coerce")
                actual = pd.to_numeric(selected["nctcog_directional_volume"], errors="coerce")
                expected_volume = np.where(direction.eq("AB"), ab, ba)
                if (not np.isfinite(actual.to_numpy(dtype=float)).all()
                        or (actual < 0).any()
                        or not np.allclose(actual.to_numpy(dtype=float), expected_volume,
                                           rtol=0, atol=0.01, equal_nan=False)):
                    raise ValueError("Directional volume disagrees with source AB/BA volume")

                compact = frame[OUTPUT_COLUMNS[:-2]].copy()
                compact["nctcog_key"] = pd.Series(pd.NA, index=frame.index, dtype="string")
                compact.loc[assigned, "nctcog_key"] = (
                    selected["nctcog_id"].astype("string") + "_"
                    + direction.map({"AB": "0", "BA": "1"})
                )
                compact["nctcog_volume"] = frame["nctcog_directional_volume"]
                compact.to_csv(output, index=False, header=rows == 0)
                rows += len(frame)
                assigned_total += len(selected)
                print(f"Checked and written {rows:,}/{source.metadata.num_rows:,} rows", flush=True)

        if rows != source.metadata.num_rows:
            raise ValueError("CSV row count differs from Parquet metadata")

        mapping = pd.DataFrame(sorted(pairs), columns=["facility_type", "link_type"])
        print("\nFACILITY_TYPE / LINK_TYPE DISTINCT PAIRS")
        print(mapping.to_string(index=False))
        if (mapping["facility_type"].duplicated().any()
                or mapping["link_type"].duplicated().any()):
            raise ValueError("Facility type and link type are not one-to-one; "
                             "review the pairs before dropping link_type")

        temporary.replace(FINAL_NETWORK_CSV)
    except Exception:
        print(f"Conversion stopped. Existing final CSV was not replaced; "
              f"partial output: {temporary}")
        raise

    print("\nCOMPACT CSV COMPLETE")
    print(f"Rows: {rows:,} | assigned: {assigned_total:,} | unmatched: {rows-assigned_total:,}")
    print("Columns:", ", ".join(OUTPUT_COLUMNS))
    print(f"Elapsed: {time.perf_counter()-start:.1f}s")
    print(f"Saved: {FINAL_NETWORK_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    convert_parquet_to_csv(**vars(parser.parse_args()))
