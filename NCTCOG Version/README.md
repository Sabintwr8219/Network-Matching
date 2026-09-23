# DFW OSM–NCTCOG Network Attribution

This workflow attaches directional NCTCOG roadway attributes and daily volume to the DFW OSM short-link network. It preserves every original OSM link and records how each accepted NCTCOG assignment was obtained.

## Method

1. Select complete Dallas and Fort Worth OSM links within the NCTCOG study extent.
2. Convert NCTCOG links into travel-direction records.
   - `Dir = 0`: create `AB` and `BA` records.
   - `Dir = 1`: create the `AB` record only.
3. Divide OSM and NCTCOG geometry into matching pieces of at most 10 metres.
4. Match pieces using the fixed limits of 6 metres and 15 degrees.
5. Consolidate piece support to one dominant NCTCOG directional parent per OSM link.
6. Repair only conservative topology gaps:
   - one-link gaps bounded by the same reliable NCTCOG parent;
   - multi-link chains bounded by the same reliable parent and passing travel-progression checks;
   - geometrically identical reverse OSM link pairs.
7. Leave conflicting junctions and uncertain links unmatched.
8. Attach NCTCOG attributes to all OSM rows and validate the final network.

## Production scripts

| Stage | Script | Role |
|---:|---|---|
| 02 | `02_prepare_dfw_osm.py` | Creates the OSM-only DFW input from the Dallas and Fort Worth district files. |
| 03 | `03_prepare_nctcog.py` | Creates directional NCTCOG records and validates direction and volume totals. |
| 04 | `04_segment_10m.py` | Builds 10-m OSM and NCTCOG matching pieces in EPSG:32614. |
| 05 | `05_spatial_matching.py` | Applies the fixed distance and heading match rules. |
| 06 | `06_consolidate_matches.py` | Selects dominant NCTCOG support for each original OSM link. |
| 07 | `07_gap_filling.py` | Evaluates and applies conservative single- and multi-link gap repairs. |
| 08 | `08_topology_repairs.py` | Applies reverse-pair inference and reports unresolved junction conflicts. |
| 09 | `09_nctcog_coverage.py` | Measures retained direct length coverage on NCTCOG directional records. |
| 10 | `10_final_export.py` | Writes the full OSM network with NCTCOG attributes and provenance. |
| 11 | `11_validate_final.py` | Verifies row identity, keys, assignments, directions, and volumes. |

`run_nctcog_pipeline.py` runs or resumes these stages. `paths.py` contains the shared project locations.

## Inputs

The repository is expected at a project root containing `.git` and these local data files:

- `Initial Input Files/NCTCOG/links_NCTCOG.csv`
- `Output/New Version/Final/OSM_Short_Level_CSV/District_Link_List/Dallas_Link_List.csv`
- `Output/New Version/Final/OSM_Short_Level_CSV/District_Link_List/Fort_Worth_Link_List.csv`

The prepared OSM file is written to:

- `NCTCOG Version/Prepared Inputs/DFW_OSM_short_links_only.csv`

Source datasets and generated outputs should remain outside Git.

## Run

Activate the repository virtual environment and run the complete workflow:

```powershell
Set-Location -LiteralPath "D:\Sabin Project\San Antonio\Network-Processing"
& ".\.venv\Scripts\python.exe" ".\NCTCOG Version\run_nctcog_pipeline.py"
```

The current project already has valid outputs through Stage 9. Complete only the final export and validation with:

```powershell
& ".\.venv\Scripts\python.exe" ".\NCTCOG Version\run_nctcog_pipeline.py" --start 10
```

Existing completed stages are skipped. Add `--force` only when intentionally rebuilding selected outputs. Add `--csv` if a large CSV copy is required; Parquet is the default and recommended format.

## Final outputs

- `Output/Final/DFW_OSM_with_NCTCOG_attributes.parquet`
- `Output/Final/DFW_OSM_NCTCOG_validation.json`
- optional: `Output/Final/DFW_OSM_with_NCTCOG_attributes.csv`

The final Parquet retains all original OSM columns and adds:

- NCTCOG directional key, ID, direction, endpoints, name, functional class, and length fields;
- AB, BA, total, and selected directional daily volume;
- assignment method and direct-match support measures;
- gap-repair and reverse-pair provenance flags.

Unmatched OSM links remain in the file with null NCTCOG attributes and `assignment_method = unmatched`.

## Interpretation

`nctcog_directional_volume` is an attribute of the NCTCOG directional parent. The same value is copied to every OSM child link assigned to that parent. Do not sum this field across OSM links; doing so repeats the parent volume. Use unique `nctcog_direction_key` records for NCTCOG-level totals.

The last validated assignment state contained 3,780,936 OSM links, including 806,798 links assigned to 57,119 of 69,477 NCTCOG directional records. Uncertain cases remain unmatched rather than being forced.
