# TxDOT–OSM Network Processing

This folder contains the rewritten network-processing workflow for
extracting TxDOT and OpenStreetMap networks, matching their links,
applying topology and gap corrections, and exporting the resulting network.

## Workflow

Run the stages in the following order, or use `run_pipeline.py`.

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_txdot_extraction.py` | Prepare the TxDOT network and roadway attributes. |
| 02 | `02_osm_extraction.py` | Extract and prepare the OSM network. |
| 03 | `03_segment_10m.py` | Divide links into short pieces for spatial matching. |
| 04 | `04_freeway_spatial_matching.py` | Spatially match freeway pieces to TxDOT records. |
| 05 | `05_dominant_gid_restore.py` | Consolidate matches using the dominant TxDOT identifier and restore links. |
| 06 | `06_topology_correction.py` | Apply topology-based corrections. |
| 07 | `07_gap_resolution.py` | Resolve eligible gaps in freeway assignments. |
| 08 | `08_arterial_matching.py` | Match arterial links. |
| 09 | `09_combine_networks.py` | Combine the freeway and arterial results. |
| 10 | `10_assign_district.py` | Attach TxDOT district information. |
| 11 | `11_short_network_export.py` | Export the short-link network. |
| 12 | `12_osm_level_export.py` | Export the network at the OSM level. |

`paths.py` defines the input and output locations.
`run_pipeline.py` compiles the Python files, runs the twelve stages
sequentially, and stops if a stage fails.

## Setup

Use a Git clone of the repository: `paths.py` locates the project root
by searching for `.git`.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

If the project virtual environment already exists, use it without
recreating it.

## Input data

Place the required source datasets under:

- `Initial Input Files/TxDOT/`: roadway, speed-limit, through-lane,
  functional-classification, and district-boundary datasets.
- `Initial Input Files/OSM/`: the OSM PBF extract.

The exact filenames are configured in `New Version/paths.py`.
Keep the district shapefile and its companion files together.

Input datasets are supplied separately and are not included in Git.

Check the configured inputs from the repository root:

```powershell
& ".\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'.\New Version'); import paths; paths.check_inputs(); print('All required input paths found.')"
```

## Run

Run the complete workflow from the repository root:

```powershell
& ".\.venv\Scripts\python.exe" ".\New Version\run_pipeline.py"
```

To run an individual stage:

```powershell
& ".\.venv\Scripts\python.exe" ".\New Version\04_freeway_spatial_matching.py"
```

Individual stages require the outputs of their preceding stages.

## Outputs

Outputs are written under `Output/New Version/`:

- `Intermediate/`: extracted networks and intermediate processing results.
- `Final/`: exported networks.
- `pipeline_run.log`: console output from the complete pipeline run.


Generated outputs and the virtual environment are excluded from Git.