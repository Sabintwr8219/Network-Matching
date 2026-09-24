import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from paths import (
    FINAL_NETWORK_FILE,
    FINAL_NETWORK_CSV,
    FINAL_VALIDATION_FILE,
    INTERMEDIATE_DIR,
    NCTCOG_CODE_DIR,
    NCTCOG_COVERAGE_FILE,
    NCTCOG_PREPARED_FILE,
    OSM_ASSIGNMENT_FILE,
    OSM_INPUT,
)


@dataclass(frozen=True)
class Stage:
    number: int
    label: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]


STAGES = (
    Stage(2, "Prepare DFW OSM source", (("02_prepare_dfw_osm.py",),), (OSM_INPUT,)),
    Stage(
        3,
        "Prepare directional NCTCOG records",
        (("03_prepare_nctcog.py",),),
        (NCTCOG_PREPARED_FILE,),
    ),
    Stage(
        4,
        "Create 10-m matching pieces",
        (("04_segment_10m.py",),),
        (
            INTERMEDIATE_DIR / "OSM_segmented_10m.parquet",
            INTERMEDIATE_DIR / "NCTCOG_segmented_10m.parquet",
        ),
    ),
    Stage(
        5,
        "Spatially match pieces",
        (("05_spatial_matching.py",),),
        (INTERMEDIATE_DIR / "OSM_NCTCOG_piece_matches.parquet",),
    ),
    Stage(
        6,
        "Consolidate piece matches",
        (("06_consolidate_matches.py",),),
        (
            INTERMEDIATE_DIR / "OSM_link_match_summary.parquet",
            INTERMEDIATE_DIR / "OSM_NCTCOG_parent_support.parquet",
        ),
    ),
    Stage(
        7,
        "Apply conservative single- and multi-gap repairs",
        (("07_gap_filling.py", "--stage", "all"),),
        (INTERMEDIATE_DIR / "OSM_link_matches_after_multi_gap.parquet",),
    ),
    Stage(
        8,
        "Apply reverse-pair repairs and evaluate junctions",
        (
            ("08_topology_repairs.py", "--stage", "reverse_candidates"),
            ("08_topology_repairs.py", "--stage", "reverse_apply"),
            ("08_topology_repairs.py", "--stage", "junction_candidates"),
        ),
        (
            OSM_ASSIGNMENT_FILE,
            INTERMEDIATE_DIR / "OSM_junction_candidates.parquet",
        ),
    ),
    Stage(
        9,
        "Measure NCTCOG length coverage",
        (("09_nctcog_coverage.py",),),
        (NCTCOG_COVERAGE_FILE,),
    ),
    Stage(
        10,
        "Export the attributed OSM network",
        (("10_final_export.py",),),
        (FINAL_NETWORK_FILE,),
    ),
    Stage(
        11,
        "Validate the final network",
        (("11_validate_final.py",),),
        (FINAL_VALIDATION_FILE,),
    ),
    Stage(
        12,
        "Export compact checked CSV",
        (("12_parquet_to_csv.py",),),
        (FINAL_NETWORK_CSV,),
    ),
)


# Run one pipeline command with the active Python interpreter.
def run_command(command):
    full_command = [sys.executable, "-u", *command]
    print("Running:", " ".join(full_command), flush=True)
    subprocess.run(full_command, cwd=NCTCOG_CODE_DIR, check=True)


# Execute a selected stage range and skip completed outputs by default.
def run_pipeline(start, stop, force=False):
    selected = [stage for stage in STAGES if start <= stage.number <= stop]
    if not selected:
        raise ValueError(f"No stages selected for range {start} through {stop}.")

    for stage in selected:
        print(f"\n[{stage.number:02d}] {stage.label}", flush=True)
        complete = all(path.is_file() for path in stage.outputs)
        if complete and not force:
            print("Skipped: expected output already exists.", flush=True)
            continue

        for command in stage.commands:
            command = list(command)
            if force and stage.number in {2, 10, 12}:
                command.append("--overwrite")
            run_command(command)

        missing = [str(path) for path in stage.outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Stage {stage.number} completed without expected output(s):\n"
                + "\n".join(missing)
            )

    print("\nSelected NCTCOG pipeline stages completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run or resume the DFW OSM-NCTCOG attribution workflow."
    )
    parser.add_argument("--start", type=int, default=2, choices=range(2, 13))
    parser.add_argument("--stop", type=int, default=12, choices=range(2, 13))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected stages even when their outputs exist.",
    )
    args = parser.parse_args()
    if args.start > args.stop:
        parser.error("--start cannot be greater than --stop")
    run_pipeline(args.start, args.stop, force=args.force)


if __name__ == "__main__":
    main()
