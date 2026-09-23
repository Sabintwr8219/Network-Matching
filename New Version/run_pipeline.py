import subprocess
import sys
from datetime import datetime
from pathlib import Path

from paths import OUTPUT_DIR


CODE_DIR = Path(__file__).resolve().parent
LOG_FILE = OUTPUT_DIR / "pipeline_run.log"

STAGES = [
    "01_txdot_extraction.py",
    "02_osm_extraction.py",
    "03_segment_10m.py",
    "04_freeway_spatial_matching.py",
    "05_dominant_gid_restore.py",
    "06_topology_correction.py",
    "07_gap_resolution.py",
    "08_arterial_matching.py",
    "09_combine_networks.py",
    "10_assign_district.py",
    "11_short_network_export.py",
    "12_osm_level_export.py",
]


# Compile every rewritten Python file before starting the long pipeline run.
def compile_scripts():
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(CODE_DIR)],
        cwd=CODE_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError("Compilation failed. Pipeline was not started.")


# Run one stage and copy its console output to the pipeline log.
def run_stage(stage_number, script_name, log):
    script = CODE_DIR / script_name

    if not script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script}")

    heading = f"\n[{stage_number:02d}/12] {script_name}\n"
    print(heading, end="")
    log.write(heading)
    log.flush()

    process = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=CODE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Pipeline stopped because {script_name} failed with exit code {return_code}."
        )


# Compile the code and run all twelve processing stages in sequence.
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compile_scripts()

    start_time = datetime.now()

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        message = f"Pipeline started: {start_time:%Y-%m-%d %H:%M:%S}\n"
        print(message, end="")
        log.write(message)

        for stage_number, script_name in enumerate(STAGES, start=1):
            run_stage(stage_number, script_name, log)

        end_time = datetime.now()
        elapsed = end_time - start_time

        message = (
            f"\nPipeline completed: {end_time:%Y-%m-%d %H:%M:%S}\n"
            f"Total runtime: {elapsed}\n"
            f"Log: {LOG_FILE}\n"
        )

        print(message, end="")
        log.write(message)


if __name__ == "__main__":
    main()
