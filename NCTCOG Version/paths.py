import os
from pathlib import Path


# Find the repository root from the .git folder.
def find_project_root():
    configured = os.environ.get("NCTCOG_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"NCTCOG_PROJECT_ROOT does not exist: {root}"
            )
        return root

    current = Path(__file__).resolve().parent

    for folder in [current, *current.parents]:
        if (folder / ".git").exists():
            return folder

    raise FileNotFoundError(
        "Could not find the repository root. Run inside the Git repository "
        "or set NCTCOG_PROJECT_ROOT."
    )


PROJECT_ROOT = find_project_root()

NCTCOG_CODE_DIR = PROJECT_ROOT / "NCTCOG code"

NCTCOG_INPUT = (
    PROJECT_ROOT / "Initial Input Files" / "NCTCOG" / "links_NCTCOG.csv"
)
DISTRICT_INPUT_DIR = (
    PROJECT_ROOT
    / "Output"
    / "New Version"
    / "Final"
    / "OSM_Short_Level_CSV"
    / "District_Link_List"
)
DISTRICT_OSM_INPUTS = [
    DISTRICT_INPUT_DIR / "Dallas_Link_List.csv",
    DISTRICT_INPUT_DIR / "Fort_Worth_Link_List.csv",
    DISTRICT_INPUT_DIR / "Bryan_Link_List.csv",
    DISTRICT_INPUT_DIR / "Paris_Link_List.csv",
    DISTRICT_INPUT_DIR / "Tyler_Link_List.csv",
    DISTRICT_INPUT_DIR / "Waco_Link_List.csv",
    DISTRICT_INPUT_DIR / "Wichita_Falls_Link_List.csv",
]
PREPARED_INPUT_DIR = NCTCOG_CODE_DIR / "Prepared Inputs"
OSM_INPUT = (
    PREPARED_INPUT_DIR / "DFW_OSM_short_links_only.csv"
)

OUTPUT_DIR = NCTCOG_CODE_DIR / "Output"
INTERMEDIATE_DIR = OUTPUT_DIR / "Intermediate"
FINAL_DIR = OUTPUT_DIR / "Final"

NCTCOG_PREPARED_FILE = INTERMEDIATE_DIR / "NCTCOG_prepared_links.csv"
NCTCOG_COVERAGE_FILE = INTERMEDIATE_DIR / "NCTCOG_direct_length_coverage.parquet"
OSM_ASSIGNMENT_FILE = (
    INTERMEDIATE_DIR / "OSM_link_matches_after_reverse_pairs.parquet"
)
FINAL_NETWORK_FILE = FINAL_DIR / "DFW_OSM_with_NCTCOG_attributes.parquet"
FINAL_NETWORK_CSV = FINAL_DIR / "DFW_OSM_with_NCTCOG_attributes.csv"
FINAL_VALIDATION_FILE = FINAL_DIR / "DFW_OSM_NCTCOG_validation.json"


# Create the rewritten output folders.
def create_output_folders():
    for folder in [OUTPUT_DIR, INTERMEDIATE_DIR, FINAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# Check that all required source files exist.
def check_inputs():
    files = [
        NCTCOG_INPUT,
        OSM_INPUT,
    ]

    missing = [file for file in files if not file.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" +
            "\n".join(str(file) for file in missing)
        )


create_output_folders()
