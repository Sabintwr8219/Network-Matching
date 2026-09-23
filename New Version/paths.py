from pathlib import Path


# Find the repository root from the .git folder.
def find_project_root():
    current = Path(__file__).resolve().parent

    for folder in [current, *current.parents]:
        if (folder / ".git").exists():
            return folder

    raise FileNotFoundError("Could not find project root containing .git")


PROJECT_ROOT = find_project_root()

INPUT_DIR = PROJECT_ROOT / "Initial Input Files"
TXDOT_INPUT_DIR = INPUT_DIR / "TxDOT"
OSM_INPUT_DIR = INPUT_DIR / "OSM"

OUTPUT_DIR = PROJECT_ROOT / "Output" / "New Version"
INTERMEDIATE_DIR = OUTPUT_DIR / "Intermediate"
FINAL_DIR = OUTPUT_DIR / "Final"

TXDOT_DIR = INTERMEDIATE_DIR / "TxDOT"
OSM_DIR = INTERMEDIATE_DIR / "OSM"
MOTORWAY_DIR = INTERMEDIATE_DIR / "Motorway"
TRUNK_DIR = INTERMEDIATE_DIR / "Trunk"


# Raw input files
TXDOT_ROADWAY = TXDOT_INPUT_DIR / "TxDOT_Roadways_3004912656256173571.txt"
TXDOT_SPEED = TXDOT_INPUT_DIR / "TxDOT_Speed_Limits_-8028908546543462748.txt"
TXDOT_LANES = TXDOT_INPUT_DIR / "TxDOT_Number_of_Through_Lanes_-7291081630889162709.txt"
TXDOT_FC = TXDOT_INPUT_DIR / "TxDOT_Functional_Classification_8550865030343399182.txt"

TXDOT_DISTRICTS = (
    TXDOT_INPUT_DIR
    / "TxDOT_Districts_5095849743065131776"
    / "District_Boundaries.shp"
)

OSM_PBF = OSM_INPUT_DIR / "texas-260715.osm.pbf"


# Create the rewritten output folders.
def create_output_folders():
    for folder in [
        OUTPUT_DIR,
        INTERMEDIATE_DIR,
        FINAL_DIR,
        TXDOT_DIR,
        OSM_DIR,
        MOTORWAY_DIR,
        TRUNK_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


# Check that all required source files exist.
def check_inputs():
    files = [
        TXDOT_ROADWAY,
        TXDOT_SPEED,
        TXDOT_LANES,
        TXDOT_FC,
        TXDOT_DISTRICTS,
        OSM_PBF,
    ]

    missing = [file for file in files if not file.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" +
            "\n".join(str(file) for file in missing)
        )


create_output_folders()