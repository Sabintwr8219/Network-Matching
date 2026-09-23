import pandas as pd

from paths import INTERMEDIATE_DIR


FREEWAY_INPUT = INTERMEDIATE_DIR / "matched_network_basic_with_group_chain_Multiple_Fixed.csv"
ARTERIAL_INPUT = INTERMEDIATE_DIR / "TxDOT Network - Arterial.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "Augmented Network OSM-TxDOT.csv"


# Stack the completed freeway and arterial networks into one dataset.
def main():
    print("Stage 09 - Combine networks")

    freeway = pd.read_csv(FREEWAY_INPUT, low_memory=False)
    arterial = pd.read_csv(ARTERIAL_INPUT, low_memory=False)

    combined = pd.concat([freeway, arterial], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Freeway links: {len(freeway):,}")
    print(f"Arterial links: {len(arterial):,}")
    print(f"Combined links: {len(combined):,}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
