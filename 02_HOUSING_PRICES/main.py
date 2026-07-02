from pathlib import Path

import pandas as pd

from scripts.data_cleaning import clean_data

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "data_raw"
CLEAN_DATA_DIR = DATA_DIR / "data_clean"


def main() -> None:
    CLEAN_DATA_DIR.mkdir(exist_ok=True)

    for name in ("train", "test"):
        print(f"--- Cleaning {name}.csv ---")
        df_raw = pd.read_csv(RAW_DATA_DIR / f"{name}.csv")
        df_clean = clean_data(df_raw)
        out_path = CLEAN_DATA_DIR / f"{name}_clean.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"Saved {df_clean.shape[0]} rows, {df_clean.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    main()
