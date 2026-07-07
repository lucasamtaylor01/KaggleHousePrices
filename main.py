from pathlib import Path
import pandas as pd
from scripts.cleaning import cleaning

# PATH DEFINITIONS

BASE_DIR = Path().resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "data_raw"

CLEAN_DATA_DIR = DATA_DIR / "data_clean"
CLEAN_DATA_DIR.mkdir(exist_ok=True)

def main():

    df_train_raw = pd.read_csv(RAW_DATA_DIR / "train.csv")
    df_test_raw = pd.read_csv(RAW_DATA_DIR / "test.csv")

    print("\nSTARTING DATA CLEANING PROCESS...\n\n")
    df_train_clean = cleaning(df_train_raw, 0)
    df_train_clean.to_csv(CLEAN_DATA_DIR / 'train_clean.csv', index=False)

    df_test_clean = cleaning(df_test_raw, 1)
    df_test_clean.to_csv(CLEAN_DATA_DIR / 'test_clean.csv', index=False)

if __name__ == "__main__":
    main()