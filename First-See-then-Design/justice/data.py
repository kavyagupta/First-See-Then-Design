import pandas as pd

REQUIRED_COLS = ["prob_Y", "G"]

def load_dataset(path: str, prob_col: str = "prob_Y", group_col: str = "G") -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Rename key columns to canonical names
    df = df.rename(columns={prob_col: "prob_Y", group_col: "G"})

    # Check required columns
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    return df.copy()
