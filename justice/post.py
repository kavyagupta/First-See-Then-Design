import pandas as pd

def _min_group_u(group_utils: dict) -> float:
    vals = list(group_utils.values())
    return float(min(vals)) if vals else float("nan")

def _disparity(group_utils: dict) -> float:
    vals = list(group_utils.values())
    return float(max(vals) - min(vals)) if vals else float("nan")

def attach_derived_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["min_group_u"] = df["group_utils"].apply(_min_group_u)
    df["disparity"]   = df["group_utils"].apply(_disparity)
    return df
