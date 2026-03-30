import numpy as np
import pandas as pd
from .post import _min_group_u, _disparity

# ------------------------------------------------------------
# Tolerant merge (unchanged)
# ------------------------------------------------------------
def _tolerant_merge(df: pd.DataFrame, pts: pd.DataFrame,
                    xcol: str, ycol: str, decimals: int = 9):

    A = df.copy()
    B = pts.copy()

    A[f"__{xcol}__r"] = A[xcol].round(decimals)
    A[f"__{ycol}__r"] = A[ycol].round(decimals)
    B[f"__{xcol}__r"] = B[xcol].round(decimals)
    B[f"__{ycol}__r"] = B[ycol].round(decimals)

    merged = A.merge(
        B[[f"__{xcol}__r", f"__{ycol}__r"]].drop_duplicates(),
        on=[f"__{xcol}__r", f"__{ycol}__r"],
        how="inner",
        validate="m:m",
    )

    return merged.drop(columns=[f"__{xcol}__r", f"__{ycol}__r"])

# ------------------------------------------------------------
# Egalitarian Pareto front (MAX owner, MIN regulator cost)
# Utopia: bottom-right
# ------------------------------------------------------------
def get_pareto_front_egal(points: np.ndarray) -> np.ndarray:
    """
    points[:,0] = owner utility (maximize)
    points[:,1] = regulator cost (disparity) (minimize)
    """
    # owner desc, disparity asc
    sorted_points = points[np.lexsort((points[:, 1], -points[:, 0]))]

    pareto = []
    min_cost = np.inf
    for owner, cost in sorted_points:
        if cost < min_cost:
            pareto.append([owner, cost])
            min_cost = cost

    return np.array(pareto)

def pareto_front_egalitarian(results_df: pd.DataFrame):
    x = results_df["owner_u"].to_numpy(float)
    disparity = results_df["group_utils"].apply(_disparity).to_numpy(float)

    pf = get_pareto_front_egal(np.stack([x, disparity], axis=1))
    pf_points = pd.DataFrame(pf, columns=["owner_u", "regulator_cost"])

    tmp = results_df.copy()
    tmp["regulator_cost"] = disparity
    pf_rows = _tolerant_merge(
        tmp, pf_points,
        xcol="owner_u",
        ycol="regulator_cost"
    )

    return pf_points, pf_rows


def get_pareto_front_df_egal(
    df: pd.DataFrame, 
    maximize_col: str, 
    minimize_col: str
) -> pd.DataFrame:
    """
    Finds the Pareto front in a DataFrame based on two objectives:
    one to maximize and one to minimize.

    Args:
        df: The input DataFrame containing all points/models.
        maximize_col: The name of the column to maximize (e.g., 'Accuracy').
        minimize_col: The name of the column to minimize (e.g., 'Fairness_Disparity').

    Returns:
        A DataFrame containing only the rows that lie on the Pareto front.
    """
    
    # 1. Sort the DataFrame
    #    - We sort primarily by the maximization column in DESCENDING order.
    #    - We sort secondarily by the minimization column in ASCENDING order.
    sorted_df = df.sort_values(
        by=[maximize_col, minimize_col], 
        ascending=[False, True]
    ).reset_index(drop=True)

    pareto_rows = []
    
    # Initialize the minimum value seen so far for the minimization objective
    min_objective_val = np.inf

    # 2. Iterate and Check for Pareto Dominance
    for _, row in sorted_df.iterrows():
        current_min_val = row[minimize_col]

        # Check for Pareto dominance:
        # A point is on the Pareto front if its minimization objective value is 
        # strictly better (lower) than the minimum value seen so far among points 
        # with equal or higher maximization values.
        if current_min_val < min_objective_val:
            # If it's Pareto optimal, append the entire row (Series)
            pareto_rows.append(row)
            
            # Update the minimum value seen
            min_objective_val = current_min_val

    # 3. Construct the final Pareto front DataFrame
    if pareto_rows:
        return pd.DataFrame(pareto_rows).reset_index(drop=True)
    else:
        # Handle the case where the input DataFrame might be empty or problematic
        return pd.DataFrame(columns=df.columns)




# ------------------------------------------------------------
# Rawlsian Pareto front (MAX owner, MAX regulator utility)
# Utopia: top-right
# ------------------------------------------------------------
def get_pareto_front_rawls(points: np.ndarray) -> np.ndarray:
    """
    points[:,0] = owner utility (maximize)
    points[:,1] = regulator utility (min-group utility) (maximize)
    """
    # owner desc, rawls desc
    sorted_points = points[np.lexsort((-points[:, 1], -points[:, 0]))]

    pareto = []
    max_rawls = -np.inf
    for owner, rawls in sorted_points:
        if rawls > max_rawls:
            pareto.append([owner, rawls])
            max_rawls = rawls

    return np.array(pareto)

def pareto_front_rawlsian(results_df: pd.DataFrame):
    x = results_df["owner_u"].to_numpy(float)
    min_group = results_df["group_utils"].apply(_min_group_u).to_numpy(float)

    pf = get_pareto_front_rawls(np.stack([x, min_group], axis=1))
    pf_points = pd.DataFrame(pf, columns=["owner_u", "regulator_utility"])

    tmp = results_df.copy()
    tmp["regulator_utility"] = min_group
    pf_rows = _tolerant_merge(
        tmp, pf_points,
        xcol="owner_u",
        ycol="regulator_utility"
    )

    return pf_points, pf_rows


def get_pareto_front_df_rawls(
    df: pd.DataFrame,
    maximize_col_1: str,
    maximize_col_2: str
) -> pd.DataFrame:
    """
    Finds the Pareto front in a DataFrame based on two objectives,
    both of which are to be maximized.

    Args:
        df: The input DataFrame containing all points/models.
        maximize_col_1: The name of the first column to maximize.
        maximize_col_2: The name of the second column to maximize.

    Returns:
        A DataFrame containing only the rows that lie on the Pareto front.
    """

    # 1. Sort by both objectives in descending order
    sorted_df = df.sort_values(
        by=[maximize_col_1, maximize_col_2],
        ascending=[False, False]
    ).reset_index(drop=True)

    pareto_rows = []

    # Track the maximum value seen so far for the second objective
    max_objective_val = -np.inf

    # 2. Iterate and check for Pareto optimality
    for _, row in sorted_df.iterrows():
        current_val = row[maximize_col_2]

        # A point is Pareto optimal if it strictly improves
        # the second objective among points with >= first objective
        if current_val > max_objective_val:
            pareto_rows.append(row)
            max_objective_val = current_val

    # 3. Construct output DataFrame
    if pareto_rows:
        return pd.DataFrame(pareto_rows).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns)


####### debugging code #######
def compare_df_to_numpy_pf(
    df: pd.DataFrame,
    cols: tuple,
    pf_np: np.ndarray,
    atol: float = 1e-4,
) -> bool:
    """
    Compare two DataFrame columns to a NumPy Pareto front array.

    Args:
        df: Pareto-front DataFrame.
        cols: Tuple of column names to compare (in order).
        pf_np: NumPy array of shape (n, 2).
        atol: Absolute tolerance for floating point comparison.

    Returns:
        True if they represent the same set of points, else False.
    """
    if pf_np.ndim != 2 or pf_np.shape[1] != 2:
        raise ValueError("pf_np must be of shape (n, 2)")

    df_vals = df.loc[:, cols].to_numpy(dtype=float)

    if len(df_vals) != len(pf_np):
        return False

    # Sort both lexicographically for order-invariant comparison
    df_sorted = df_vals[np.lexsort((df_vals[:, 1], df_vals[:, 0]))]
    pf_sorted = pf_np[np.lexsort((pf_np[:, 1], pf_np[:, 0]))]

    return np.allclose(df_sorted, pf_sorted, atol=atol)
