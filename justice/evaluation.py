from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from .config import CostMatrix
from typing import Dict, Any
from scipy.special import expit

from .config import has_settings,get_settings ## USED ONLY FOR EDGE CASE: different costs per group

def sigmoid_np(x, clip=80.0):
    # x_clip = np.clip(x, -clip, clip)
    # test = 1.0 / (1.0 + np.exp(-x))
    
    ## better numerical precision
    result = expit(x)
    # if not np.all(np.isclose(test, result, rtol=1e-7, atol=1e-9)):
    #     raise ValueError("sigmoid implementations are not numerically close")
    return result

def expected_utility_from_costs(Z: np.ndarray, pY: np.ndarray, costs: CostMatrix, group: int = -1) -> float:
    Z = Z.astype(float); p = pY.astype(float)

    ## edge case: different costs per group
    if has_settings() and group!=-1:
        GLOBAL_SETTINGS=get_settings()
        if group==0:
            term = (costs.C_TP*Z*p
                    + costs.C_TN*(1-Z)*(1-p)
                    + costs.C_FP*Z*(1-p)
                    + costs.C_FN*(1-Z)*p)
        else:
            term = (GLOBAL_SETTINGS.customer_costs_1.C_TP*Z*p
                    + GLOBAL_SETTINGS.customer_costs_1.C_TN*(1-Z)*(1-p)
                    + GLOBAL_SETTINGS.customer_costs_1.C_FP*Z*(1-p)
                    + GLOBAL_SETTINGS.customer_costs_1.C_FN*(1-Z)*p)
    else:
        term = (costs.C_TP*Z*p
                + costs.C_TN*(1-Z)*(1-p)
                + costs.C_FP*Z*(1-p)
                + costs.C_FN*(1-Z)*p)
    return float(term.mean())

def expected_owner_utility_from_dataset(Z: np.ndarray, pY: np.ndarray, df: pd.DataFrame, owner_spec: Dict[str, Any]) -> float:
    Z = Z.astype(float)
    p = pY.astype(float)

    kind = owner_spec.get("kind", "german_credit")
    u00 = 0.0
    u01 = 0.0

    if kind == "german_credit":
        L = df[owner_spec.get("L_col", "L")].to_numpy(float)
        duration = df[owner_spec.get("duration_col", "duration")].to_numpy(float) 

        u10 = L * float(owner_spec.get("alpha_bad", 0.01))
        u11 = L * (duration / 12.0)

    elif kind == "homecredit":
        L = df[owner_spec.get("L_col", "L")].to_numpy(float)
        A = df[owner_spec.get("A_col", "A")].to_numpy(float)

        u10 = L * float(owner_spec.get("alpha_bad", 0.001))
        u11 = A * 12.0 * float(owner_spec.get("years", 15.0)) - L

    elif kind == "synthetic_with_columns":
        L = df[owner_spec.get("L_col", "L")].to_numpy(float)
        duration = df[owner_spec.get("duration_col", "duration")].to_numpy(float)
        rate = float(owner_spec.get("rate", 1))

        u10 = np.full_like(L, L.mean() * float(owner_spec.get("alpha_bad", 0.1)))
        u11 = np.full_like(L, L.mean() * duration.mean() * rate)

    else:
        raise ValueError(f"Unknown owner_spec.kind={kind}")

    exp_u = (
        Z * (p * u11 - (1-p)*u10)
    )
    return float(exp_u.mean())


def group_utilities(Z: np.ndarray, pY: np.ndarray, G: np.ndarray, costs: CostMatrix) -> Dict[object, float]:
    utils: Dict[object, float] = {}
    for g in np.unique(G):
        mask = (G == g)
        utils[g] = expected_utility_from_costs(Z[mask], pY[mask], costs, group=g)
    return utils

def evaluate_policy(
    tau: float,
    sigma: Optional[float],
    df: pd.DataFrame,
    owner_costs: CostMatrix,
    customer_costs: CostMatrix,
    rng: np.random.Generator,
    n_samples: int = 10,                 
) -> Tuple[float, Dict[object, float]]:
    """
    sigma=None  -> deterministic: Z = 1[p >= tau]
    sigma=float -> stochastic:    P(Z=1) = sigmoid( sigma * (p - tau) )
    """
    p = df["prob_Y"].to_numpy(float)
    S = df["G"].to_numpy()  

    if (sigma is None):
        Z = (p >= tau).astype(float)
        Z_batch = np.tile(Z[None, :], (n_samples, 1))
    else:
        logits = sigma * (p - tau)
        q = sigmoid_np(logits)         # [N]
        U = rng.random(size=(n_samples, p.shape[0]))        # [B,N]
        Z_batch = (U < q[None, :]).astype(float)            # [B,N]

    # utilities per sample, then average
    owner_vals = []
    group_maps = []

    for b in range(n_samples):
        Zb = Z_batch[b]
        if isinstance(owner_costs, dict):  # owner_spec dict
            owner_vals.append(expected_owner_utility_from_dataset(Zb, p, df, owner_costs))
        else:
            owner_vals.append(expected_utility_from_costs(Zb, p, owner_costs))
        group_maps.append(group_utilities(Zb, p, S, customer_costs))

    owner_u = float(np.mean(owner_vals))

    # average each group's utility across samples
    groups = list(group_maps[0].keys())
    group_u: Dict[object, float] = {
        g: float(np.mean([gm[g] for gm in group_maps])) for g in groups
    }

    return owner_u, group_u

def summarize_results(records):
    return pd.DataFrame.from_records(
        records, columns=["tau", "sigma", "owner_u", "group_utils"]
        
    )




