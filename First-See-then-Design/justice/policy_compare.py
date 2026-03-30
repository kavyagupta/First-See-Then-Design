import os, json, itertools
import numpy as np
import pandas as pd
from typing import Optional, Sequence, List, Tuple, Dict, Any
from .evaluation import evaluate_policy
from .pareto import get_pareto_front_egal, get_pareto_front_rawls, get_pareto_front_df_egal, get_pareto_front_df_rawls,compare_df_to_numpy_pf
from .post import attach_derived_metrics
from scripts.metrics_utils import (
    compute_hypervolume,
    compute_hypervolume_rawls,
    stochastic_gain,
    average_stochastic_gain,
)

def get_common_nadir(pf_list):
    """
    Compute a common nadir point for hypervolume.
    Assumes canonical form: maximize x, minimize y.
    """
    xs = np.concatenate([pf[:, 0] for pf in pf_list])
    ys = np.concatenate([pf[:, 1] for pf in pf_list])
    return xs.min()- 0.05, ys.max()+ 0.05  # x_nadir, y_nadir
    # return xs.min(), ys.max() # x_nadir, y_nadir

def get_common_utopia(pf_list):
    """
    Compute a common utopia point for hypervolume.
    Assumes canonical form: maximize x, minimize y.
    """
    xs = np.concatenate([pf[:, 0] for pf in pf_list])
    ys = np.concatenate([pf[:, 1] for pf in pf_list])
    return xs.max(), ys.min() 



def get_common_nadir_rawls(pf_list):
    """
    Compute a common nadir point for hypervolume.
    Assumes canonical form: maximize x, maximize y.
    """
    xs = np.concatenate([pf[:, 0] for pf in pf_list])
    ys = np.concatenate([pf[:, 1] for pf in pf_list])
    return xs.min()- 0.05, ys.min()- 0.05  # x_nadir, y_nadir
    # return xs.min(), ys.min()  # x_nadir, y_nadir

def get_common_utopia_rawls(pf_list):
    """
    Compute a common utopia point for hypervolume.
    Assumes canonical form: maximize x, maximize y.
    """
    xs = np.concatenate([pf[:, 0] for pf in pf_list])
    ys = np.concatenate([pf[:, 1] for pf in pf_list])
    return xs.max(), ys.max()


def compute_normalized_hv(hv, ref_point, utopia_point):
    max_hv = np.prod(np.abs(np.array(utopia_point) - np.array(ref_point)))
    return hv / max_hv


def _weighted_mean_owner(owner_vals: List[float], group_sizes: List[int]) -> float:
    total = float(sum(group_sizes))
    if total == 0:
        return float(np.mean(owner_vals))
    return float(np.dot(owner_vals, np.array(group_sizes, dtype=float)) / total)

def _all_pairs(taus: np.ndarray, sigmas: Sequence[Optional[float]]) -> List[Tuple[float, Optional[float]]]:
    """List of (tau, sigma) pairs for one group."""
    out: List[Tuple[float, Optional[float]]] = []
    for t in taus:
        for s in sigmas:
            out.append((float(t), (None if s is None else float(s))))
    return out

def _sample_products(
    per_group_pairs: List[List[Tuple[float, Optional[float]]]],
    max_combos: int,
    rng: np.random.Generator
):
    """Randomly sample up to max_combos combos from the Cartesian product of per-group pairs."""
    lens = [len(pg) for pg in per_group_pairs]
    total = 1
    for L in lens:
        total *= L
    if total <= max_combos:
        for combo in itertools.product(*per_group_pairs):
            yield combo
        return
    # sampling
    for _ in range(max_combos):
        combo = tuple(pg[int(rng.integers(0, len(pg)))] for pg in per_group_pairs)
        yield combo

def sweep_policies(df, settings, rng, stochastic: bool = False, group_specific: bool = False, max_combos: int = 10000):
    """
    - Shared deterministic:           one tau, sigma=None
    - Shared stochastic:              one tau, sigma in settings.sigmas
    - Group-specific deterministic:   tau^g per group with sigma^g=None
    - Group-specific stochastic:      tau^g per group with sigma^g in settings.sigmas
    For G groups, group-specific search is sampled if the full grid is too large.
    """
    taus = np.linspace(0.1, 0.99, settings.thresholds)
    sigmas = settings.sigmas if stochastic else [None]

    # groups = list(pd.unique(df["G"]))  # labels can be str/int
    groups = sorted(pd.unique(df["G"]))

    G = len(groups)

    records = []
    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(settings.seed)

    if not group_specific:
        # SHARED policy
        for sigma in sigmas:
            for tau in taus:
                owner_u, group_u = evaluate_policy(
                    tau=tau, sigma=sigma, df=df,
                    owner_costs=settings.owner_costs,
                    customer_costs=settings.customer_costs,
                    rng=rng_local
                )
                records.append({
                    "policy": "shared",
                    "stochastic": (sigma is not None),
                    "tau": float(tau),
                    "sigma": (None if sigma is None else float(sigma)),
                    "owner_u": owner_u,
                    "group_utils": group_u,
                })
    else:
    
        per_group: List[List[Tuple[float, Optional[float]]]] = []
        for _ in range(G):
            per_group.append(_all_pairs(taus, sigmas))

        iterator = _sample_products(per_group, max_combos=max_combos, rng=rng_local)

        for combo in iterator:
            owner_vals: List[float] = []
            group_sizes: List[int] = []
            merged_gu = {}
            for g_label, (tau_g, sigma_g) in zip(groups, combo):
                sub = df.loc[df["G"] == g_label]
                o, gu = evaluate_policy(
                    tau=tau_g, sigma=sigma_g, df=sub,
                    owner_costs=settings.owner_costs,
                    customer_costs=settings.customer_costs, rng=rng_local
                )
                owner_vals.append(o)
                group_sizes.append(len(sub))
                merged_gu[g_label] = gu[g_label]

            owner_u = _weighted_mean_owner(owner_vals, group_sizes)

            records.append({
                "policy": "group-specific",
                "stochastic": any(s is not None for (_, s) in combo),
                "tau_vec": json.dumps([t for (t, _) in combo]),
                "sigma_vec": json.dumps([s for (_, s) in combo]),
                "owner_u": float(owner_u),
                "group_utils": merged_gu,
            })

    dfres = pd.DataFrame(records)
    return attach_derived_metrics(dfres)


def compare_pf_including_rawls(df, settings, rng, outdir, max_combos: int = 1000000):
    """Based on compare_pf. Basically repeated the metric calculations, storing of files etc."""
    os.makedirs(outdir, exist_ok=True)
    egal_dir = os.path.join(outdir, "all_policies")
    os.makedirs(egal_dir, exist_ok=True)
    combos = [
        ("shared", False),
        ("shared", True),
        ("group-specific", False),
        ("group-specific", True),
    ]
    pf_store_egal = {}
    pf_store_rawls = {}

    for name, stoch in combos:
        res = sweep_policies(
            df, settings, rng,
            stochastic=stoch,
            group_specific=(name=="group-specific"),
            max_combos=max_combos
        )

        res.to_csv(
        os.path.join(egal_dir, f"train_all_points_{name}_{'stoch' if stoch else 'det'}.csv"),
        index=False,
    )

        # Egalitarian PF
        X = res["owner_u"].to_numpy(float)
        Y_egal = res["disparity"].to_numpy(float)
        Y_rawls = res["min_group_u"].to_numpy(float)

        ### GET THE ROWS OF THE DATAFRAME THAT ARE THE PARETO FRONT POLICIES
        df_pf_egal=get_pareto_front_df_egal(res,"owner_u","disparity")
        df_pf_rawls=get_pareto_front_df_rawls(res,"owner_u","min_group_u")
        

        ### NUMPY - BASED PARETO FRONTS, USED LATER FOR METRICS
        np_pf_egal = get_pareto_front_egal(np.stack([X, Y_egal], axis=1))
        np_pf_rawls = get_pareto_front_rawls(np.stack([X, Y_rawls], axis=1))
        test_a=compare_df_to_numpy_pf(df_pf_egal,("owner_u","disparity"),np_pf_egal)
        test_b=compare_df_to_numpy_pf(df_pf_rawls,("owner_u","min_group_u"),np_pf_rawls)
        if not (test_a and test_b):
            raise ValueError("dataframe-based PF, and numpy-array PF are different!")
        ###

        key = f"{name}_{'stoch' if stoch else 'det'}"
        pf_store_egal[key] = np_pf_egal
        pf_store_rawls[key] = np_pf_rawls

        pd.DataFrame(df_pf_egal).to_csv(f"{outdir}/train_pf_egal_{key}.csv", index=False)

        pd.DataFrame(df_pf_rawls).to_csv(f"{outdir}/train_pf_rawls_{key}.csv", index=False)
    

    #### CALCULATE POLICY-WIDE METRICS (based on numpy)
    for fairness_type in ["egal","rawls"]:
        if fairness_type=="egal":
            pf_store=pf_store_egal
            pf_list = list(pf_store.values())
            nadir = get_common_nadir(pf_list)
            utopia = get_common_utopia(pf_list)
        else:
            pf_store=pf_store_rawls
            pf_list = list(pf_store.values())
            nadir = get_common_nadir_rawls(pf_list)        
            utopia = get_common_utopia_rawls(pf_list)

        hv = {}
        for key, pf in pf_store.items():
            if fairness_type=="egal":
                hv[key] = compute_hypervolume(pf, nadir)
            else:
                hv[key] = compute_hypervolume_rawls(pf, nadir)
            hv["normalized_"+key] = compute_normalized_hv(hv[key],nadir,utopia)
        with open(f"{outdir}/train_hypervolume_{fairness_type}.json", "w") as f:
            json.dump(hv, f, indent=2)


        gain_summary = {}

        # Shared policies
        x_shared, gain_shared = stochastic_gain(
            pf_store["shared_det"],
            pf_store["shared_stoch"],
            fairness_type=fairness_type,
        )

        # Group-specific policies
        x_group, gain_group = stochastic_gain(
            pf_store["group-specific_det"],
            pf_store["group-specific_stoch"],
            fairness_type=fairness_type,
        )

        gain_summary["shared_avg_gain"] = float(np.mean(gain_shared))
        gain_summary["group_avg_gain"] = float(np.mean(gain_group))

        np.savez(
            f"{outdir}/train_stochastic_gain_{fairness_type}.npz",
            x_shared=x_shared,
            gain_shared=gain_shared,
            x_group=x_group,
            gain_group=gain_group,
        )

        with open(f"{outdir}/train_stochastic_gain_summary_{fairness_type}.json", "w") as f:
            json.dump(gain_summary, f, indent=2)

        avg_gain_shared = average_stochastic_gain(
        pf_store["shared_det"],
        pf_store["shared_stoch"],
        fairness_type=fairness_type,
    )

        # Group-specific policy
        avg_gain_group = average_stochastic_gain(
            pf_store["group-specific_det"],
            pf_store["group-specific_stoch"],
            fairness_type=fairness_type,
        )

        avg_gain_summary = {
        "shared": avg_gain_shared,
        "group-specific": avg_gain_group,
    }

        with open(os.path.join(outdir, f"train_avg_stochastic_gain_{fairness_type}.json"), "w") as f:
            json.dump(avg_gain_summary, f, indent=2)


######### TEST TIME #########

def get_utilities_of_specific_policies(
    df: pd.DataFrame,
    policy_df: pd.DataFrame,
    settings,
    rng,
):
    """
    Replaces sweep policies function.
    Evaluate a given DataFrame of policies (shared or group-specific).

    policy_df must contain:
      - policy == "shared" with columns: tau, sigma
      - policy == "group-specific" with columns: tau_vec, sigma_vec (JSON lists)

    Returns a DataFrame identical in structure to sweep_policies output.
    """
    # groups = list(pd.unique(df["G"]))
    groups = sorted(pd.unique(df["G"]))
    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(settings.seed)

    records = []

    for _, row in policy_df.iterrows():
        if row["policy"] == "shared":
            tau = float(row["tau"])
            sigma = None if pd.isna(row["sigma"]) else float(row["sigma"])

            owner_u, group_u = evaluate_policy(
                tau=tau,
                sigma=sigma,
                df=df,
                owner_costs=settings.owner_costs,
                customer_costs=settings.customer_costs,
                rng=rng_local,
            )

            records.append({
                "policy": "shared",
                "stochastic": (sigma is not None),
                "tau": tau,
                "sigma": sigma,
                "owner_u": float(owner_u),
                "group_utils": group_u,
            })

        elif row["policy"] == "group-specific":
            tau_vec = json.loads(row["tau_vec"])
            sigma_vec = json.loads(row["sigma_vec"])

            owner_vals = []
            group_sizes = []
            merged_gu = {}

            for g_label, tau_g, sigma_g in zip(groups, tau_vec, sigma_vec):
                sub = df.loc[df["G"] == g_label]

                sigma_g = None if sigma_g is None else float(sigma_g)

                o, gu = evaluate_policy(
                    tau=tau_g,
                    sigma=sigma_g,
                    df=sub,
                    owner_costs=settings.owner_costs,
                    customer_costs=settings.customer_costs,
                    rng=rng_local,
                )

                owner_vals.append(o)
                group_sizes.append(len(sub))
                merged_gu[g_label] = gu[g_label]

            owner_u = _weighted_mean_owner(owner_vals, group_sizes)

            records.append({
                "policy": "group-specific",
                "stochastic": any(s is not None for s in sigma_vec),
                "tau_vec": json.dumps(tau_vec),
                "sigma_vec": json.dumps(sigma_vec),
                "owner_u": float(owner_u),
                "group_utils": merged_gu,
            })

        else:
            raise ValueError(f"Unknown policy type: {row['policy']}")

    dfres = pd.DataFrame(records)
    return attach_derived_metrics(dfres)

def compute_nadir_utopia_from_dict_of_policies(all_types_of_train_pf_policies):
    """
    Compute common nadir and utopia points from training Pareto-front policies.

    Parameters
    ----------
    all_types_of_train_pf_policies : dict[str, pd.DataFrame]
        Keys like:
        'egal_group-specific_det', 'rawls_shared_stoch', ...
        Values are dataframes containing at least:
          - 'owner_u'
          - 'disparity' (egal)
          - 'min_group_u' (rawls)

    Returns
    -------
    nadir_egal : np.ndarray or None
    utopia_egal : np.ndarray or None
    nadir_rawls : np.ndarray or None
    utopia_rawls : np.ndarray or None
    """

    pf_list_egal = []
    pf_list_rawls = []

    for key, df in all_types_of_train_pf_policies.items():
        justice = key.split("_", 1)[0]

        X = df["owner_u"].to_numpy(float)

        if justice == "egal":
            Y = df["disparity"].to_numpy(float)
            pf = get_pareto_front_egal(np.stack([X, Y], axis=1))
            pf_list_egal.append(pf)

        elif justice == "rawls":
            Y = df["min_group_u"].to_numpy(float)
            pf = get_pareto_front_rawls(np.stack([X, Y], axis=1))
            pf_list_rawls.append(pf)

    nadir_egal = utopia_egal = None
    nadir_rawls = utopia_rawls = None

    if pf_list_egal:
        nadir_egal = get_common_nadir(pf_list_egal)
        utopia_egal = get_common_utopia(pf_list_egal)

    if pf_list_rawls:
        nadir_rawls = get_common_nadir_rawls(pf_list_rawls)
        utopia_rawls = get_common_utopia_rawls(pf_list_rawls)

    return nadir_egal, utopia_egal, nadir_rawls, utopia_rawls


def calculate_test_time_metrics(df, settings, rng, outdir, all_types_of_train_pf_policies,fairness_type):
    """Based on compare_pf_including_rawls."""
    os.makedirs(outdir, exist_ok=True)
    test_dir = os.path.join(outdir, "all_policies")
    os.makedirs(test_dir, exist_ok=True)
    pf_store = {}

    for key_name, train_pf_policies in all_types_of_train_pf_policies.items():

        name=train_pf_policies["policy"][0]
        stoch=train_pf_policies["stochastic"][0]
        justice= key_name.split("_", 1)[0]
        assert justice==fairness_type


        res = get_utilities_of_specific_policies(df, train_pf_policies,settings, rng)

        res.to_csv(
            os.path.join(test_dir, f"test_from_{fairness_type}_PF_all_points_{name}_{'stoch' if stoch else 'det'}.csv"),
            index=False,
        )

        key = f"{name}_{'stoch' if stoch else 'det'}"
        # Egalitarian PF
        X = res["owner_u"].to_numpy(float)
        if fairness_type=="egal":
            Y_egal = res["disparity"].to_numpy(float)
            df_pf=get_pareto_front_df_egal(res,"owner_u","disparity")
            np_pf = get_pareto_front_egal(np.stack([X, Y_egal], axis=1))
            pf_store[key] = np_pf
            test=compare_df_to_numpy_pf(df_pf,("owner_u","disparity"),np_pf)
        else:
            Y_rawls = res["min_group_u"].to_numpy(float)
            df_pf=get_pareto_front_df_rawls(res,"owner_u","min_group_u")
            np_pf = get_pareto_front_rawls(np.stack([X, Y_rawls], axis=1))
            pf_store[key] = np_pf
            test=compare_df_to_numpy_pf(df_pf,("owner_u","min_group_u"),np_pf)

        pd.DataFrame(df_pf).to_csv(f"{outdir}/test_pf_{fairness_type}_{key}.csv", index=False)
        if not test:
            raise ValueError("dataframe-based PF, and numpy-array PF are different!")

    #### CALCULATE POLICY-WIDE METRICS (based on numpy)
    # if you want to find nadir/utopia during test time
    pf_list = list(pf_store.values())
    if fairness_type=="egal":            
        nadir = get_common_nadir(pf_list)
        utopia = get_common_utopia(pf_list)
    else:
        nadir = get_common_nadir_rawls(pf_list)        
        utopia = get_common_utopia_rawls(pf_list)
    ############
    # if you want to use utopia and nadir from train time
    # nadir_egal, utopia_egal, nadir_rawls, utopia_rawls=compute_nadir_utopia_from_dict_of_policies(all_types_of_train_pf_policies)
    # if fairness_type=="egal":            
    #     nadir = nadir_egal
    #     utopia = utopia_egal
    # else:
    #     nadir = nadir_rawls
    #     utopia = utopia_rawls

    hv = {}
    for key, pf in pf_store.items():
        if fairness_type=="egal":
            hv[key] = compute_hypervolume(pf, nadir)
        else:
            hv[key] = compute_hypervolume_rawls(pf, nadir)
        hv["normalized_"+key] = compute_normalized_hv(hv[key],nadir,utopia)

    with open(f"{outdir}/test_hypervolume_{fairness_type}.json", "w") as f:
        json.dump(hv, f, indent=2)


    gain_summary = {}

    # Shared policies
    x_shared, gain_shared = stochastic_gain(
        pf_store["shared_det"],
        pf_store["shared_stoch"],
        fairness_type=fairness_type,
    )

    # Group-specific policies
    x_group, gain_group = stochastic_gain(
        pf_store["group-specific_det"],
        pf_store["group-specific_stoch"],
        fairness_type=fairness_type,
    )

    gain_summary["shared_avg_gain"] = float(np.mean(gain_shared))
    gain_summary["group_avg_gain"] = float(np.mean(gain_group))

    np.savez(
        f"{outdir}/test_stochastic_gain_{fairness_type}.npz",
        x_shared=x_shared,
        gain_shared=gain_shared,
        x_group=x_group,
        gain_group=gain_group,
    )

    with open(f"{outdir}/test_stochastic_gain_summary_{fairness_type}.json", "w") as f:
        json.dump(gain_summary, f, indent=2)

    avg_gain_shared = average_stochastic_gain(
    pf_store["shared_det"],
    pf_store["shared_stoch"],
    fairness_type=fairness_type,
)

    # Group-specific policy
    avg_gain_group = average_stochastic_gain(
        pf_store["group-specific_det"],
        pf_store["group-specific_stoch"],
        fairness_type=fairness_type,
    )

    avg_gain_summary = {
    "shared": avg_gain_shared,
    "group-specific": avg_gain_group,
}

    with open(os.path.join(outdir, f"test_avg_stochastic_gain_{fairness_type}.json"), "w") as f:
        json.dump(avg_gain_summary, f, indent=2)
