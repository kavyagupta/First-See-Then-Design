import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from justice import load_config, load_dataset
from justice.policy_compare import compare_pf_including_rawls, calculate_test_time_metrics



def plot_pf_overlay(outdir, justice):
    """
    Plot 4 Pareto fronts for a given justice notion.
    """
    plt.figure(figsize=(6, 4))

    base = os.path.join(outdir, justice)
    curves = [
        ("Shared (Deterministic)",         "pf_shared_det.csv"),
        ("Shared (Stochastic)",            "pf_shared_stoch.csv"),
        ("Group-specific (Deterministic)", "pf_group_det.csv"),
        ("Group-specific (Stochastic)",    "pf_group_stoch.csv"),
    ]

    for label, fname in curves:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing {path}")
            continue

        df = pd.read_csv(path).sort_values("owner_u")

        if justice == "egalitarian":
            y = df["disparity"]
            ylabel = "Disparity (↓)"
        else:  # rawlsian
            y = df["min_group_u"]
            ylabel = "Min-group utility (↑)"

        plt.plot(df["owner_u"], y, label=label, linewidth=2)

    plt.xlabel("Owner utility (↑)")
    plt.ylabel(ylabel)
    plt.title(f"Pareto Fronts ({justice.capitalize()})")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()

    fname = f"pareto_{justice}.pdf"
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def plot_stochastic_gain(outdir, justice):
    """
    Plot stochastic fairness gain over deterministic baseline.
    """
    path = os.path.join(outdir, f"stochastic_gain_{justice}.npz")
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}")
        return

    data = np.load(path)

    plt.figure(figsize=(6, 4))

    plt.plot(
        data["x_shared"],
        data["gain_shared"],
        label="Shared policy",
        linewidth=2,
        color="green",
    )

    plt.plot(
        data["x_group"],
        data["gain_group"],
        label="Group-specific policy",
        linewidth=2,
        color="red",
    )

    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Owner utility (↑)")
    ylabel = (
        "Fairness gain (↓ disparity)"
        if justice == "egalitarian"
        else "Fairness gain (↑ min-group utility)"
    )
    plt.ylabel(ylabel)

    plt.title(f"Stochastic Gain over Deterministic ({justice.capitalize()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"stochastic_gain_{justice}.pdf"
    plt.savefig(os.path.join(outdir, fname))
    plt.close()



def load_train_pf_dfs(folder: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load all train_pf_*.csv files in a folder into two dictionaries:
    - out_egal: files whose name contains 'egal'
    - out_rawls: files whose name contains 'rawls'

    Dictionary keys are everything after 'train_pf_' in the filename.

    Example:
        train_pf_egal_foo.csv   -> out_egal['egal_foo']
        train_pf_rawls_bar.csv  -> out_rawls['rawls_bar']
    """
    folder = Path(folder)
    out_egal = {}
    out_rawls = {}

    for csv_path in folder.glob("train_pf_*.csv"):
        key = csv_path.stem.replace("train_pf_", "", 1)

        if "egal" in key:
            out_egal[key] = pd.read_csv(csv_path)
        elif "rawls" in key:
            out_rawls[key] = pd.read_csv(csv_path)

    return out_egal, out_rawls



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--outdir", default="viz_out")
    parser.add_argument("--max_combos", type=int, default=1_000_000)
    args = parser.parse_args()

    # Load experiment setup
    settings = load_config(args.config)

    ### USED ONLY FOR EDGE CASE: different costs per group
    if settings.customer_costs_1 is not None:
        from justice.config import set_settings
        set_settings(settings)
    ###

    rng = np.random.default_rng(settings.seed)
    df = load_dataset(args.data)
    test_df = load_dataset(args.test_data)


    # compare_pf(
    #     df,
    #     settings,
    #     rng,
    #     args.outdir,
    #     max_combos=args.max_combos,
    # )

    # for justice in ["egalitarian"]:
    #     plot_pf_overlay(args.outdir, justice)
    #     plot_stochastic_gain(args.outdir, justice)

    compare_pf_including_rawls(
        df,
        settings,
        rng,
        args.outdir,
        max_combos=args.max_combos,
    )

    all_train_pf_policies_egal, all_train_pf_policies_rawls = load_train_pf_dfs(args.outdir)
    
    for fairness_type in ["egal","rawls"]:
        if fairness_type == "egal":
            all_train_pf_policies = all_train_pf_policies_egal
        else: 
            all_train_pf_policies = all_train_pf_policies_rawls
        calculate_test_time_metrics(test_df,settings,rng,args.outdir,all_train_pf_policies,fairness_type=fairness_type)


if __name__ == "__main__":
    main()
