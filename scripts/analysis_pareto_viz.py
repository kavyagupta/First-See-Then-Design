import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from justice import load_config, load_dataset
from justice.evaluation import evaluate_policy
from justice.post import attach_derived_metrics
from justice.pareto import get_pareto_front, pareto_front_egalitarian, pareto_front_rawlsian


param_range = np.linspace(0.1, 0.99, 100)  # fixed

def run_eval_over_tau(df, settings, rng):
    rows = []
    for tau in param_range:
        owner_u, group_u = evaluate_policy(
            tau=float(tau), sigma=None, df=df,     # deterministic line
            owner_costs=settings.owner_costs,
            customer_costs=settings.customer_costs, rng=rng
        )
        rows.append([float(tau), owner_u, group_u])
    res = pd.DataFrame(rows, columns=["tau", "owner_u", "group_utils"])
    return attach_derived_metrics(res)

def plot_pareto(points_df, xlab, ylab, title, out_path):
    plt.figure(); plt.plot(points_df.iloc[:,0], points_df.iloc[:,1], linewidth=2)
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="viz_out")
    parser.add_argument("--rawls_include_owner", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    settings = load_config(args.config)
    rng = np.random.default_rng(settings.seed)
    df = load_dataset(args.data)

    res = run_eval_over_tau(df, settings, rng)

    X = res["owner_u"].to_numpy(float)
    Y = res["disparity"].to_numpy(float)
    pf = get_pareto_front(np.stack([X, Y], axis=1))
    pf_df = pd.DataFrame(pf, columns=["owner_u", "disparity"])
    pf_df.to_csv(os.path.join(args.outdir, "pf_generic.csv"), index=False)

    egal_pf_pts, _ = pareto_front_egalitarian(res)
    raw_pf_pts, _ = pareto_front_rawlsian(res)
    egal_pf_pts.to_csv(os.path.join(args.outdir, "pf_egalitarian.csv"), index=False)
    raw_pf_pts.to_csv(os.path.join(args.outdir, "pf_rawlsian.csv"), index=False)

    plot_pareto(pf_df, "Owner utility", "Disparity", "Generic PF (owner vs disparity)",
                os.path.join(args.outdir, "pf_generic.png"))
    plot_pareto(egal_pf_pts, "Owner utility", "disparity", "Egalitarian PF",
                os.path.join(args.outdir, "pf_egalitarian.png"))
    plot_pareto(raw_pf_pts, "Owner utility", "regulator_utility", "Rawlsian PF",
                os.path.join(args.outdir, "pf_rawlsian.png"))

if __name__ == "__main__":
    main()
