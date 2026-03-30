import numpy as np
from .evaluation import evaluate_policy

def grid_search(df, settings, rng):
    taus = np.linspace(0.1, 0.99, settings.thresholds)
    sigmas = settings.sigmas if settings.stochastic else [None]  # None => deterministic

    records = []
    for tau in taus:
        for sigma in sigmas:
            owner_u, group_u = evaluate_policy(
                tau, sigma, df,
                settings.owner_costs, settings.customer_costs, rng
            )
            records.append([float(tau), (None if sigma is None else float(sigma)), owner_u, group_u])
    return None, records
