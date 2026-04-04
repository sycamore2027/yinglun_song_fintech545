import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize


def maxSR(covar, means, rf, bounds=None):
    covar = np.asarray(covar, dtype=float)
    means = np.asarray(means, dtype=float)
    n = len(means)

    if bounds is None:
        bounds = [(0.0, None)] * n

    def neg_sr(w):
        excess_ret = w @ means - rf
        vol = np.sqrt(w @ covar @ w)
        if vol <= 0:
            return np.inf
        return -(excess_ret / vol)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    starts = [np.full(n, 1.0 / n)]
    for i in range(n):
        lower, upper = bounds[i]
        lower = 0.0 if lower is None else lower
        upper = np.inf if upper is None else upper
        if lower <= 1.0 <= upper:
            w0 = np.zeros(n)
            w0[i] = 1.0
            starts.append(w0)

    best_result = None
    best_value = np.inf
    best_successful = None
    best_successful_value = np.inf

    for w0 in starts:
        result = minimize(
            neg_sr,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
        if result.success and result.fun < best_successful_value:
            best_successful_value = result.fun
            best_successful = result

    chosen = best_successful if best_successful is not None else best_result
    weights = np.clip(chosen.x, 0.0, None)
    weights = weights / weights.sum()
    return weights, chosen.status


base_dir = Path(__file__).resolve().parent

means = np.arange(0.09, 0.04, -0.01)
pd.DataFrame({"Mean": means}).to_csv(base_dir / "test10_3_means.csv", index=False)

covar = pd.read_csv(base_dir / "test5_3.csv").to_numpy()
means = pd.read_csv(base_dir / "test10_3_means.csv")["Mean"].to_numpy()
rf = 0.04

msr, status = maxSR(covar, means, rf)
pd.DataFrame({"W": msr}).to_csv(base_dir / "testout10_3.csv", index=False)

print(f"status={status}")
print(pd.DataFrame({"W": msr}))
