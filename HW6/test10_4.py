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
    else:
        bounds = [tuple(map(float, b)) for b in bounds]

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
        if lower <= 1.0 <= upper:
            w0 = np.full(n, 0.0)
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
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    weights = np.minimum(np.maximum(chosen.x, lower), upper)
    weights = weights / weights.sum()
    return weights, chosen.status


base_dir = Path(__file__).resolve().parent

covar = pd.read_csv(base_dir / "test5_3.csv").to_numpy()
means = pd.read_csv(base_dir / "test10_3_means.csv")["Mean"].to_numpy()
rf = 0.04
bounds = np.column_stack((np.full(5, 0.1), np.full(5, 0.5)))

msr, status = maxSR(covar, means, rf, bounds)
pd.DataFrame({"W": msr}).to_csv(base_dir / "testout10_4.csv", index=False)

print(f"status={status}")
print(pd.DataFrame({"W": msr}))
