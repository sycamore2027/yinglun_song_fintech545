#10.1 Risk Parity, normal assumption
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize


def pvol(w, covar):
    return np.sqrt(w @ covar @ w)


def pCSD(w, covar):
    pVol = pvol(w, covar)
    return w * (covar @ w) / pVol


def sseCSD(w, covar):
    n = len(w)
    csd = pCSD(w, covar)
    mCSD = np.sum(csd) / n
    dCsd = csd - mCSD
    return 1.0e5 * np.sum(dCsd ** 2)


def riskParity(covar):
    n = covar.shape[0]
    w0 = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, None)] * n
    result = minimize(sseCSD, w0, args=(covar,), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x, result.status


base_dir = Path(__file__).resolve().parent

cin = pd.read_csv(base_dir / "test5_2.csv").to_numpy()
rpp, status = riskParity(cin)
pd.DataFrame({"W": rpp}).to_csv(base_dir / "testout10_1.csv", index=False)
