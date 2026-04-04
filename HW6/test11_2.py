import numpy as np
import pandas as pd
from pathlib import Path


def load_csv(local_path, fallback_path):
    if local_path.exists():
        return pd.read_csv(local_path)
    return pd.read_csv(fallback_path)


def compounded_return(x):
    return np.expm1(np.log1p(x).sum())


def expost_factor(weights, stock_returns, factor_returns, beta):
    stock_returns = stock_returns.copy()
    factor_returns = factor_returns.copy()

    stock_cols = list(stock_returns.columns)
    factor_cols = list(factor_returns.columns)

    w = np.asarray(weights, dtype=float)
    mat_returns = stock_returns[stock_cols].to_numpy(dtype=float)
    ff_returns = factor_returns[factor_cols].to_numpy(dtype=float)
    beta = np.asarray(beta, dtype=float)

    n = mat_returns.shape[0]

    p_return = np.empty(n, dtype=float)
    resid_return = np.empty(n, dtype=float)
    weight_hist = np.empty((n, len(w)), dtype=float)
    factor_weight_hist = np.empty((n, len(factor_cols)), dtype=float)

    last_w = w.copy()
    for i in range(n):
        weight_hist[i, :] = last_w
        factor_weight_hist[i, :] = np.sum(beta * last_w[:, None], axis=0)

        last_w = last_w * (1.0 + mat_returns[i, :])
        p_r = last_w.sum()
        last_w = last_w / p_r
        p_return[i] = p_r - 1.0
        resid_return[i] = p_return[i] - factor_weight_hist[i, :] @ ff_returns[i, :]

    total_ret = compounded_return(p_return)
    k = np.log1p(total_ret) / total_ret
    carino_k = np.log1p(p_return) / p_return / k

    attrib = ff_returns * factor_weight_hist * carino_k[:, None]
    alpha_attrib = resid_return * carino_k

    factor_with_alpha = factor_cols + ["Alpha"]
    factor_data = factor_returns.copy()
    factor_data["Alpha"] = resid_return
    factor_data["Portfolio"] = p_return

    attribution_rows = []
    total_row = {"Value": "TotalReturn"}
    return_row = {"Value": "Return Attribution"}

    for j, col in enumerate(factor_cols):
        total_row[col] = compounded_return(factor_data[col].to_numpy(dtype=float))
        return_row[col] = attrib[:, j].sum()

    total_row["Alpha"] = compounded_return(factor_data["Alpha"].to_numpy(dtype=float))
    return_row["Alpha"] = alpha_attrib.sum()
    total_row["Portfolio"] = compounded_return(factor_data["Portfolio"].to_numpy(dtype=float))
    return_row["Portfolio"] = total_row["Portfolio"]
    attribution_rows.extend([total_row, return_row])

    y = np.column_stack((ff_returns * factor_weight_hist, resid_return))
    x = np.column_stack((np.ones(n), p_return))
    b = np.linalg.inv(x.T @ x) @ x.T @ y
    csd = b[1, :] * np.std(p_return, ddof=1)

    vol_row = {"Value": "Vol Attribution"}
    for j, col in enumerate(factor_with_alpha):
        vol_row[col] = csd[j]
    vol_row["Portfolio"] = np.std(p_return, ddof=1)
    attribution_rows.append(vol_row)

    attribution = pd.DataFrame(attribution_rows)
    weights_df = pd.DataFrame(weight_hist, columns=stock_cols)
    factor_weights_df = pd.DataFrame(factor_weight_hist, columns=factor_cols)
    return attribution, weights_df, factor_weights_df


base_dir = Path(__file__).resolve().parent
ref_dir = base_dir.parents[1] / "FinTech-545-Fall2025" / "testfiles" / "data"

weights = load_csv(base_dir / "test11_2_weights.csv", ref_dir / "test11_2_weights.csv")
factor_returns = load_csv(base_dir / "test11_2_factor_returns.csv", ref_dir / "test11_2_factor_returns.csv")
stock_returns = load_csv(base_dir / "test11_2_stock_returns.csv", ref_dir / "test11_2_stock_returns.csv")
beta_df = load_csv(base_dir / "test11_2_beta.csv", ref_dir / "test11_2_beta.csv")

weights.to_csv(base_dir / "test11_2_weights.csv", index=False)
factor_returns.to_csv(base_dir / "test11_2_factor_returns.csv", index=False)
stock_returns.to_csv(base_dir / "test11_2_stock_returns.csv", index=False)
beta_df.to_csv(base_dir / "test11_2_beta.csv", index=False)

attribution, updated_weights, factor_weights = expost_factor(
    weights["W"].to_numpy(),
    stock_returns,
    factor_returns,
    beta_df.iloc[:, 1:].to_numpy(),
)

attribution.to_csv(base_dir / "testout11_2.csv", index=False)

print(attribution)
