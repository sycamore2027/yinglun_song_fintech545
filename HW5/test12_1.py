import csv
from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from pathlib import Path


@dataclass
class GBSMResult:
    value: float
    delta: float
    gamma: float
    vega: float
    rho: float
    theta: float


def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def gbsm(call, underlying, strike, ttm, rf, b, ivol):
    d1 = (log(underlying / strike) + (b + 0.5 * ivol**2) * ttm) / (ivol * sqrt(ttm))
    d2 = d1 - ivol * sqrt(ttm)
    growth = exp((b - rf) * ttm)
    discount = exp(-rf * ttm)
    pdf_d1 = normal_pdf(d1)

    if call:
        delta = growth * normal_cdf(d1)
        value = underlying * delta - strike * discount * normal_cdf(d2)
        theta = (
            -underlying * growth * pdf_d1 * ivol / (2.0 * sqrt(ttm))
            - (b - rf) * underlying * growth * normal_cdf(d1)
            - rf * strike * discount * normal_cdf(d2)
        )
        rho = ttm * strike * discount * normal_cdf(d2)
    else:
        delta = growth * (normal_cdf(d1) - 1.0)
        value = strike * discount * normal_cdf(-d2) - underlying * growth * normal_cdf(-d1)
        theta = (
            -underlying * growth * pdf_d1 * ivol / (2.0 * sqrt(ttm))
            + (b - rf) * underlying * growth * normal_cdf(-d1)
            + rf * strike * discount * normal_cdf(-d2)
        )
        rho = -ttm * strike * discount * normal_cdf(-d2)

    gamma = growth * pdf_d1 / (underlying * ivol * sqrt(ttm))
    vega = underlying * growth * pdf_d1 * sqrt(ttm)

    return GBSMResult(value=value, delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "test12_1.csv"
output_path = base_dir / "testout12_1.csv"

rows = []
with input_path.open(newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row["ID"].strip():
            rows.append(row)

with output_path.open("w", newline="") as csv_file:
    fieldnames = ["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for option in rows:
        result = gbsm(
            call=option["Option Type"] == "Call",
            underlying=float(option["Underlying"]),
            strike=float(option["Strike"]),
            ttm=float(option["DaysToMaturity"]) / float(option["DayPerYear"]),
            rf=float(option["RiskFreeRate"]),
            b=float(option["RiskFreeRate"]) - float(option["DividendRate"]),
            ivol=float(option["ImpliedVol"]),
        )
        writer.writerow(
            {
                "ID": int(float(option["ID"])),
                "Value": result.value,
                "Delta": result.delta,
                "Gamma": result.gamma,
                "Vega": result.vega,
                "Rho": result.rho,
                "Theta": result.theta,
            }
        )
