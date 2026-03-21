import csv
import sys
from math import exp, sqrt
from pathlib import Path


def bt_american(call, underlying, strike, ttm, rf, b, ivol, steps):
    dt = ttm / steps
    u = exp(ivol * sqrt(dt))
    d = 1.0 / u
    pu = (exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    discount = exp(-rf * dt)
    sign = 1.0 if call else -1.0

    def node_count(level):
        return (level + 1) * (level + 2) // 2

    def index(i, j):
        return node_count(j - 1) + i

    option_values = [0.0] * node_count(steps)

    for j in range(steps, -1, -1):
        for i in range(j, -1, -1):
            idx = index(i, j)
            price = underlying * (u**i) * (d ** (j - i))
            exercise_value = max(0.0, sign * (price - strike))
            option_values[idx] = exercise_value

            if j < steps:
                continuation_value = discount * (
                    pu * option_values[index(i + 1, j + 1)] + pd * option_values[index(i, j + 1)]
                )
                option_values[idx] = max(exercise_value, continuation_value)

    return option_values[0]


def finite_difference_gradient(func, params):
    gradient = []
    rel_step = sys.float_info.epsilon ** (1.0 / 3.0)

    for idx, value in enumerate(params):
        step = rel_step * max(1.0, abs(value))
        up = list(params)
        down = list(params)
        up[idx] += step
        down[idx] -= step
        derivative = (func(up) - func(down)) / (2.0 * step)
        gradient.append(derivative)

    return gradient


def make_pricer(call):
    def price(params):
        return bt_american(
            call=call,
            underlying=params[0],
            strike=params[1],
            ttm=params[2],
            rf=params[3],
            b=params[4],
            ivol=params[5],
            steps=500,
        )

    return price


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "test12_1.csv"
output_path = base_dir / "testout12_2.csv"

rows = []
with input_path.open(newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row["ID"].strip():
            rows.append(row)

call_pricer = make_pricer(True)
put_pricer = make_pricer(False)

results = []
for option in rows:
    params = [
        float(option["Underlying"]),
        float(option["Strike"]),
        float(option["DaysToMaturity"]) / float(option["DayPerYear"]),
        float(option["RiskFreeRate"]),
        float(option["RiskFreeRate"]) - float(option["DividendRate"]),
        float(option["ImpliedVol"]),
    ]

    pricer = call_pricer if option["Option Type"] == "Call" else put_pricer
    value = pricer(params)
    gradient = finite_difference_gradient(pricer, params)

    gamma_shift = 1.5
    gamma_up = list(params)
    gamma_down = list(params)
    gamma_up[0] += gamma_shift
    gamma_down[0] -= gamma_shift
    gamma = (pricer(gamma_up) + pricer(gamma_down) - 2.0 * value) / (gamma_shift**2)

    results.append(
        {
            "ID": int(float(option["ID"])),
            "Value": value,
            "Delta": gradient[0],
            "Gamma": gamma,
            "Vega": gradient[5],
            "Rho": gradient[3],
            "Theta": gradient[2],
        }
    )

with output_path.open("w", newline="") as csv_file:
    fieldnames = ["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
