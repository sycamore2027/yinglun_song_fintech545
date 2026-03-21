import csv
from math import exp, sqrt
from pathlib import Path


MAX_TREE_STEPS = 250


def bt_american(call, underlying, strike, ttm, rf, b, ivol, steps):
    dt = ttm / steps
    u = exp(ivol * sqrt(dt))
    d = 1.0 / u
    pu = (exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    discount = exp(-rf * dt)
    sign = 1.0 if call else -1.0

    option_values = [0.0] * (steps + 1)
    for i in range(steps + 1):
        price = underlying * (u**i) * (d ** (steps - i))
        option_values[i] = max(0.0, sign * (price - strike))

    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            price = underlying * (u**i) * (d ** (j - i))
            exercise_value = max(0.0, sign * (price - strike))
            continuation_value = discount * (pu * option_values[i + 1] + pd * option_values[i])
            option_values[i] = max(exercise_value, continuation_value)

    return option_values[0]


def bt_american_discrete_dividends(call, underlying, strike, ttm, rf, dividend_amounts, dividend_times, ivol, steps):
    if not dividend_amounts or not dividend_times or dividend_times[0] > steps:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, steps)

    dt = ttm / steps
    u = exp(ivol * sqrt(dt))
    d = 1.0 / u
    pu = (exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    discount = exp(-rf * dt)
    sign = 1.0 if call else -1.0
    first_dividend_time = dividend_times[0]

    option_values = [0.0] * (first_dividend_time + 1)
    for i in range(first_dividend_time + 1):
        price = underlying * (u**i) * (d ** (first_dividend_time - i))
        no_exercise_value = bt_american_discrete_dividends(
            call=call,
            underlying=price - dividend_amounts[0],
            strike=strike,
            ttm=ttm - first_dividend_time * dt,
            rf=rf,
            dividend_amounts=dividend_amounts[1:],
            dividend_times=[time - first_dividend_time for time in dividend_times[1:]],
            ivol=ivol,
            steps=steps - first_dividend_time,
        )
        exercise_value = max(0.0, sign * (price - strike))
        option_values[i] = max(no_exercise_value, exercise_value)

    for j in range(first_dividend_time - 1, -1, -1):
        for i in range(j + 1):
            price = underlying * (u**i) * (d ** (j - i))
            exercise_value = max(0.0, sign * (price - strike))
            continuation_value = discount * (pu * option_values[i + 1] + pd * option_values[i])
            option_values[i] = max(exercise_value, continuation_value)

    return option_values[0]


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def scale_steps_and_dividends(raw_steps, raw_dividend_times):
    if raw_steps <= MAX_TREE_STEPS:
        return raw_steps, raw_dividend_times

    scale = MAX_TREE_STEPS / raw_steps
    scaled_times = [max(1, int(round(time * scale))) for time in raw_dividend_times]
    return MAX_TREE_STEPS, scaled_times


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "test12_3.csv"
output_path = base_dir / "testout12_3.csv"

rows = []
with input_path.open(newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row["ID"].strip():
            rows.append(row)

results = []
for option in rows:
    raw_steps = int(float(option["DaysToMaturity"])) * 2
    raw_dividend_dates = [2 * value for value in parse_int_list(option["DividendDates"])]
    steps, dividend_dates = scale_steps_and_dividends(raw_steps, raw_dividend_dates)
    dividend_amounts = parse_float_list(option["DividendAmts"])

    value = bt_american_discrete_dividends(
        call=option["Option Type"] == "Call",
        underlying=float(option["Underlying"]),
        strike=float(option["Strike"]),
        ttm=float(option["DaysToMaturity"]) / float(option["DayPerYear"]),
        rf=float(option["RiskFreeRate"]),
        dividend_amounts=dividend_amounts,
        dividend_times=dividend_dates,
        ivol=float(option["ImpliedVol"]),
        steps=steps,
    )

    results.append({"ID": int(float(option["ID"])), "Value": value})

with output_path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["ID", "Value"])
    writer.writeheader()
    writer.writerows(results)
