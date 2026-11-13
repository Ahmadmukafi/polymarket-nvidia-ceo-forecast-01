# Polymarket forecaster (final): per-market volatility from the mean absolute
# 1-minute price change, averaged over 4 windows (3h, 6h, 12h, 24h), x0.9.
# Median = last price (martingale); normal quantiles widen like a random walk.
#   q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 decimals.
# Practice pinball loss (local, 6 cases): 0.270661 vs house baseline 0.363589.
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
WINDOWS = (180, 360, 720, 1440)
MULT = 0.9


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def meanabs(prices, window):
    seg = prices[-window:]
    if len(seg) < 2:
        return 0.0
    return sum(abs(b - a) for a, b in zip(seg, seg[1:])) / (len(seg) - 1)


def sigma_of(prices):
    return MULT * sum(meanabs(prices, w) for w in WINDOWS) / len(WINDOWS)


def forecast(market, horizon):
    prices = market["prices"]
    if not prices:
        raise ValueError(f"market {market['id']} has no prices")
    x = prices[-1]
    sig = sigma_of(prices)
    return [[price6(x + z * sig * math.sqrt(h)) for z in Z] for h in range(1, horizon + 1)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for file in inputs:
        cid = file[: -len(".in")]
        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            case = json.load(f)
        if case["quantiles"] != QUANTILES:
            raise ValueError(f"{file} asks for quantiles {case['quantiles']}")
        forecasts = [
            {"id": m["id"], "q": forecast(m, case["horizon_minutes"])}
            for m in case["markets"]
        ]
        with open(os.path.join(OUTPUT_DIR, f"{cid}.out"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
