"""Cox-Ross-Rubinstein binomial tree for European and American options."""

from __future__ import annotations

import math

import numpy as np

from .black_scholes import CALL, PUT


def price(kind: str, s: float, k: float, t: float, r: float, sigma: float,
          q: float = 0.0, steps: int = 500, american: bool = False) -> float:
    """Price an option on a CRR tree.

    u = exp(sigma sqrt(dt)), d = 1/u, risk-neutral up probability
    p = (exp((r - q) dt) - d) / (u - d). With `american=True`, each node takes
    the larger of the continuation value and immediate exercise.
    """
    if kind not in (CALL, PUT):
        raise ValueError("kind must be 'call' or 'put'")
    if steps < 1 or t <= 0 or sigma <= 0:
        raise ValueError("need steps >= 1 and positive maturity and volatility")

    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("tree is not arbitrage-free for these inputs; increase steps")
    disc = math.exp(-r * dt)
    sign = 1.0 if kind == CALL else -1.0

    j = np.arange(steps + 1)
    spot = s * u ** (steps - j) * d**j            # terminal prices, highest first
    values = np.maximum(sign * (spot - k), 0.0)

    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            j = np.arange(n + 1)
            spot = s * u ** (n - j) * d**j
            values = np.maximum(values, sign * (spot - k))
    return float(values[0])
