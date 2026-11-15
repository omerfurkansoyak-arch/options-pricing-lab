"""Black-Scholes-Merton prices, Greeks and implied volatility.

Notation: spot S, strike K, maturity T (years), rate r, dividend yield q,
volatility sigma. All rates are continuously compounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm

CALL, PUT = "call", "put"


def _check(kind: str) -> None:
    if kind not in (CALL, PUT):
        raise ValueError("kind must be 'call' or 'put'")


def _d1_d2(s, k, t, r, q, sigma):
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(kind: str, s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """European option price."""
    _check(kind)
    if s <= 0 or k <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= 0 or sigma <= 0:
        # no time value left: discounted intrinsic value of the forward
        forward_intrinsic = s * math.exp(-q * max(t, 0)) - k * math.exp(-r * max(t, 0))
        return max(forward_intrinsic, 0.0) if kind == CALL else max(-forward_intrinsic, 0.0)
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    if kind == CALL:
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float      # per 1.00 change in volatility (divide by 100 for per vol point)
    theta: float     # per year (divide by 365 for per calendar day)
    rho: float       # per 1.00 change in the rate


def greeks(kind: str, s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> Greeks:
    """Analytic Greeks of a European option."""
    _check(kind)
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        raise ValueError("greeks need positive spot, strike, maturity and volatility")
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    disc_q, disc_r = math.exp(-q * t), math.exp(-r * t)
    pdf = norm.pdf(d1)
    gamma = disc_q * pdf / (s * sigma * math.sqrt(t))
    vega = s * disc_q * pdf * math.sqrt(t)
    decay = -s * disc_q * pdf * sigma / (2 * math.sqrt(t))
    if kind == CALL:
        delta = disc_q * norm.cdf(d1)
        theta = decay - r * k * disc_r * norm.cdf(d2) + q * s * disc_q * norm.cdf(d1)
        rho = k * t * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = decay + r * k * disc_r * norm.cdf(-d2) - q * s * disc_q * norm.cdf(-d1)
        rho = -k * t * disc_r * norm.cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


def implied_vol(kind: str, market_price: float, s: float, k: float, t: float, r: float,
                q: float = 0.0, lo: float = 1e-6, hi: float = 5.0) -> float:
    """Volatility that reproduces `market_price`, found with Brent's method.

    Raises ValueError if the price violates no-arbitrage bounds, in which case
    no volatility in [lo, hi] can match it.
    """
    _check(kind)
    low_price = price(kind, s, k, t, r, lo, q)
    high_price = price(kind, s, k, t, r, hi, q)
    if not low_price <= market_price <= high_price:
        raise ValueError(
            f"price {market_price:.6g} is outside the attainable range "
            f"[{low_price:.6g}, {high_price:.6g}]"
        )
    return brentq(lambda v: price(kind, s, k, t, r, v, q) - market_price, lo, hi, xtol=1e-12)
