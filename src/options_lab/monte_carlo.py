"""Monte Carlo pricing under risk-neutral GBM, with variance reduction.

Every pricer returns an `Estimate` (price, standard error), so the sampling
error is always visible next to the number.

- European options: plain, antithetic variates, control variate (discounted S_T).
- Arithmetic Asian call: control variate using the geometric Asian call, which
  has a closed form and is highly correlated with the arithmetic payoff.
- American options: Longstaff-Schwartz least-squares Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import CALL, PUT


@dataclass(frozen=True)
class Estimate:
    price: float
    std_error: float

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_error, self.price + z * self.std_error


def _estimate(samples: np.ndarray) -> Estimate:
    return Estimate(float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(samples.size)))


def _payoff(kind: str, spot: np.ndarray, k: float) -> np.ndarray:
    if kind == CALL:
        return np.maximum(spot - k, 0.0)
    if kind == PUT:
        return np.maximum(k - spot, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def _terminal(s, t, r, q, sigma, z):
    return s * np.exp((r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)


def european(kind: str, s: float, k: float, t: float, r: float, sigma: float,
             n_paths: int, rng: np.random.Generator, q: float = 0.0,
             method: str = "plain") -> Estimate:
    """European option by Monte Carlo.

    method:
      "plain"       independent draws
      "antithetic"  n_paths // 2 pairs (Z, -Z), so n_paths payoffs are evaluated,
                    the same work as "plain"; the estimator averages pair means
      "control"     control variate with the discounted terminal price, whose
                    expectation is S * exp(-q T)
    """
    disc = math.exp(-r * t)
    if method == "plain":
        z = rng.standard_normal(n_paths)
        return _estimate(disc * _payoff(kind, _terminal(s, t, r, q, sigma, z), k))
    if method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        up = _payoff(kind, _terminal(s, t, r, q, sigma, z), k)
        down = _payoff(kind, _terminal(s, t, r, q, sigma, -z), k)
        return _estimate(disc * 0.5 * (up + down))
    if method == "control":
        z = rng.standard_normal(n_paths)
        s_t = _terminal(s, t, r, q, sigma, z)
        y = disc * _payoff(kind, s_t, k)
        x = disc * s_t
        return _estimate(_control_variate(y, x, s * math.exp(-q * t)))
    raise ValueError("method must be 'plain', 'antithetic' or 'control'")


def _control_variate(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
    """Return adjusted samples y - beta (x - E[x]) with the variance-minimising beta."""
    cov = np.cov(y, x, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    return y - beta * (x - x_mean)


# -- Asian options --------------------------------------------------------------

def geometric_asian_call(s: float, k: float, t: float, r: float, sigma: float,
                         n_obs: int, q: float = 0.0) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian call.

    Observations at t_i = i T / n. The log of the geometric mean is normal with
      mean m = ln S + (r - q - sigma^2/2) T (n + 1) / (2n)
      var  v = sigma^2 T (n + 1)(2n + 1) / (6 n^2)
    """
    n = n_obs
    m = math.log(s) + (r - q - 0.5 * sigma**2) * t * (n + 1) / (2 * n)
    v = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = math.sqrt(v)
    d1 = (m - math.log(k) + v) / sd
    d2 = d1 - sd
    return math.exp(-r * t) * (math.exp(m + 0.5 * v) * norm.cdf(d1) - k * norm.cdf(d2))


def _paths(s, t, r, q, sigma, n_steps, n_paths, rng) -> np.ndarray:
    """GBM paths at t_1..t_n (shape n_paths x n_steps); the initial price is not included."""
    dt = t / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    return s * np.exp(np.cumsum(log_inc, axis=1))


def asian_call(s: float, k: float, t: float, r: float, sigma: float, n_obs: int,
               n_paths: int, rng: np.random.Generator, q: float = 0.0,
               control_variate: bool = True) -> Estimate:
    """Arithmetic-average Asian call, optionally with the geometric Asian as control variate."""
    paths = _paths(s, t, r, q, sigma, n_obs, n_paths, rng)
    disc = math.exp(-r * t)
    arithmetic = disc * np.maximum(paths.mean(axis=1) - k, 0.0)
    if not control_variate:
        return _estimate(arithmetic)
    geometric = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - k, 0.0)
    exact = geometric_asian_call(s, k, t, r, sigma, n_obs, q)
    return _estimate(_control_variate(arithmetic, geometric, exact))


# -- American options -----------------------------------------------------------

def american_lsm(kind: str, s: float, k: float, t: float, r: float, sigma: float,
                 n_steps: int, n_paths: int, rng: np.random.Generator, q: float = 0.0,
                 degree: int = 3) -> Estimate:
    """American option by Longstaff-Schwartz least-squares Monte Carlo.

    Working backward from maturity, the continuation value at each exercise date
    is estimated by regressing discounted future cash flows on polynomials of the
    spot, using in-the-money paths only. A path exercises when immediate exercise
    beats the estimated continuation value.

    The same paths are used to fit and to apply the exercise rule, which biases
    the estimate slightly; the reported standard error does not include that bias.
    """
    dt = t / n_steps
    disc = math.exp(-r * dt)
    paths = _paths(s, t, r, q, sigma, n_steps, n_paths, rng)

    cash = _payoff(kind, paths[:, -1], k)          # value at the current step, per path
    for step in range(n_steps - 2, -1, -1):
        cash = cash * disc                          # discount one step back
        spot = paths[:, step]
        exercise = _payoff(kind, spot, k)
        itm = exercise > 0
        if itm.sum() > degree + 1:
            x = spot[itm] / k                       # scale for a well-conditioned regression
            coeffs = np.polynomial.polynomial.polyfit(x, cash[itm], degree)
            continuation = np.polynomial.polynomial.polyval(x, coeffs)
            do_exercise = exercise[itm] > continuation
            idx = np.flatnonzero(itm)[do_exercise]
            cash[idx] = exercise[idx]
    cash = cash * disc                              # from the first exercise date to time 0
    estimate = _estimate(cash)
    # exercising immediately is always allowed
    intrinsic = float(_payoff(kind, np.array([s]), k)[0])
    return Estimate(max(estimate.price, intrinsic), estimate.std_error)
