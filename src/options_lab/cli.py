"""Command line interface.

    python -m options_lab price --kind put --spot 100 --strike 100 --maturity 1 --rate 0.05 --vol 0.2
    python -m options_lab iv --kind call --price 10.45 --spot 100 --strike 100 --maturity 1 --rate 0.05
    python -m options_lab variance --paths 100000
"""

from __future__ import annotations

import argparse

import numpy as np

from . import binomial, black_scholes as bs, monte_carlo as mc


def cmd_price(a) -> None:
    rng = np.random.default_rng(a.seed)
    args = (a.kind, a.spot, a.strike, a.maturity, a.rate, a.vol)
    g = bs.greeks(*args, q=a.div)
    print(f"European {a.kind}")
    print(f"  Black-Scholes:        {bs.price(*args, q=a.div):.4f}")
    print(f"  Binomial ({a.steps} steps): {binomial.price(*args, q=a.div, steps=a.steps):.4f}")
    est = mc.european(*args, a.paths, rng, q=a.div, method="control")
    print(f"  Monte Carlo (control): {est.price:.4f} ± {est.std_error:.4f}")
    print(f"  delta {g.delta:+.4f}  gamma {g.gamma:.4f}  vega {g.vega:.4f}  "
          f"theta {g.theta:+.4f}  rho {g.rho:+.4f}")
    print(f"American {a.kind}")
    print(f"  Binomial ({a.steps} steps): "
          f"{binomial.price(*args, q=a.div, steps=a.steps, american=True):.4f}")
    lsm = mc.american_lsm(*args, 50, a.paths, rng, q=a.div)
    print(f"  Longstaff-Schwartz:   {lsm.price:.4f} ± {lsm.std_error:.4f}")


def cmd_iv(a) -> None:
    vol = bs.implied_vol(a.kind, a.price, a.spot, a.strike, a.maturity, a.rate, q=a.div)
    print(f"implied volatility: {vol:.4%}")


def cmd_variance(a) -> None:
    """Compare standard errors of the estimators at equal numbers of payoff evaluations."""
    s, k, t, r, vol = a.spot, a.strike, a.maturity, a.rate, a.vol
    exact = bs.price("call", s, k, t, r, vol)
    print(f"European call, Black-Scholes = {exact:.4f}, {a.paths} paths")
    base = None
    for method in ("plain", "antithetic", "control"):
        est = mc.european("call", s, k, t, r, vol, a.paths, np.random.default_rng(a.seed),
                          method=method)
        se_per_payoff = est.std_error
        base = base or se_per_payoff
        print(f"  {method:<10} {est.price:.4f} ± {est.std_error:.4f}   "
              f"variance reduction x{(base / se_per_payoff) ** 2:.1f}")
    print(f"Arithmetic Asian call, 50 observations, {a.paths} paths")
    plain = mc.asian_call(s, k, t, r, vol, 50, a.paths, np.random.default_rng(a.seed),
                          control_variate=False)
    cv = mc.asian_call(s, k, t, r, vol, 50, a.paths, np.random.default_rng(a.seed))
    print(f"  plain      {plain.price:.4f} ± {plain.std_error:.4f}")
    print(f"  control    {cv.price:.4f} ± {cv.std_error:.4f}   "
          f"variance reduction x{(plain.std_error / cv.std_error) ** 2:.0f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="options_lab")
    sub = p.add_subparsers(dest="command", required=True)

    def market(sp):
        sp.add_argument("--spot", type=float, default=100.0)
        sp.add_argument("--strike", type=float, default=100.0)
        sp.add_argument("--maturity", type=float, default=1.0)
        sp.add_argument("--rate", type=float, default=0.05)
        sp.add_argument("--div", type=float, default=0.0)
        sp.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("price", help="price a European and an American option three ways")
    market(pr)
    pr.add_argument("--kind", choices=["call", "put"], default="put")
    pr.add_argument("--vol", type=float, default=0.2)
    pr.add_argument("--steps", type=int, default=500)
    pr.add_argument("--paths", type=int, default=100_000)
    pr.set_defaults(func=cmd_price)

    iv = sub.add_parser("iv", help="implied volatility from a market price")
    market(iv)
    iv.add_argument("--kind", choices=["call", "put"], default="call")
    iv.add_argument("--price", type=float, required=True)
    iv.set_defaults(func=cmd_iv)

    va = sub.add_parser("variance", help="compare variance reduction techniques")
    market(va)
    va.add_argument("--vol", type=float, default=0.2)
    va.add_argument("--paths", type=int, default=100_000)
    va.set_defaults(func=cmd_variance)
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
