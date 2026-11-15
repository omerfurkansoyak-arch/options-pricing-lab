import math

import pytest

from options_lab import black_scholes as bs

ARGS = dict(s=100.0, k=105.0, t=0.75, r=0.04, sigma=0.25, q=0.02)


def test_textbook_value():
    # Hull: S=42, K=40, T=0.5, r=10%, sigma=20% -> call 4.76, put 0.81
    assert bs.price("call", 42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.005)
    assert bs.price("put", 42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.81, abs=0.005)


def test_put_call_parity_with_dividends():
    c = bs.price("call", **ARGS)
    p = bs.price("put", **ARGS)
    forward = ARGS["s"] * math.exp(-ARGS["q"] * ARGS["t"]) - ARGS["k"] * math.exp(-ARGS["r"] * ARGS["t"])
    assert c - p == pytest.approx(forward, abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = bs.greeks(kind, **ARGS)

    def bumped(**change):
        return bs.price(kind, **{**ARGS, **change})

    h = 1e-4
    s, sig, r, t = ARGS["s"], ARGS["sigma"], ARGS["r"], ARGS["t"]
    delta = (bumped(s=s + h) - bumped(s=s - h)) / (2 * h)
    gamma = (bumped(s=s + 1e-2) - 2 * bumped() + bumped(s=s - 1e-2)) / 1e-4
    vega = (bumped(sigma=sig + h) - bumped(sigma=sig - h)) / (2 * h)
    rho = (bumped(r=r + h) - bumped(r=r - h)) / (2 * h)
    theta = -(bumped(t=t + h) - bumped(t=t - h)) / (2 * h)   # theta is d/d(calendar time)
    assert g.delta == pytest.approx(delta, abs=1e-6)
    assert g.gamma == pytest.approx(gamma, abs=1e-5)
    assert g.vega == pytest.approx(vega, abs=1e-5)
    assert g.rho == pytest.approx(rho, abs=1e-5)
    assert g.theta == pytest.approx(theta, abs=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8, 2.0])
def test_implied_vol_round_trip(kind, vol):
    px = bs.price(kind, 100, 110, 0.5, 0.03, vol, q=0.01)
    assert bs.implied_vol(kind, px, 100, 110, 0.5, 0.03, q=0.01) == pytest.approx(vol, abs=1e-7)


def test_implied_vol_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        bs.implied_vol("call", 150.0, 100, 100, 1.0, 0.05)   # call worth more than the stock


def test_expired_option_is_intrinsic():
    assert bs.price("call", 110, 100, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert bs.price("put", 110, 100, 0.0, 0.05, 0.2) == 0.0
