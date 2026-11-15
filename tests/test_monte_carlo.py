import numpy as np
import pytest

from options_lab import binomial, black_scholes as bs, monte_carlo as mc

S, K, T, R, VOL = 100.0, 100.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_estimators_are_unbiased(method, kind):
    exact = bs.price(kind, S, K, T, R, VOL, q=0.01)
    est = mc.european(kind, S, K, T, R, VOL, 200_000, np.random.default_rng(1), q=0.01, method=method)
    assert abs(est.price - exact) < 4 * est.std_error


def test_variance_reduction_lowers_standard_error():
    def se(method):
        return mc.european("call", S, K, T, R, VOL, 100_000, np.random.default_rng(2), method=method).std_error
    assert se("antithetic") < se("plain")
    assert se("control") < 0.6 * se("plain")


def test_geometric_asian_closed_form_matches_simulation():
    rng = np.random.default_rng(3)
    paths = mc._paths(S, T, R, 0.0, VOL, 12, 400_000, rng)
    payoff = np.exp(-R * T) * np.maximum(np.exp(np.log(paths).mean(axis=1)) - K, 0.0)
    se = payoff.std(ddof=1) / np.sqrt(payoff.size)
    assert abs(payoff.mean() - mc.geometric_asian_call(S, K, T, R, VOL, 12)) < 4 * se


def test_geometric_asian_with_one_observation_is_european():
    assert mc.geometric_asian_call(S, K, T, R, VOL, 1) == pytest.approx(bs.price("call", S, K, T, R, VOL))


def test_asian_control_variate_agrees_and_is_much_tighter():
    plain = mc.asian_call(S, K, T, R, VOL, 50, 100_000, np.random.default_rng(4), control_variate=False)
    cv = mc.asian_call(S, K, T, R, VOL, 50, 100_000, np.random.default_rng(4))
    assert abs(plain.price - cv.price) < 4 * plain.std_error
    assert cv.std_error < plain.std_error / 10
    # the arithmetic mean is never below the geometric mean
    assert cv.price > mc.geometric_asian_call(S, K, T, R, VOL, 50)


def test_longstaff_schwartz_matches_binomial_american_put():
    tree = binomial.price("put", S, K, T, R, VOL, steps=2000, american=True)
    lsm = mc.american_lsm("put", S, K, T, R, VOL, 50, 200_000, np.random.default_rng(5))
    assert lsm.price == pytest.approx(tree, abs=0.05)
    assert lsm.price > bs.price("put", S, K, T, R, VOL)


def test_longstaff_schwartz_never_below_intrinsic():
    lsm = mc.american_lsm("put", 60.0, 100.0, 1.0, 0.10, 0.2, 20, 5_000, np.random.default_rng(6))
    assert lsm.price >= 40.0
