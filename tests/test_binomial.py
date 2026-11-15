import pytest

from options_lab import binomial, black_scholes as bs


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_converges_to_black_scholes(kind):
    exact = bs.price(kind, 100, 95, 1.0, 0.05, 0.3, q=0.02)
    assert binomial.price(kind, 100, 95, 1.0, 0.05, 0.3, q=0.02, steps=2000) == pytest.approx(exact, abs=0.01)


def test_error_shrinks_with_more_steps():
    exact = bs.price("put", 100, 100, 1.0, 0.05, 0.2)
    coarse = abs(binomial.price("put", 100, 100, 1.0, 0.05, 0.2, steps=50) - exact)
    fine = abs(binomial.price("put", 100, 100, 1.0, 0.05, 0.2, steps=800) - exact)
    assert fine < coarse / 5


def test_american_call_without_dividends_equals_european():
    eu = binomial.price("call", 100, 100, 1.0, 0.05, 0.2, steps=400)
    am = binomial.price("call", 100, 100, 1.0, 0.05, 0.2, steps=400, american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_put_has_early_exercise_premium():
    eu = binomial.price("put", 100, 100, 1.0, 0.05, 0.2, steps=1000)
    am = binomial.price("put", 100, 100, 1.0, 0.05, 0.2, steps=1000, american=True)
    assert am > eu + 0.3
    # Widely quoted benchmark for these inputs: American put about 6.09
    assert am == pytest.approx(6.09, abs=0.01)


def test_deep_in_the_money_american_put_is_worth_intrinsic():
    am = binomial.price("put", 50, 100, 1.0, 0.10, 0.2, steps=500, american=True)
    assert am == pytest.approx(50.0, abs=1e-9)
