# options-pricing-lab

Option pricing three ways (closed form, lattice, Monte Carlo), cross-checked against
each other. Every Monte Carlo number comes with its standard error.

| Method | What it prices | Module |
|---|---|---|
| Black-Scholes-Merton | European calls and puts, all five Greeks, implied volatility | `black_scholes.py` |
| CRR binomial tree | European and American options | `binomial.py` |
| Monte Carlo | European (plain, antithetic, control variate), arithmetic Asian (geometric control variate), American (Longstaff-Schwartz) | `monte_carlo.py` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

## Usage

```bash
export PYTHONPATH=src
python3 -m options_lab price --kind put --spot 100 --strike 100 --maturity 1 --rate 0.05 --vol 0.2
python3 -m options_lab variance --paths 100000
python3 -m options_lab iv --kind call --price 10.4506 --spot 100 --strike 100 --maturity 1 --rate 0.05
```

## Results

S = K = 100, T = 1, r = 5%, sigma = 20%, 100,000 paths, seed 0.

**The three methods agree.**

| Put | Black-Scholes | Binomial (500 steps) | Monte Carlo |
|---|---|---|---|
| European | 5.5735 | 5.5695 | 5.5713 ± 0.0178 |
| American | n/a | 6.0888 | 6.0739 ± 0.0227 (Longstaff-Schwartz, 50 exercise dates) |

The American put is worth about 0.52 more than the European one: the early exercise premium.

**Variance reduction, at the same number of payoff evaluations.**

| Estimator | Price ± s.e. | Variance vs plain |
|---|---|---|
| European call, plain | 10.4362 ± 0.0465 | 1x |
| European call, antithetic | 10.4721 ± 0.0329 | 2.0x lower |
| European call, control variate (S_T) | 10.4484 ± 0.0178 | 6.9x lower |
| Arithmetic Asian call, plain | 5.8505 ± 0.0256 | 1x |
| Arithmetic Asian call, geometric control variate | 5.8572 ± 0.0007 | about 1,300x lower |

The geometric Asian call has a closed form and is almost perfectly correlated with the
arithmetic one, so using it as a control removes nearly all the sampling noise.

## How correctness is checked

The 32 tests check the pricers against things known independently:

- textbook Black-Scholes values and put-call parity with dividends
- analytic Greeks against central finite differences
- implied volatility round trips from 5% to 200% vol, and rejection of arbitrage prices
- the binomial tree converging to Black-Scholes as steps increase
- American call without dividends equal to the European call; deep in-the-money American put equal to intrinsic value
- every Monte Carlo estimator within 4 standard errors of the closed form
- Longstaff-Schwartz within 0.05 of a 2,000-step binomial tree

## Limitations

- Constant volatility and rates; no smile, no stochastic volatility, no discrete dividends.
- Longstaff-Schwartz fits and applies the exercise rule on the same paths and allows exercise
  only on a discrete grid. Both bias the price slightly; the standard error does not cover that bias.
- The binomial tree oscillates as steps change; no smoothing or Richardson extrapolation is applied.
