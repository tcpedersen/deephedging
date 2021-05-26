# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

import simulators
import derivatives
import utils
import books
import hedge_models
import preprocessing
import approximators

from constants import FLOAT_DTYPE

class BrownianMotion(simulators.GBM):
    def __init__(self, diffusion):
        super().__init__(
            rate=0.,
            drift=tf.constant([0.], FLOAT_DTYPE),
            diffusion=tf.constant([[diffusion]], FLOAT_DTYPE)
            )

    def advance(self, state, rvs, dt, risk_neutral):
        return state + self.diffusion * tf.sqrt(dt) * rvs


class BachelierBinary(derivatives.BinaryCall):
    def __init__(self, maturity, strike, volatility):
        super().__init__(maturity, strike, volatility)

    def adjoint(self, time, instrument, numeraire):
        raise NotImplementedError

    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time
        vol_time = self.volatility * tf.sqrt(ttm)
        d = (instrument - self.strike) / vol_time

        return utils.norm_cdf(d)

    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time
        vol_time = self.volatility * tf.sqrt(ttm)
        d = (instrument - self.strike) / vol_time

        return utils.norm_pdf(d) / vol_time

    def gamma(self, time, instrument, numeraire):
        ttm = self.maturity - time
        vol_time = self.volatility * tf.sqrt(ttm)
        d = (instrument - self.strike) / vol_time

        return -d * self.delta(time, instrument, numeraire)

cost = False

spot = 1
strike = 1
timesteps = 14
sigma = 0.2
maturity = timesteps / 250

if cost:
    instrument_simulator = simulators.GBM(0.0, 0.0, [[sigma]])
    derivative = derivatives.PutCall(maturity, strike, 0.0, sigma, 1)
else:
    instrument_simulator = BrownianMotion(sigma)
    derivative = BachelierBinary(maturity, strike, sigma)
numeraire_simulator = simulators.ConstantBankAccount(0.0)

book = books.DerivativeBook(
    maturity,
    instrument_simulator,
    numeraire_simulator)

book.add_derivative(derivative, 0, 1.0)

init_instruments = tf.constant([spot], FLOAT_DTYPE)
init_numeraire = tf.constant([1.0], FLOAT_DTYPE)

driver = utils.HedgeDriver(
    timesteps=timesteps,
    frequency=0, # no need for frequency for non-path dependent derivatives.
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1/100 if cost else None,
    risk_neutral=True,
    learning_rate=1e-1
    )
driver.verbose = 2

risklevels = [0.05, 0.5, 0.95] if not cost else [0.95]
for alpha in risklevels:
    driver.add_testcase(
        f"deep network {alpha}",
        hedge_models.NeuralHedge(
            timesteps=timesteps,
            instrument_dim=book.instrument_dim,
            internal_dim=0,
            num_layers=4,
            num_units=5,
            activation=tf.keras.activations.softplus),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.LinearFeatureHedge(
            timesteps=timesteps,
            instrument_dim=book.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] \
                * (1 + (driver.cost is not None))),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale")

train_size, test_size = int(2**18), int(2**18)
driver.train(train_size, epochs=1000, batch_size=64)
driver.test(test_size)

# ==============================================================================
# === visualise
raw_data = driver.sample(int(2**18))
idx = 8

if cost:
    case = driver.testcases[1]
    input_data = driver.get_input(case, raw_data)
    ratios = case["model"].strategy(input_data[0], training=False)

    x1 = raw_data["instruments"][:, 0, idx]
    x2 = ratios[:, 0, idx - 1]
    y = ratios[:, 0, idx] - raw_data["delta"][:, 0, idx]

    plt.figure()
    plt.xlabel("value of underlying instrument")
    plt.ylabel("holdings from previous period")

    plt.scatter(x1.numpy(), x2.numpy(), c=y.numpy(), s=0.5)
    plt.colorbar()
    plt.ioff()
    plt.savefig(fr"figures\riskaverseplot-cost-{case['name']}.png", dpi=500)
else:
    colours = ["#E32D91", "#C830CC", "#4EA6DC", "#4775E7", "#8971E1"]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colours)
    time, instruments, numeraire = raw_data["time"], raw_data["instruments"], \
        raw_data["numeraire"]

    plt.figure()
    x = raw_data["instruments"][:, 0, idx]
    key = tf.argsort(x)
    delta = derivative.delta(time, instruments, numeraire)[:, 0, idx]
    gamma = derivative.gamma(time, instruments, numeraire)[:, 0, idx]

    # plt.plot(tf.gather(x, key).numpy(), tf.gather(delta, key).numpy(), "--", color="black")
    plt.plot(tf.gather(x, key).numpy(), tf.gather(gamma, key).numpy(), "-.", color="black")

    for case in driver.testcases:
        input_data = driver.get_input(case, raw_data)
        strategy = case["model"].strategy(input_data[0], training=False)
        y = strategy[:, 0, idx] - delta # remove
        plt.plot(tf.gather(x, key).numpy(), tf.gather(y, key).numpy())
    # plt.xlim(0.85, 1.15)
    plt.xlabel("value of underlying instrument")
    plt.ylabel("exposure to underlying instrument")
    plt.legend(["\u0394", "\u0393"] + [f"\u03B1={alpha:.0%}" for alpha in risklevels])
    plt.savefig(r"figures\riskaverseplot-nocost.eps")
