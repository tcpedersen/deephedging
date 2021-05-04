# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

from time import perf_counter

import utils
import simulators
import derivatives
import books
import gradient_models
import gradient_driver
import hedge_models

from constants import FLOAT_DTYPE

tf.get_logger().setLevel('ERROR') # HACK otherwise function retracing message

class BrownianMotion(simulators.GBM):
    def __init__(self, diffusion):
        super().__init__(
            rate=0.,
            drift=tf.constant([0.], FLOAT_DTYPE),
            diffusion=tf.constant([[diffusion]], FLOAT_DTYPE)
            )

    def advance(self, state, rvs, dt, risk_neutral):
        return state + self.diffusion * tf.sqrt(dt) * rvs


class BachelierPutCall(derivatives.PutCall):
    def __init__(self, maturity, strike, volatility):
        super().__init__(maturity, strike, 0.0, volatility, 1.0)


    def adjoint(self, time, instrument, numeraire):
        diff = self.theta * (instrument[..., -1, tf.newaxis] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)

        return self.theta * itm * tf.ones_like(instrument) / numeraire[-1]


    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time
        vol_time = self.volatility * tf.sqrt(ttm)
        d = (instrument - self.strike) / vol_time

        return vol_time * (d * utils.norm_cdf(d) + utils.norm_pdf(d))

    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time
        vol_time = self.volatility * tf.sqrt(ttm)
        d = (instrument - self.strike) / vol_time

        return utils.norm_cdf(d)


class BachelierBook(books.DerivativeBook):
    def exploring_start(self, state, batch_size, loc, scale):
        rvs = tf.random.normal(
            shape=(batch_size, tf.shape(state)[-1]),
            mean=loc,
            stddev=scale,
            dtype=FLOAT_DTYPE
            )

        return rvs

spot = 1
strike = 1
maturity = 1
timesteps = 52
sigma = 0.2

start_time = 5 * maturity

instrument_simulator = BrownianMotion(sigma)
numeraire_simulator = simulators.ConstantBankAccount(0.)

book = BachelierBook(
    maturity,
    instrument_simulator,
    numeraire_simulator)

derivative = BachelierPutCall(maturity, strike, sigma)
book.add_derivative(derivative, 0, 1.)

init_instruments = tf.constant([spot], FLOAT_DTYPE)
init_numeraire = tf.constant([1.], FLOAT_DTYPE)

# ==============================================================================
# === train gradient models
warmup_train_size = int(sys.argv[1])
activation = tf.keras.activations.softplus

folder_name = r"results\polyparrot"
number_of_tests = 2**3

test_warmup_drivers = []
test_hedge_drivers = []

for num in range(number_of_tests):
    print(f"train size {warmup_train_size} at test {num + 1} ".ljust(80, "="),
          end="")
    start = perf_counter()

    warmup_driver = gradient_driver.GradientDriver(
        timesteps=timesteps,
        frequency=0,
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        learning_rate_min=1e-5,
        learning_rate_max=1e-2
        )

    warmup_driver.set_exploration(spot, tf.sqrt(start_time * sigma**2))

    for layers in [2, 3, 4, 5]:
        for units in [5]:
            warmup_driver.add_testcase(
                name=f"payoff layers {layers} units {units}",
                model=gradient_models.SequenceValueNetwork(
                    layers=layers,
                    units=units,
                    activation=activation
                    ),
                train_size=warmup_train_size
                )

            warmup_driver.add_testcase(
                name=f"twin layers {layers} units {units}",
                model=gradient_models.SequenceTwinNetwork(
                    layers=layers,
                    units=units,
                    activation=activation
                    ),
                train_size=warmup_train_size
                )

    warmup_driver.train(100, 32) # NOTE maybe too low batch_size
    test_warmup_drivers.append(warmup_driver)

    # ==========================================================================
    # === run hedge experiment
    train_size, test_size = int(2**10), int(2**18)
    alpha = 0.95

    driver = utils.HedgeDriver(
        timesteps=timesteps,
        frequency=0, # no need for frequency for non-path dependent derivatives.
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        cost=None,
        risk_neutral=False,
        learning_rate=1e-1
        )

    driver.add_testcase(
        "continuous-time",
        hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")

    for case in warmup_driver.testcases:
        driver.add_testcase(
            case["name"],
            hedge_models.FeatureHedge(),
            risk_measure=hedge_models.ExpectedShortfall(alpha),
            normaliser=None,
            feature_function=warmup_driver.make_feature_function(case),
            price_type="arbitrage")

    driver.train(train_size, 1, int(2**10))
    driver.test(test_size)

    test_hedge_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(folder_name, fr"trainsize-{warmup_train_size}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

utils.driver_data_dumb(
    test_warmup_drivers,
    ["train_time"],
    file_name
    )

utils.driver_data_dumb(
    test_hedge_drivers,
    ["train_risk", "test_risk",
     "test_mean_value", "test_mean_abs_value", "test_variance_value",
     "test_mean_costs", "test_mean_abs_costs", "test_variance_costs",
     "test_mean_wealth", "test_mean_abs_wealth", "test_variance_wealth",
     "test_wealth_with_price_abs_mean", "test_wealth_with_price_variance",
     "price", "train_time"],
    file_name
    )
