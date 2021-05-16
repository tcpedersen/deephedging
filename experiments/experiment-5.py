# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

from time import perf_counter

import utils
import derivatives
import random_books
import gradient_models
import gradient_driver
import hedge_models

tf.get_logger().setLevel('ERROR')

timesteps = 13
init_instruments, init_numeraire, book = random_books.random_empty_book(
    timesteps / 52, 1, 0.02, 0.05, 0.2)
init_numeraire = tf.ones_like(init_numeraire)

spread = 10
itm = derivatives.PutCall(
    book.maturity,
    init_instruments - spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
atm = derivatives.PutCall(
    book.maturity,
    init_instruments,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
otm = derivatives.PutCall(
    book.maturity,
    init_instruments + spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)

book.add_derivative(itm, 0, 1)
book.add_derivative(atm, 0, -2)
book.add_derivative(otm, 0, 1)

# ==============================================================================
# === train gradient models
warmup_train_size = int(sys.argv[1])

folder_name = r"results\experiment-5"

number_of_tests = 8 # too slow to run multiple times?

test_warmup_drivers = []
test_hedge_drivers = []

for num in range(number_of_tests):
    print(f"size {warmup_train_size} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    warmup_driver = gradient_driver.GradientDriver(
        timesteps=timesteps,
        frequency=0,
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        learning_rate_min=1e-7,
        learning_rate_max=1e-2
        )

    warmup_driver.set_exploration(100.0, 15.0)

    for layers in [2, 4]:
        for units in [20]:
            warmup_driver.add_testcase(
                name=f"value network layers {layers}",
                model=gradient_models.SequenceValueNetwork(
                    layers=layers,
                    units=units,
                    activation=tf.keras.activations.softplus
                    ),
                train_size=warmup_train_size
                )

            warmup_driver.add_testcase(
                name=f"twin network layers {layers}",
                model=gradient_models.SequenceTwinNetwork(
                    layers=layers,
                    units=units,
                    activation=tf.keras.activations.softplus
                    ),
                train_size=warmup_train_size
                )

            warmup_driver.add_testcase(
                name=f"delta network layers {layers}",
                model=gradient_models.SequenceDeltaNetwork(
                    layers=layers,
                    units=units,
                    activation=tf.keras.activations.sigmoid
                    ),
                train_size=warmup_train_size
                )

    warmup_driver.train(100, 64)
    test_warmup_drivers.append(warmup_driver)

    # ==============================================================================
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

    # no need to train as test computes analytical ES
    driver.train(train_size, 1, int(2**10))
    driver.test(test_size)

    test_hedge_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(folder_name, fr"size-{warmup_train_size}.txt")
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
