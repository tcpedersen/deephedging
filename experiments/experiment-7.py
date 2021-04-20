# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

from time import perf_counter

import utils
import books
import gradient_models
import gradient_driver
import hedge_models
import approximators
import preprocessing

tf.get_logger().setLevel('ERROR')

# ==============================================================================
# === train gradient models
warmup_train_size = int(2**13)
layers = 4
units = 20
dimension = int(sys.argv[1])

folder_name = r"figures\markovian-add\experiment-7"

number_of_tests = 2**3

test_warmup_drivers = []
test_hedge_drivers = []

for num in range(number_of_tests):
    print(f"dimension {dimension} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    timesteps = 13
    init_instruments, init_numeraire, book = books.random_put_call_book(
        timesteps / 52, dimension, dimension, dimension, num)

    warmup_driver = gradient_driver.GradientDriver(
        timesteps=timesteps,
        frequency=0,
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        learning_rate_min=1e-7,
        learning_rate_max=1e-2
        )

    warmup_driver.set_exploration(100., 15.)

    warmup_driver.add_testcase(
        name="twin network",
        model=gradient_models.SequenceTwinNetwork(
            layers=layers,
            units=units,
            activation=tf.keras.activations.softplus
            ),
        train_size=warmup_train_size
        )

    warmup_driver.train(100, 64)
    test_warmup_drivers.append(warmup_driver)

    # ==============================================================================
    # === run hedge experiment
    train_size, test_size = int(2**18), int(2**18)
    alpha = 0.95

    driver = utils.HedgeDriver(
        timesteps=timesteps,
        frequency=0, # no need for frequency for non-path dependent derivatives.
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        cost=1/100,
        risk_neutral=False,
        learning_rate=1e-1
        )
    driver.add_testcase(
        name="continuous-time",
        model=hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")

    driver.add_testcase(
        name="continuous time feature map",
        model=hedge_models.LinearFeatureHedge(
            timesteps,
            book.instrument_dim,
            [approximators.IdentityFeatureMap] \
                * (1 + int(driver.cost is not None))
        ),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")


    driver.add_testcase(
        "deep network",
        hedge_models.NeuralHedge(
            timesteps=timesteps,
            instrument_dim=book.instrument_dim,
            internal_dim=0,
            num_layers=4,
            num_units=10,
            activation=tf.keras.activations.softplus),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

    for case in warmup_driver.testcases:
        driver.add_testcase(
            case["name"] + " feature map",
            model=hedge_models.LinearFeatureHedge(
                timesteps,
                book.instrument_dim,
                [approximators.IdentityFeatureMap] \
                    * (1 + int(driver.cost is not None))
            ),
            risk_measure=hedge_models.ExpectedShortfall(alpha),
            normaliser=None,
            feature_function=warmup_driver.make_feature_function(case),
            price_type="arbitrage")

    driver.train(train_size, 1000, int(2**10))
    driver.test(test_size)

    test_hedge_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(folder_name, fr"dimension-{dimension}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

utils.driver_data_dumb(
    test_warmup_drivers,
    ["train_time"],
    file_name
    )

utils.driver_data_dumb(
    test_hedge_drivers,
    ["test_risk", "test_wealth_with_price_abs_mean",
     "test_wealth_with_price_variance", "price", "train_time"],
    file_name
    )
