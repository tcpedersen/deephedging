# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

from time import perf_counter

import utils
import random_books
import hedge_models

tf.get_logger().setLevel('ERROR')

# ==============================================================================
# === train gradient models
rate = 0.02
drift = 0.05
dimension = int(sys.argv[1])

folder_name = r"results\experiment-12"

number_of_tests = 2**8

test_hedge_drivers = []

for num in range(number_of_tests):
    print(f"dimension {dimension} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    timesteps = 14
    volatility = tf.random.uniform((dimension, ), 0.2, 0.3)
    init_instruments, init_numeraire, book = random_books.random_empty_book(
        timesteps / 250, dimension, rate, drift, volatility, num)
    # random_books.add_butterfly(init_instruments, book, 10)
    random_books.add_calls(init_instruments, book)

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
        risk_neutral=True,
        learning_rate=1e-1
        )
    driver.add_testcase(
        name="continuous-time",
        model=hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")

    driver.train(train_size, 1, int(2**10))
    driver.test(test_size)

    test_hedge_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(folder_name, fr"dimension-{dimension}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

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


