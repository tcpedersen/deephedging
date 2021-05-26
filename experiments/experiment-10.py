# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys

from time import perf_counter

import hedge_models
import utils
import preprocessing
import random_books

# ==============================================================================
cost = True
folder_name = r"results\experiment-10"

# ==============================================================================
# === hyperparameters
rate = 0.02
drift = 0.05
volatility = 0.2
spread = 10

train_size, test_size, timesteps = int(2**18), int(2**18), 14
hedge_multiplier = 1
alpha = 0.95
dimension = 8

units = int(sys.argv[1])
layers = int(sys.argv[2])
batch_size = int(sys.argv[3])
activation = tf.keras.activations.get(str(sys.argv[4]))

num_trials = 16
lst_of_drivers =  []

# ==============================================================================
# ===

for num in range(num_trials):
    str_ = f"{units} {layers} {batch_size} with {activation.__name__} at test {num + 1} "
    print(str_.ljust(80, "="), end="")
    start = perf_counter()

    init_instruments, init_numeraire, book = random_books.random_empty_book(
        timesteps / 250, dimension, rate, drift, volatility, num)
    random_books.add_butterfly(init_instruments, book, spread)

    driver = utils.HedgeDriver(
        timesteps=timesteps * hedge_multiplier,
        frequency=0, # no need for frequency for non-path dependent derivatives.
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        cost=1/100 if cost else None,
        risk_neutral=not cost,
        learning_rate=1e-1
        )

    driver.add_testcase(
        "continuous-time",
        hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")

    driver.add_testcase(
        "deep network",
        hedge_models.NeuralHedge(
            timesteps=timesteps * hedge_multiplier,
            instrument_dim=book.instrument_dim,
            internal_dim=0,
            num_layers=layers,
            num_units=units,
            activation=activation),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

    driver.train(train_size, 1000, batch_size)
    driver.test(test_size)
    lst_of_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(
    folder_name, fr"{units}_{layers}_{batch_size}_{activation.__name__}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

utils.driver_data_dumb(
    lst_of_drivers,
    ["train_risk", "test_risk",
     "test_mean_value", "test_mean_abs_value", "test_variance_value",
     "test_mean_costs", "test_mean_abs_costs", "test_variance_costs",
     "test_mean_wealth", "test_mean_abs_wealth", "test_variance_wealth",
     "test_wealth_with_price_abs_mean", "test_wealth_with_price_variance",
     "price", "train_time"],
    file_name
    )
