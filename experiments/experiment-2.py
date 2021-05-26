# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys

from functools import partial
from time import perf_counter

import hedge_models
import utils
import approximators
import preprocessing
import random_books

# ==============================================================================
if str(sys.argv[1]) == "cost":
    cost = True
else:
    cost = False

folder_name = r"results\experiment-2\cost" if cost \
    else r"results\experiment-2\no-cost"

# ==============================================================================
# === hyperparameters
rate = 0.02
drift = 0.05
volatility = 0.4

train_size, test_size, timesteps = int(2**18), int(2**18), 12
hedge_multiplier = 1
alpha = 0.95
dimension = int(sys.argv[2])

units = 15
layers = 4

activation = tf.keras.activations.softplus
num_trials = 8
lst_of_drivers =  []

if cost:
    risk_measure = partial(hedge_models.MeanVariance, aversion=1.)
else:
    risk_measure = partial(hedge_models.ExpectedShortfall, alpha=alpha)


for num in range(num_trials):
    print(f"dimension {dimension} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    init_instruments, init_numeraire, book = random_books.random_empty_book(
        timesteps / 12, dimension, rate, drift, volatility, num)
    random_books.add_dga_calls(init_instruments, book)

    driver = utils.HedgeDriver(
        timesteps=timesteps * hedge_multiplier,
        frequency=0, # no need for frequency
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        cost=1/100 if cost else None,
        risk_neutral=not cost,
        learning_rate=1e-1
        )

    driver.add_testcase(
        name="continuous-time",
        model=hedge_models.FeatureHedge(),
        risk_measure=risk_measure(),
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
        risk_measure=risk_measure(),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

    driver.add_testcase(
        name="identity feature map",
        model=hedge_models.LinearFeatureHedge(
            timesteps * hedge_multiplier,
            book.instrument_dim,
            [approximators.IdentityFeatureMap] \
                * (1 + int(driver.cost is not None))
        ),
        risk_measure=risk_measure(),
        normaliser=None,
        feature_function="delta",
        price_type="arbitrage")

    driver.add_testcase(
        "deep memory network",
        hedge_models.NeuralHedge(
            timesteps=timesteps * hedge_multiplier,
            instrument_dim=book.instrument_dim,
            internal_dim=book.instrument_dim,
            num_layers=layers,
            num_units=units,
            activation=activation),
        risk_measure=risk_measure(),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

    driver.add_testcase(
        "lstm",
        hedge_models.RecurrentHedge(
            timesteps=timesteps * hedge_multiplier,
            rnn=tf.keras.layers.LSTM,
            instrument_dim=book.instrument_dim,
            cells=2,
            units=15),
        risk_measure=risk_measure(),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale_with_time",
        price_type="arbitrage"
        )


    if driver.cost is not None or not driver.risk_neutral:
        driver.add_liability_free(
            hedge_models.LinearFeatureHedge(
                timesteps=timesteps * hedge_multiplier,
                instrument_dim=book.instrument_dim,
                mappings=[approximators.IdentityFeatureMap] \
                    * (1 + (driver.cost is not None))),
            risk_measure=risk_measure(),
            normaliser=preprocessing.MeanVarianceNormaliser(),
            feature_function="log_martingale")

    driver.train(train_size, 1000, int(2**10))
    driver.test(test_size)
    lst_of_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")


file_name = os.path.join(folder_name, fr"dimension-{dimension}.txt")
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
