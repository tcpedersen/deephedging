# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
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
cost = False

rate = 0.02
drift = 0.05
volatility = 0.2

train_size, test_size, timesteps = int(2**18), int(2**18), 14
alpha = 0.95
dimension = 1
frequency = 4

units = 15
layers = 4

activation = tf.keras.activations.softplus
num_trials = 8
lst_of_drivers =  []
risk_measure = partial(hedge_models.ExpectedShortfall, alpha=alpha)

init_instruments, init_numeraire, book = random_books.random_empty_book(
    timesteps / 250, dimension, rate, drift, volatility, 69)
random_books.add_rko(init_instruments, book, 5)

driver = utils.HedgeDriver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1/100 if cost else None,
    risk_neutral=not cost,
    learning_rate=1e-1
    )


driver.add_testcase(
    "deep network",
    hedge_models.NeuralHedge(
        timesteps=timesteps,
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
    "lstm",
    hedge_models.RecurrentHedge(
        timesteps=timesteps,
        rnn=tf.keras.layers.LSTM,
        instrument_dim=book.instrument_dim,
        cells=2,
        units=15),
    risk_measure=risk_measure(),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale_with_time",
    price_type="arbitrage"
    )

driver.verbose = 2
driver.train(train_size, epochs=1000, batch_size=int(2**10))
driver.test(test_size)


def metric(time, instruments, numeraire):
    derivative = driver.book.derivatives[0]["derivative"]

    return derivative.crossed(instruments[:, 0, :])

raw_data = driver.sample(test_size, metrics=[metric])
crossed = raw_data["metrics"][0]

k = 3
mask = tf.math.logical_and(crossed[..., ::2**4][..., k - 1] == False,
                           crossed[..., ::2**4][..., k] == True)

case = driver.testcases[0]
assert case["name"] == "deep network"
input_data = driver.get_input(case, raw_data)
nonmemorystrategy = case["model"].strategy(input_data[0], training=False)

case = driver.testcases[1]
assert case["name"] == "lstm"
input_data = driver.get_input(case, raw_data)
memorystrategy = case["model"].strategy(input_data[0], training=False)

plt.figure()
colours = ["#E32D91", "#4775E7"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colours)
x = raw_data["time"][:-1]

names = ["deep network", "lstm"]
for name, strategy in zip(names, [nonmemorystrategy, memorystrategy]):
    full = tf.boolean_mask(strategy, mask, axis=0)[:, 0, :]
    mean = tf.reduce_mean(full, axis=0)
    std = tf.math.reduce_std(full, axis=0)
    cl = mean - std
    cu = mean + std

    plt.fill_between(x.numpy(), cl.numpy(), cu.numpy(), alpha=0.5)
    plt.plot(x.numpy(), mean.numpy(), label=name)
plt.vlines(x[k], *plt.ylim(), colors="red", linestyle="dashed",
           label="crossing time")

plt.xlabel("time")
plt.ylabel("exposure to underlying instrument")
plt.legend()
plt.savefig(r"figures\memory.png", dpi=500)

