# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import models
from books import random_black_scholes_put_call_book
from constants import FLOAT_DTYPE
from utils import PeakSchedule, MeanVarianceNormaliser

def prepare_sample(instruments, numeraire):
    information = tf.math.log(instruments / numeraire)
    martingales = instruments / numeraire
    payoff = book.payoff(instruments, numeraire)

    return [information, martingales, payoff]


def train_model(model, inputs, alpha, normalise=True):
    # normalise data
    normaliser = MeanVarianceNormaliser()
    norm_information = normaliser.fit_transform(inputs[0]) if normalise else inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2]]

    # compile model
    risk_measure = models.ExpectedShortfall(alpha)
    optimizer = tf.keras.optimizers.Adam(1e-1)
    model.compile(risk_measure, optimizer=optimizer)

    # define callbacks
    batch_size, epochs = 2**10, 100

    early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor="loss", verbose=1, patience=2)

    # schedule = PeakSchedule(1e-4, 1e-1, epochs)
    # lr_schedule = LearningRateScheduler(schedule, verbose=1)

    callbacks = [early_stopping, reduce_lr]

    # train
    history = model.fit(norm_inputs,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

    return history, norm_inputs, normaliser


def test_model(model, inputs, normaliser=None):
    # normalise data
    if normaliser is not None:
        norm_information =  normaliser.transform(inputs[0])
    else:
        norm_information = inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2]]

    # test model
    value, costs = model(norm_inputs)
    risk = model.risk_measure(value - costs - inputs[2])

    return norm_inputs, risk


def plot_distributions(models, inputs, prices):
    data = []
    for model, input, price in zip(models, inputs, prices):
        value, costs = model(input)
        wealth = price + value - costs - input[2]
        data.append([value, costs, wealth])

    # wealth
    plt.figure()
    for value, costs, wealth in data:
        plot_data = np.random.choice(wealth, 250000, replace=False)
        plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    plt.show()

    # value
    plt.figure()
    for value, costs, wealth in data:
        plot_data = np.random.choice(value, 250000, replace=False)
        plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    plt.show()

    # costs
    plt.figure()
    for value, costs, wealth in data:
        plot_data = np.random.choice(costs, 250000, replace=False)
        plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    plt.show()

    return data



# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, num_steps = int(2**20), int(2**20), 30
alpha = 0.95
cost = 1. / 100
num_layers, num_units = 2, 15


# ==============================================================================
# === sample train data
init_instruments, init_numeraire, book = random_black_scholes_put_call_book(
    num_steps / 250, 25, 10, 10, 69)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_train_paths, num_steps, False)


# ==============================================================================
# === train simple model
train = prepare_sample(instruments, numeraire)

if cost is not None:
    simple_model = models.CostSimpleHedge(
        num_steps, book.instrument_dim, num_layers, num_units, cost)
else:
    simple_model = models.SimpleHedge(
        num_steps, book.instrument_dim, num_layers, num_units)

history, norm_train, normaliser = train_model(simple_model, train, alpha)


# ==============================================================================
# === train benchmark
benchmark = [instruments, train[1], train[2]]

benchmark_model = models.DeltaHedge(num_steps, book, numeraire)
if cost is not None:
    benchmark_model.add_cost_layers(cost)

_, _, _ = train_model(benchmark_model, benchmark, alpha, False)


# ==============================================================================
# === train no liability
no_liability = [train[0], train[1], tf.zeros_like(train[2])]

if cost is not None:
    no_liability_model = models.CostSimpleHedge(
        num_steps, book.instrument_dim, num_layers, num_units, cost)
else:
    no_liability_model = models.SimpleHedge(
        num_steps, book.instrument_dim, num_layers, num_units)

_, _, _ = train_model(no_liability_model, no_liability, alpha)


# ==============================================================================
# === sample test data
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_test_paths, num_steps, False)

test = prepare_sample(instruments, numeraire)


# ==============================================================================
# === calculate risk
norm_test, simple_risk = test_model(simple_model, test, normaliser)

benchmark = [instruments, test[1], test[2]]
norm_benchmark, benchmark_risk = test_model(benchmark_model, test, None)

no_liability = [test[0], test[1], tf.zeros_like(test[2])]
_, no_liability_risk = test_model(no_liability_model, no_liability, normaliser)

print(f"simple model risk: {simple_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")


# ==============================================================================
# === calculate prices
simple_model_price = (simple_risk - no_liability_risk)
benchmark_price = tf.squeeze(book.value(time, instruments[0, tf.newaxis, ...], numeraire))[0]
no_liability_price = 0. # TODO is this true?

print(f"simple_model price: {simple_model_price:5f}")
print(f"benchmark price: {benchmark_price:5f}")


# ==============================================================================
# === calculate total risk
print(f"simple model total risk: {simple_risk - simple_model_price:5f}")
print(f"benchmark total risk: {benchmark_risk - benchmark_price:5f}")
print(f"no liability total risk: {no_liability_risk - no_liability_price:5f}")


# ==============================================================================
# === visualise payoff
# plot_payoff(simple_model, norm_train, simple_model_price, book)
# plot_payoff(benchmark_model, norm_benchmark, benchmark_price, book)
# plot_payoff(no_liability_model, norm_no_liability, no_liability_price, book)


# ==============================================================================
# === visualise distribution
plot_distributions([simple_model, benchmark_model],
                   [norm_test, norm_benchmark],
                   [simple_model_price, benchmark_price])
