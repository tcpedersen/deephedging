# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from models import SimpleHedge, DeltaHedge, EntropicRisk, ExpectedShortfall
from books import random_simple_put_call_book
from constants import FLOAT_DTYPE
from utils import PeakSchedule, MeanVarianceNormaliser

def split_sample(sample):
    numeraire = sample[:, -1, tf.newaxis, :]
    information = instruments = sample[:, :-1, :] / numeraire
    payoff = book.payoff(sample) / numeraire[:, 0, -1]

    return [information, instruments, payoff]


def train_model(model, inputs, alpha, normalise=True):
    # normalise data
    normaliser = MeanVarianceNormaliser()
    norm_information = normaliser.fit_transform(inputs[0]) if normalise else inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2]]

    # compile model
    risk_measure = ExpectedShortfall(alpha)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(risk_measure, optimizer=optimizer)

    # define callbacks
    batch_size, epochs = 2**10, 50

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

    return history, norm_inputs


def plot_payoff(model, inputs, price, book):
    hedge_payoff = price + model(inputs) + inputs[2]

    plt.figure()
    plt.scatter(inputs[1][..., -1], hedge_payoff, s=0.5)

    x = tf.cast(tf.linspace(*plt.xlim(), 1000), FLOAT_DTYPE)
    y = book.payoff(x[..., tf.newaxis, tf.newaxis])
    plt.plot(x, y, color="black")

    plt.show()


def plot_distributions(models, inputs, prices):
    plt.figure()
    for model, input, price in zip(models, inputs, prices):
        data = price + model(input)
        plot_data = np.random.choice(data, 100000)
        plt.hist(plot_data, bins=100, density=True, alpha=0.5)
    plt.show()


# ==============================================================================
# === hyperparameters
num_paths, num_steps = int(10**6), 7
alpha = 0.95


# ==============================================================================
# === sample train data
init_state, book = random_simple_put_call_book(num_steps / 250)
time, train_samples = book.sample_paths(init_state, num_paths, num_steps, False)


# ==============================================================================
# === train simple model
train = split_sample(train_samples)

simple_model = SimpleHedge(num_steps, book.instrument_dim, 2, 15)
_, norm_train = train_model(simple_model, train, alpha)


# ==============================================================================
# === train benchmark
benchmark = [train[1], train[1], train[2]]

benchmark_model = DeltaHedge(num_steps, book)
_, norm_benchmark = train_model(benchmark_model, benchmark, alpha, False)


# ==============================================================================
# === train no liability
no_liability = [train[0], train[1], tf.zeros_like(train[2])]

no_liability_model = SimpleHedge(num_steps, book.instrument_dim, 2, 15)
_, norm_no_liability = train_model(no_liability_model, no_liability, alpha)


# ==============================================================================
# === calculate risk
simple_risk = simple_model.risk_measure(simple_model(norm_train))
benchmark_risk = benchmark_model.risk_measure(benchmark_model(norm_benchmark))
no_liability_risk = no_liability_model.risk_measure(no_liability_model(norm_no_liability))

print(f"simple model risk: {simple_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")

# ==============================================================================
# === calculate prices
simple_model_price = (simple_risk - no_liability_risk)
benchmark_price = (book.book_value(init_state, 0.) / init_state[-1])[0, 0]
no_liability_price = 0.

print(f"simple_model price: {simple_model_price:5f}")
print(f"benchmark price: {benchmark_price:5f}")

# ==============================================================================
# === calculate total risk
print(f"simple model total risk: {simple_risk - simple_model_price:5f}")
print(f"benchmark total risk: {benchmark_risk - benchmark_price:5f}")
print(f"no liability total risk: {no_liability_risk - no_liability_price:5f}")

# ==============================================================================
# === visualise payoff
plot_payoff(simple_model, norm_train, simple_model_price, book)
plot_payoff(benchmark_model, norm_benchmark, benchmark_price, book)
plot_payoff(no_liability_model, norm_no_liability, no_liability_price, book)


# ==============================================================================
# === visualise distribution
plot_distributions([simple_model, benchmark_model],
                   [norm_train, norm_benchmark],
                   [simple_model_price, benchmark_price])
