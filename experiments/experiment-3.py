# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from models import SimpleHedge, DeltaHedge, EntropicRisk, ExpectedShortfall, CostSimpleHedge, CostDeltaHedge
from books import random_simple_put_call_book, random_black_scholes_put_call_book
from constants import FLOAT_DTYPE
from utils import PeakSchedule, MeanVarianceNormaliser

def split_sample(sample, grads):
    numeraire = sample[:, -1, tf.newaxis, :]
    information = instruments = sample[:, :-1, :] / numeraire
    payoff = book.payoff(sample) / numeraire[:, 0, -1]

    return [information, instruments, payoff, grads]


def train_model(model, inputs, alpha, normalise=True):
    # normalise data
    normaliser = MeanVarianceNormaliser()
    norm_information = normaliser.fit_transform(inputs[0]) if normalise else inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2], inputs[3]]

    # compile model
    risk_measure = ExpectedShortfall(alpha)
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
    norm_inputs = [norm_information, inputs[1], inputs[2], inputs[3]]

    # test model
    test = model(norm_inputs)
    risk = model.risk_measure(test)

    return norm_inputs, risk


# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, num_steps = int(10**5), int(10**6), 7
alpha = 0.95
cost = 1. / 100


# ==============================================================================
# === sample train data
init_state, book = random_black_scholes_put_call_book(
    num_steps / 250, 10, 10, 10, 69)

time, train_samples = book.sample_paths(
    init_state, num_train_paths, num_steps, False)
train_grads = book.book_delta(train_samples, time)


# ==============================================================================
# === train simple model
train = split_sample(train_samples, train_grads)

simple_model = CostSimpleHedge(num_steps, book.instrument_dim, 2, 15, cost)
simple_model.use_gradients = True

_, _, normaliser = train_model(simple_model, train, alpha)


# ==============================================================================
# === sample test data
time, test_samples = book.sample_paths(
    init_state, num_test_paths, num_steps, False)
test_grads = book.book_delta(test_samples, time)

test = split_sample(test_samples, test_grads)

# ==============================================================================
# === calculate risk
norm_test, simple_risk = test_model(simple_model, test, normaliser)

print(f"simple model risk: {simple_risk:5f}")
