# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import models
import books
import utils

# ==============================================================================
# === hyperparameters
batch_size, timesteps = int(2**16), 7
num_hedges_per_day = 4
alpha = 0.95


# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = books.random_barrier_book(
    timesteps / 250, 1, 1, 1, 72)
time, instruments, numeraire = book.sample_paths(
    init_instruments,
    init_numeraire,
    batch_size,
    timesteps * num_hedges_per_day,
    True)


# ==============================================================================
# === setup model
model = models.DeltaHedge(book, time, numeraire)
model.compile(models.ExpectedShortfall(alpha))

train = utils.benchmark_input(time, instruments, numeraire, book)


# ==============================================================================
# ====

# improves speed
@tf.function
def run(x):
    return model(x)

value, _ = model(train)

price = book.value(time, instruments, numeraire)[0, 0]
print(f"initial investment: {price * numeraire[0]:.4f}.")

payoff = utils.precise_mean(book.payoff(time, instruments, numeraire))
print(f"average discounted option payoff: {payoff:.4f}.")

hedge_wealth = price + utils.precise_mean(value)
print(f"average discounted portfolio value: {hedge_wealth:.4f}.")

# =============================================================================
# === visualize
utils.plot_distributions([model], [train], [price])

utils.plot_barrier_payoff(
        model,
        train,
        price,
        time,
        instruments,
        numeraire,
        book
    )