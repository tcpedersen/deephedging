# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import models
import books
import utils

# ==============================================================================
# === hyperparameters
batch_size, time_steps = int(2**16), 30
alpha = 0.95


# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = books.random_geometric_asian_book(
    time_steps / 250, 1, 1, 1, 69)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, batch_size, time_steps * 20, False)


# ==============================================================================
# === setup model
model = models.DeltaHedge(book, time, numeraire)
model.compile(models.ExpectedShortfall(alpha))

train = utils.benchmark_input(time, instruments, numeraire, book)


# ==============================================================================
# ====
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

