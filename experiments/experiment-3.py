# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import models
from books import random_barrier_book
from utils import precise_mean

# ==============================================================================
# === hyperparameters
batch_size, time_steps = int(2**16), 30
alpha = 0.95


# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = random_barrier_book(
    time_steps / 250, 1, 1, 1, 69)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, batch_size, time_steps * 20, False)


# ==============================================================================
# === setup model
model = models.DeltaHedge(book, time, numeraire)

train = [instruments / numeraire,
         instruments / numeraire,
         book.payoff(instruments, numeraire)]


# ==============================================================================
# ====
value, _ = model(train)

price = book.value(time, instruments, numeraire)[0, 0]
print(f"initial investment: {price * numeraire[0]:.4f}.")

payoff = precise_mean(book.payoff(instruments, numeraire))
print(f"average discounted option payoff: {payoff:.4f}.")

hedge_wealth = price + precise_mean(value)
print(f"average discounted portfolio value: {hedge_wealth:.4f}.")

# =============================================================================
# === visualize
derivative = book.derivatives[0]["derivative"]
crossed = tf.squeeze(tf.reduce_any(derivative.crossed(instruments), 2))
payoff = book.payoff(instruments, numeraire)
xlim = (tf.reduce_min(instruments[:, 0, -1]), tf.reduce_max(instruments[:, 0, -1]))

for indices in [crossed, ~crossed]:
    m = tf.boolean_mask(instruments[..., 0, -1], indices, 0)

    key = tf.argsort(m, 0)
    x = tf.gather(m, key)

    y1 = tf.gather(payoff[indices], key)
    y2 = tf.gather(tf.boolean_mask(price + value, indices), key)

    plt.figure()
    plt.xlim(*xlim)
    plt.scatter(x, y2, s=0.5)
    plt.plot(x, y1, color="black")
    plt.show()
