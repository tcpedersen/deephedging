# -*- coding: utf-8 -*-
import tensorflow as tf

from books import simple_put_call_book
import models

# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, time_steps = int(2**20), int(2**20), 30
alpha = 0.95


# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = simple_put_call_book(
    1., 100., 105., 0.05, 0.1, 0.2, 1.)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_train_paths, time_steps, False)


# ==============================================================================
# === setup model
model = models.DeltaHedge(book, time, numeraire)

train = [instruments / numeraire,
         instruments / numeraire,
         book.payoff(instruments, numeraire)]


# ==============================================================================
# ====
# https://www.dropbox.com/s/g1elm90a6i6x79n/hedge_scatter.R?dl=0

value, _ = model(train)

hedge_ratios = model.hedge_ratios(train)
initial_hedge_ratio = hedge_ratios[0, 0, 0]
print(f"initial hedge ratio: {initial_hedge_ratio:.4f}, should be {0.5422283:.4f}.")

price = book.value(time, instruments, numeraire)[0, 0]
print(f"initial investment: {price * numeraire[0]:.4f}, should be {8.0214:.4f}.")

payoff = tf.reduce_mean(book.payoff(instruments, numeraire) * numeraire[0])
print(f"average discounted option payoff: {payoff:.4f}, should be {11.0641:.4f}.")

hedge_wealth = tf.reduce_mean((price + value) * numeraire[0])
print(f"average discounted portfolio value: {hedge_wealth:.4f}, should be {11.0559:.4f} ({11.02643:.4f}, {11.08537:.4f}).")
