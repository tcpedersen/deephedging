# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from constants import FLOAT_DTYPE
from books import random_simple_put_call_book
import models

# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, time_steps = int(2**20), int(2**20), 12
alpha = 0.95

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = random_simple_put_call_book(
    1., 100., 105., 0.05, 0.1, 0.2, 1.)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_train_paths, time_steps, False)

# ==============================================================================
# === setup model
model = models.DeltaHedge(time_steps, book, numeraire)
risk_measure = models.ExpectedShortfall(alpha)
optimizer = tf.keras.optimizers.Adam(1e-1)
model.compile(risk_measure, optimizer=optimizer)

train = [instruments,
         instruments / numeraire,
         book.payoff(instruments, numeraire)]

# ==============================================================================
# === train model
batch_size, epochs = 2**10, 100
early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor="loss", verbose=1, patience=2)
callbacks = [early_stopping, reduce_lr]

history = model.fit(
    train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

# ==============================================================================
# ====
# https://www.dropbox.com/s/g1elm90a6i6x79n/hedge_scatter.R?dl=0

value, costs = model(train)

initial_hedge_ratio = tf.squeeze(model.strategy_layers[0](
    [tf.constant([0.], FLOAT_DTYPE), instruments[..., 0], numeraire[tf.newaxis, 0]]))[0]
print(f"initial hedge ratio: {initial_hedge_ratio:.4f}, should be {0.5422283:.4f}.")

price = tf.squeeze(book.value(time, instruments[0, tf.newaxis, ...], numeraire))[0]
print(f"initial investment: {price:.4f}, should be {8.0214:.4f}.")

payoff = tf.reduce_mean(book.payoff(instruments, numeraire) * numeraire[0])
print(f"average discounted option payoff: {payoff:.4f}, should be {11.0445:.4f}.")

hedge_wealth = tf.reduce_mean((price + value - costs) * numeraire[0])
print(f"average discounted portfolio value: {hedge_wealth:.4f}, should be {11.0242:.4f}.")
