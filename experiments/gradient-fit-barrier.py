# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

import books
import gradient_models
from constants import FLOAT_DTYPE

instrument_dim = 1
init_instruments, init_numeraire, book = books.random_barrier_book(
    1.25, instrument_dim, instrument_dim, instrument_dim, 72)
batch_size, timesteps = int(2**20), 14
frequency = 0

time, instruments, numeraire, payoff, dpayoff = book.gradient_payoff(
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    batch_size=batch_size,
    timesteps=timesteps,
    frequency=frequency,
    risk_neutral=True,
    exploring_scale=0.2)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=10,
    min_delta=1e-4,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    verbose=1,
    patience=2
)

mlr = gradient_models.MemorySemiRecurrent(
    timesteps=timesteps,
    internal_dim=instrument_dim,
    num_layers=4,
    num_units=15)

w = 0.5
mlr.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
            loss="mean_squared_error",
            loss_weights=[w, 1 - w])

mlr.fit(instruments, [payoff, dpayoff], 2**12, 50,
        callbacks=[early_stopping, reduce_lr])

x = tf.cast(tf.tile(tf.linspace(15, 200, 10000)[:, tf.newaxis],
                    [1, timesteps + 1])[:, tf.newaxis, :], FLOAT_DTYPE)
y = mlr(x)[1][..., :-1] * numeraire[:-1]
z = book.delta(time, x, numeraire) * numeraire
xlim = (50, 150)

i_crossed = book.derivatives[0]["derivative"].crossed(instruments[:, 0, :])
x_crossed = book.derivatives[0]["derivative"].crossed(x[:, 0, :])

for idx in tf.range(timesteps):
    for i_mask, x_mask in [(i_crossed[..., idx], x_crossed[..., idx]),
                           (~i_crossed[..., idx], ~x_crossed[..., idx])]:
        plt.figure()

        plt.scatter(tf.boolean_mask(instruments, i_mask, 0)[:, 0, idx],
                    tf.boolean_mask(dpayoff, i_mask, 0)[:, 0, idx],
                    s=0.5)

        plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
                 tf.boolean_mask(y, x_mask, 0)[:, 0, idx],
                 color="black")
        plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
                 tf.boolean_mask(z, x_mask, 0)[:, 0, idx],
                 "--",
                 color="red")
        # plt.xlim(*xlim)
        # plt.ylim(0, 1)
        plt.show()
