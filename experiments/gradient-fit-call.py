# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

import books
import gradient_models
from constants import FLOAT_DTYPE

instrument_dim = 1
init_instruments, init_numeraire, book = books.random_put_call_book(
    1.25, instrument_dim, instrument_dim, instrument_dim, 70)
batch_size, timesteps = int(2**18), 12
risk_neutral = True

time, instruments, numeraire, payoff, dpayoff = book.gradient_payoff(
    init_instruments, init_numeraire, batch_size, timesteps, risk_neutral)

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

mlsr = gradient_models.MemorylessSemiRecurrent(timesteps, instrument_dim, 4, 10)
mlsr.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
             loss="mean_squared_error")

mlsr.fit(instruments, dpayoff[..., :-1], 1024, 50,
         callbacks=[early_stopping, reduce_lr])

x = tf.cast(tf.tile(tf.linspace(15, 200, 10000)[:, tf.newaxis],
                    [1, timesteps + 1])[:, tf.newaxis, :], FLOAT_DTYPE)
y = mlsr(x) * numeraire[:-1]
z = book.delta(time, x, numeraire) * numeraire
xlim = (50, 150)

for idx in tf.range(timesteps):
    plt.figure()

    plt.scatter(instruments[:, 0, idx], dpayoff[:, 0, idx], s=0.5)
    plt.plot(x[:, 0, idx], y[:, 0, idx], color="black")
    plt.plot(x[:, 0, idx], z[:, 0, idx], "--", color="red")
    plt.xlim(*xlim)
    plt.ylim(0, 1)
    plt.show()