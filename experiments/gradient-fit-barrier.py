# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

import books
import gradient_models
import utils
import preprocessing

from constants import FLOAT_DTYPE

instrument_dim = 1
batch_size, timesteps = int(2**20), 14
init_instruments, init_numeraire, book = books.random_barrier_book(
    timesteps / 250, instrument_dim, instrument_dim, instrument_dim, 72)
frequency = 2**4

time, instruments, numeraire, payoff, dpayoff = book.gradient_payoff(
    init_instruments=tf.constant([book.derivatives[0]["derivative"].barrier],
                                 tf.float32), #init_instruments,
    init_numeraire=init_numeraire,
    batch_size=batch_size,
    timesteps=timesteps,
    frequency=frequency,
    risk_neutral=True,
    exploring_scale=0.2)

normaliser = preprocessing.DifferentialMeanVarianceNormaliser()
norm_instruments, norm_payoff, norm_dpayoff = \
    normaliser.fit_transform(instruments[..., :-1], payoff, dpayoff[..., :-1])

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

epochs = 100
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    utils.PeakSchedule(1e-3, 0.01, epochs), verbose=0)

mlr = gradient_models.SemiRecurrentDifferentialNetworkTest(
    timesteps=timesteps,
    internal_dim=instrument_dim,
    num_layers=2,
    num_units=15)

w = 0.5
mlr.compile(optimizer=tf.keras.optimizers.Adam(),
            loss="mean_squared_error",
            loss_weights=[w, 1 - w])

mlr.fit(norm_instruments, [norm_payoff, norm_dpayoff], 2**10, epochs,
        callbacks=[early_stopping, reduce_lr])

xlim = (tf.reduce_min(instruments), tf.reduce_max(instruments))
x = tf.cast(tf.tile(tf.linspace(*xlim, 10000)[:, tf.newaxis],
                    [1, timesteps + 1])[:, tf.newaxis, :], FLOAT_DTYPE)
norm_x, _, _ = normaliser.transform(x[..., :-1], 0, 0)

_, y, dy = normaliser.inverse_transform(norm_x, *mlr(norm_x))
z, dz = book.value(time, x, numeraire), book.delta(time, x, numeraire)

i_crossed = book.derivatives[0]["derivative"].crossed(instruments[:, 0, :])
x_crossed = book.derivatives[0]["derivative"].crossed(x[:, 0, :])
names = ["crossed", "non-crossed"]

for idx in tf.range(timesteps):
    data = [(i_crossed[..., idx], x_crossed[..., idx]),
            (~i_crossed[..., idx], ~x_crossed[..., idx])]
    for name, (i_mask, x_mask) in zip(names, data):
        # plt.figure()
        # plt.scatter(tf.boolean_mask(instruments, i_mask, 0)[:, 0, idx],
        #             tf.boolean_mask(payoff, i_mask, 0),
        #             s=0.5)
        # plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
        #          tf.boolean_mask(y, x_mask, 0)[..., idx],
        #          color="black")
        # plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
        #          tf.boolean_mask(z, x_mask, 0)[..., idx],
        #          "--",
        #          color="red")
        # plt.xlim(*xlim)
        # plt.show()

        plt.figure()
        plt.scatter(tf.boolean_mask(instruments, i_mask, 0)[:, 0, idx],
                    tf.boolean_mask(dpayoff, i_mask, 0)[:, 0, idx],
                    s=0.5)
        plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
                  tf.boolean_mask(dy, x_mask, 0)[:, 0, idx],
                  color="black")
        plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
                 tf.boolean_mask(dz, x_mask, 0)[:, 0, idx],
                 "--",
                 color="red")
        plt.xlim(*xlim)
#        plt.savefig(fr"figures/diff-memory-{idx}-{idx}-{name}.pdf")
        plt.show()
