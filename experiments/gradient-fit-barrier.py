# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

import books
import preprocessing
import gradient_driver
import gradient_models

from constants import FLOAT_DTYPE

instrument_dim = 1
train_size, test_size, timesteps = int(2**12), int(2**10), 14
init_instruments, init_numeraire, book = books.simple_barrier_book(
    timesteps / 250, 100, 105, 110, 0.02, 0.05, 0.2, 1, -1)
frequency = 4

driver = gradient_driver.GradientDriver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-4,
    learning_rate_max=1e-2)

driver.add_testcase(
    name="deep memory network",
    model=gradient_models.SemiRecurrentTwinNetwork(
        timesteps, 4, 5, book.instrument_dim),
    normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
    )

driver.train(train_size, 100, int(2**5))
driver.test(test_size)
gradient_driver.barrier_visualiser(driver, test_size)


# xlim = (tf.reduce_min(instruments), tf.reduce_max(instruments))
# x = tf.cast(tf.tile(tf.linspace(*xlim, 10000)[:, tf.newaxis],
#                     [1, timesteps + 1])[:, tf.newaxis, :], FLOAT_DTYPE)
# norm_x, _, _ = normaliser.transform(x[..., :-1], 0, 0)

# _, y, dy = normaliser.inverse_transform(norm_x, *mlr(norm_x))
# z, dz = book.value(time, x, numeraire), book.delta(time, x, numeraire)

# i_crossed = book.derivatives[0]["derivative"].crossed(instruments[:, 0, :])
# x_crossed = book.derivatives[0]["derivative"].crossed(x[:, 0, :])
# names = ["crossed", "non-crossed"]

# for idx in tf.range(timesteps):
#     data = [(i_crossed[..., idx], x_crossed[..., idx]),
#             (~i_crossed[..., idx], ~x_crossed[..., idx])]
#     for name, (i_mask, x_mask) in zip(names, data):
#         # plt.figure()
#         # plt.scatter(tf.boolean_mask(instruments, i_mask, 0)[:, 0, idx],
#         #             tf.boolean_mask(payoff, i_mask, 0),
#         #             s=0.5)
#         # plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
#         #          tf.boolean_mask(y, x_mask, 0)[..., idx],
#         #          color="black")
#         # plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
#         #          tf.boolean_mask(z, x_mask, 0)[..., idx],
#         #          "--",
#         #          color="red")
#         # plt.xlim(*xlim)
#         # plt.show()

#         plt.figure()
#         plt.scatter(tf.boolean_mask(instruments, i_mask, 0)[:, 0, idx],
#                     tf.boolean_mask(dpayoff, i_mask, 0)[:, 0, idx],
#                     s=0.5)
#         plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
#                   tf.boolean_mask(dy, x_mask, 0)[:, 0, idx],
#                   color="black")
#         plt.plot(tf.boolean_mask(x, x_mask, 0)[:, 0, idx],
#                  tf.boolean_mask(dz, x_mask, 0)[:, 0, idx],
#                  "--",
#                  color="red")
#         plt.xlim(*xlim)
# #        plt.savefig(fr"figures/diff-memory-{idx}-{idx}-{name}.pdf")
#         plt.show()
