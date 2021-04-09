# -*- coding: utf-8 -*-
import tensorflow as tf
import books

import utils

timesteps = 250
# init_instruments, init_numeraire, book = books.simple_barrier_book(
#     timesteps / 250, 100, 100, 95, 0.02, 0.05, 0.2, -1, -1)
init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
    timesteps / 250, 100, 110, 0.02, 0.05, 0.2, 1)

time, instruments, numeraire = book.sample_paths(
    init_instruments,
    init_numeraire,
    int(2**0),
    timesteps,
    True,
    use_sobol=False
    )

delta = book.delta(time, instruments[0, tf.newaxis, ...], numeraire)
adjoint = book.adjoint(time, instruments, numeraire)
clean = tf.boolean_mask(
    adjoint,
    tf.reduce_all(tf.math.is_finite(adjoint), axis=[1, 2]),
    0)



print(delta[0, 0, 0])
print(utils.precise_mean(clean, axis=0)[0, 0])
print(utils.predice_confidence_interval(clean[:, 0, 0], 0.95))