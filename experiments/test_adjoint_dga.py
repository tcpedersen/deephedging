# -*- coding: utf-8 -*-
import tensorflow as tf
import books

import utils
from constants import FLOAT_DTYPE

timesteps = 2
# init_instruments, init_numeraire, book = books.simple_barrier_book(
#     timesteps / 250, 100, 100, 95, 0.02, 0.05, 0.2, -1, -1)
init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
    500 / 250, 100, 100, 0.05, 0.05, 0.2, 1)

time, instruments, numeraire = book.sample_paths(
    init_instruments,
    init_numeraire,
    int(2**24),
    timesteps,
    True,
    use_sobol=True
    )

delta = book.delta(time, instruments[0, tf.newaxis, ...], numeraire)
adjoint = book.adjoint(time, instruments, numeraire)
clean = tf.boolean_mask(
    adjoint,
    tf.reduce_all(tf.math.is_finite(adjoint), axis=[1, 2]),
    0)



print(delta[0, 0, 0])
print(utils.precise_mean(clean, axis=0)[0, 0])
