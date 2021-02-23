# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

from books import random_black_scholes_put_call_book

class test_GBM(unittest.TestCase):
    def test_moments(self):
        batch_size, num_steps = 2**21, 3

        init_state, book = random_black_scholes_put_call_book(
            num_steps / 250, 1, 3, 4, 69)

        time, paths = book.sample_paths(
            init_state, batch_size, num_steps, False)

        m = (book.drift - book.volatility**2 / 2.)[..., tf.newaxis] * time
        vsq = tf.square(book.volatility)[..., tf.newaxis] * time

        returns = tf.math.log(paths / paths[..., 0, tf.newaxis])
        mean, variance = tf.nn.moments(returns, 0)

        tf.debugging.assert_near(mean[:book.instrument_dim, ...], m, atol=1e-3)
        tf.debugging.assert_near(variance[:book.instrument_dim, ...], vsq, atol=1e-3)