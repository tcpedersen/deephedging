# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

from constants import FLOAT_DTYPE
from simulators import GBM
from books import random_black_scholes_parameters

class test_GBM(unittest.TestCase):
    def test_moments(self):
        maturity = 0.6
        batch_size, time_steps = 2**20, 3
        init_instruments, init_numeraire, drift, rate, diffusion = \
            random_black_scholes_parameters(maturity, 4, 5, 420)

        simulator = GBM(rate, drift, diffusion)

        time = tf.cast(tf.linspace(0., maturity, time_steps + 1), FLOAT_DTYPE)
        paths = simulator.simulate(time, init_instruments, batch_size, False)

        m = (drift - tf.square(simulator.volatility) / 2.)[..., tf.newaxis] * time[tf.newaxis, ...]
        vsq = tf.square(simulator.volatility)[..., tf.newaxis] * time[tf.newaxis, ...]

        returns = tf.math.log(paths / paths[..., 0, tf.newaxis])
        mean, variance = tf.nn.moments(returns, 0)

        tf.debugging.assert_near(mean, m, atol=1e-3)
        tf.debugging.assert_near(variance, vsq, atol=1e-3)