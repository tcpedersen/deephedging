# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf
import numpy as np

from constants import FLOAT_DTYPE
from simulators import GBM
from books import random_black_scholes_parameters

class test_GBM(unittest.TestCase):
    def test_moments(self):
        for use_sobol in [True, False]:
            maturity = 0.6
            batch_size, time_steps = 2**20, 3
            init_instruments, init_numeraire, drift, rate, diffusion = \
                random_black_scholes_parameters(maturity, 4, 5, 420)

            simulator = GBM(rate, drift, diffusion)

            time = tf.cast(tf.linspace(0., maturity, time_steps + 1), FLOAT_DTYPE)
            paths = simulator.simulate(
                time=time,
                init_state=init_instruments,
                batch_size=batch_size,
                risk_neutral=False,
                use_sobol=use_sobol,
                skip=0)

            m = (drift - tf.square(simulator.volatility) / 2.)[..., tf.newaxis] \
                * time[tf.newaxis, ...]
            vsq = tf.square(simulator.volatility)[..., tf.newaxis] \
                * time[tf.newaxis, ...]

            returns = tf.math.log(paths / paths[..., 0, tf.newaxis])
            mean, variance = tf.nn.moments(returns, 0)

            tf.debugging.assert_near(mean, m, atol=1e-3)
            tf.debugging.assert_near(variance, vsq, atol=1e-3)

    def test_correlation(self):
        maturity = 0.6
        batch_size, timesteps = 2**20, 3
        init_instruments, init_numeraire, drift, rate, diffusion = \
            random_black_scholes_parameters(maturity, 4, 5, 420)

        simulator = GBM(rate, drift, diffusion)

        rvs = simulator.rvs(batch_size, timesteps, use_sobol=True, skip=0)

        for k in tf.range(timesteps):
            sample = rvs[:, k, :]
            loc = tf.reduce_mean(sample, 0)
            cov = tf.convert_to_tensor(
                np.cov(sample, rowvar=False), FLOAT_DTYPE)

            tf.debugging.assert_near(loc, tf.zeros_like(init_instruments),
                                     atol=1e-4)
            tf.debugging.assert_near(cov, simulator.correlation, atol=1e-4)
