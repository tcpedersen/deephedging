# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf
import numpy as np

from constants import FLOAT_DTYPE
import simulators
import books

class test_GBM(unittest.TestCase):
    def test_moments(self):
        for use_sobol in [True, False]:
            maturity = 0.6
            batch_size, timesteps = 2**20, 3
            init_instruments, init_numeraire, book = \
                books.random_empty_book(maturity, 4, 5, 420)

            simulator = book.instrument_simulator

            time = tf.cast(tf.linspace(0., maturity, timesteps + 1), FLOAT_DTYPE)
            paths = simulator.simulate(
                time=time,
                init_state=init_instruments,
                batch_size=batch_size,
                risk_neutral=False,
                use_sobol=use_sobol,
                skip=0)

            p = simulator.drift - tf.square(simulator.volatility) / 2.
            m = p[..., tf.newaxis] * time[tf.newaxis, ...]
            vsq = tf.square(simulator.volatility)[..., tf.newaxis] \
                * time[tf.newaxis, ...]

            returns = tf.math.log(paths / paths[..., 0, tf.newaxis])
            mean, variance = tf.nn.moments(returns, 0)

            tf.debugging.assert_near(mean, m, atol=1e-3)
            tf.debugging.assert_near(variance, vsq, atol=1e-3)

    def test_correlation(self):
        maturity = 0.6
        batch_size, timesteps = 2**20, 3
        init_instruments, init_numeraire, book = \
            books.random_empty_book(maturity, 4, 5, 420)

        simulator = book.instrument_simulator

        rvs = simulator.rvs(batch_size, timesteps, use_sobol=True, skip=0)

        for k in tf.range(timesteps):
            sample = rvs[:, k, :]
            loc = tf.reduce_mean(sample, 0)
            cov = tf.convert_to_tensor(
                np.cov(sample, rowvar=False), FLOAT_DTYPE)

            tf.debugging.assert_near(loc, tf.zeros_like(init_instruments),
                                     atol=1e-4)
            tf.debugging.assert_near(cov, simulator.correlation, atol=1e-4)


class test_JumpGBM(unittest.TestCase):
    def test_moments(self):
        for risk_neutral in [True, False]:
            maturity = 0.6
            batch_size, timesteps = 2**20, 2
            init_instruments, init_numeraire, _ = books.random_empty_book(
                maturity, 1, 1, 420)

            rate, drift, diffusion = 0.02, [0.05], [[0.1]]
            intensity, jumpsize, jumpvol = 3.4, -0.2, 0.15

            simulator = simulators.JumpGBM(rate, drift, diffusion,
                                           intensity, jumpsize, jumpvol)

            time = tf.cast(tf.linspace(0., maturity, timesteps + 1),
                           FLOAT_DTYPE)
            paths = simulator.simulate(
                time=time,
                init_state=init_instruments,
                batch_size=batch_size,
                risk_neutral=risk_neutral,
                use_sobol=True,
                skip=0)

            m = jumpsize + tf.square(jumpvol) / 2.
            mu = (rate if risk_neutral else drift) \
                - intensity * (tf.math.exp(m) - 1)
            p = mu - tf.square(simulator.volatility) / 2.

            expected_mean = (p + intensity * jumpsize) * time
            expected_var = (simulator.volatility**2 \
                            + intensity * (jumpvol**2 + jumpsize**2)) * time

            returns = tf.math.log(paths[:, 0, :] / paths[:, 0, 0, tf.newaxis])
            mean, variance = tf.nn.moments(returns, 0)

            tf.debugging.assert_near(mean, expected_mean, atol=1e-3)
            tf.debugging.assert_near(variance, expected_var, atol=1e-3)
