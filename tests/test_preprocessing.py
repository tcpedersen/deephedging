# -*- coding: utf-8 -*-
import tensorflow as tf

from unittest import TestCase
from tensorflow.debugging import assert_near
import tensorflow_probability as tfp

import preprocessing
from constants import FLOAT_DTYPE

def get_degenerate_sample(batch, height, width, degenerate, seed):
    tf.random.set_seed(seed)
    z = tf.random.normal((batch, height, width), -2., 10, FLOAT_DTYPE)
    A = tf.random.uniform((height, height))
    x = A @ z # random covariance

    for h in tf.range(height):
        mask = tf.equal(x, x[..., degenerate][:, h, tf.newaxis, tf.newaxis])
        x = tf.where(mask, h.numpy(), x)

    return x


def get_low_variance_sample(batch, height, width, seed):
    tf.random.set_seed(seed)
    collector_out = []

    for i in range(height):
        collector_in = []
        for j in range(width):
            min_ = tf.random.uniform((), 1000, 1000.1, FLOAT_DTYPE)
            max_ = tf.random.uniform((), 1000.1, 1000.2, FLOAT_DTYPE)
            vals = tf.random.uniform((batch, ), min_, max_, FLOAT_DTYPE)

            collector_in.append(vals)
        collector_out.append(collector_in)
    raw = tf.convert_to_tensor(collector_out, FLOAT_DTYPE)

    return tf.transpose(raw, [2, 0, 1])


class test_MeanVarianceNormaliser(TestCase):
    def test_degenerate_col(self):
        batch, height, width, degenerate = 6, 7, 8, 4
        x = get_degenerate_sample(batch, height, width, degenerate, 69)

        normaliser = preprocessing.MeanVarianceNormaliser()
        xn = normaliser.fit_transform(x)

        assert_near(tf.reduce_mean(xn, 0), tf.zeros_like(x[0, ...]))

        # test non-degenerate
        mask = tf.equal(x[0, ...], x[0, ..., degenerate, tf.newaxis])
        expected_var = tf.ones_like(x[0, ...])
        expected_var = tf.where(mask, 0., expected_var)
        assert_near(tf.math.reduce_variance(xn, 0), expected_var)

        y = normaliser.inverse_transform(xn)
        assert_near(y, x)


    def test_low_variance(self):
        batch, height, width = 2**20, 3, 2
        x = get_low_variance_sample(batch, height, width, 69)

        normaliser = preprocessing.MeanVarianceNormaliser()
        xn = normaliser.fit_transform(x)

        tf.debugging.assert_type(xn, FLOAT_DTYPE)

        # test with FLOAT_DTYPE
        mean, variance = tf.nn.moments(xn, 0)
        assert_near(mean, tf.zeros_like(x[0, ...]))
        assert_near(variance, tf.ones_like(x[0, ...]), atol=1e-2)

        # test with tf.float64
        xc = tf.cast(x, tf.float64)
        mean, variance = tf.nn.moments(tf.cast(xn, tf.float64), 0)
        assert_near(mean, tf.zeros_like(xc[0, ...]), atol=1e-8)
        assert_near(variance, tf.ones_like(xc[0, ...]), atol=1e-8)

        y = normaliser.inverse_transform(xn)
        assert_near(y, x)


class test_ZeroComponentAnalysis(TestCase):
    def test_degenerate_col(self):
        batch, height, width, degenerate = 2**20, 3, 2, 1
        x = get_degenerate_sample(batch, height, width, degenerate, 69)

        normaliser = preprocessing.ZeroComponentAnalysis()
        xn = normaliser.fit_transform(x)

        # test mean
        assert_near(tf.reduce_mean(xn, 0), tf.zeros_like(x[0, ...]))

        # test covariance
        cov_result = tfp.stats.covariance(xn, sample_axis=0, event_axis=1)

        mask = tf.equal(x[:height, ...], x[:height, ..., degenerate, tf.newaxis])
        cov_expected = tf.transpose(tf.eye(height, batch_shape=(width, )), [1, 2, 0])
        cov_expected = tf.where(mask, 0, cov_expected)

        assert_near(cov_result, cov_expected, atol=1e-4)

        y = normaliser.inverse_transform(xn)
        assert_near(y, x, atol=1e-6)


class test_DifferentialMeanVarianceNormaliser(TestCase):
    def test_normal(self):
        def f(x):
            sqrt2 = tf.sqrt(2.)
            return (tf.square(tf.reduce_sum(x, 1) / sqrt2) - 1.) / sqrt2

        def df(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = f(x)
            return tape.gradient(y, x)

        batch, dimension, timesteps = 2**20, 2, 3

        size = (dimension, timesteps)
        a, b = tf.random.normal(size), tf.abs(tf.random.normal(size))
        z = tf.random.normal((batch, dimension, timesteps))
        z = (z - tf.reduce_mean(z, 0)) / tf.math.reduce_std(z, 0)
        x = b * z + a
        alpha, beta = tf.random.normal((1, timesteps)), \
            tf.abs(tf.random.normal((1, timesteps)))

        with tf.GradientTape() as tape:
            tape.watch(x)
            z = (x - a) / b
            h = beta * f(z) + alpha
        dhdx = tape.gradient(h, x)

        normaliser = preprocessing.DifferentialMeanVarianceNormaliser()
        norm_x, norm_h, norm_dhdx = normaliser.fit_transform(x, h, dhdx)

        tf.debugging.assert_near(norm_x, z)
        tf.debugging.assert_near(norm_h, f(z))
        tf.debugging.assert_near(norm_dhdx, df(z))

        renorm = normaliser.inverse_transform(norm_x, norm_h, norm_dhdx)
        for result, expected in zip(renorm, [x, h, dhdx]):
            tf.debugging.assert_near(result, expected)
