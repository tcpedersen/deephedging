# -*- coding: utf-8 -*-
import tensorflow as tf

import unittest
from tensorflow.debugging import assert_near

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


class test_MeanVarianceNormaliser(unittest.TestCase):
    def test_degenerate_col(self):
        batch, height, width, degenerate = 6, 7, 8, 4
        x = get_degenerate_sample(batch, height, width, degenerate, 69)
        inputs = [x[..., 0] for x in tf.split(x, width, axis=-1)]

        normaliser = preprocessing.MeanVarianceNormaliser()
        outputs = normaliser.fit_transform(inputs)

        for step, op in enumerate(outputs):
            assert_near(tf.reduce_mean(op, 0), 0.)

            expected_var = 0. if step == degenerate else 1.
            assert_near(tf.math.reduce_variance(op, 0), expected_var)

        assert_near(normaliser.inverse_transform(outputs), inputs)


    def test_low_variance(self):
        batch, height, width = 2**20, 3, 2
        x = get_low_variance_sample(batch, height, width, 69)
        inputs = [x[..., 0] for x in tf.split(x, width, axis=-1)]

        normaliser = preprocessing.MeanVarianceNormaliser()
        outputs = normaliser.fit_transform(inputs)

        for op in outputs:
            tf.debugging.assert_type(op, FLOAT_DTYPE)

            # test with FLOAT_DTYPE
            mean, variance = tf.nn.moments(op, 0)
            assert_near(mean, 0.)
            assert_near(variance, 1., atol=1e-2)

            # test with tf.float64
            mean, variance = tf.nn.moments(tf.cast(op, tf.float64), 0)
            assert_near(mean, 0., atol=1e-8)
            assert_near(variance, 1., atol=1e-8)

        assert_near(normaliser.inverse_transform(outputs), inputs)


class test_DifferentialMeanVarianceNormaliser(unittest.TestCase):
    @unittest.skip("convergence too slow.")
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



class test_PCA(unittest.TestCase):
    @unittest.skip("inverse_transform seemingly unprecise.")
    def test_PCA(self):
        batch, height, width, degenerate = int(2**10), 7, 8, 4
        x = get_degenerate_sample(batch, height, width, degenerate, 69)
        inputs = [x[..., 0] for x in tf.split(x, width, axis=-1)]

        normaliser = preprocessing.PrincipalComponentAnalysis(0.95)
        outputs = normaliser.fit_transform(inputs)
        reinputs = normaliser.inverse_transform(outputs)

        tf.debugging.assert_near(inputs, reinputs)
