# -*- coding: utf-8 -*-
import tensorflow as tf

from unittest import TestCase
from tensorflow.debugging import assert_near

from utils import MeanVarianceNormaliser
from constants import FLOAT_DTYPE

class test_MeanVarianceNormaliser(TestCase):
    def test_degenerate_col(self):
        batch_size, information_dim, num_steps = 6, 7, 8
        degenerate_col = 4
        x = tf.random.uniform((batch_size, information_dim, num_steps),
                              -2., 10, FLOAT_DTYPE)

        # ensure one row degenerate
        mask = tf.equal(x, x[..., degenerate_col, None])
        x = tf.where(mask, 1.5, x)

        normaliser = MeanVarianceNormaliser()
        xn = normaliser.fit_transform(x)

        assert_near(tf.reduce_mean(xn, 0), tf.zeros_like(x[0, ...]))

        expected_var = tf.ones_like(x[0, ...])
        expected_var = tf.where(mask, 0., expected_var)
        assert_near(tf.math.reduce_variance(xn, 0), expected_var)

        y = normaliser.inverse_transform(xn)
        assert_near(y, x)


    def test_low_variance(self):
        batch, height, width = 2**20, 4, 3
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
        x = tf.transpose(raw, [2, 0, 1])

        normaliser = MeanVarianceNormaliser()
        xn = normaliser.fit_transform(x)

        tf.debugging.assert_type(xn, FLOAT_DTYPE)

        # test with FLOAT_DTYPE
        mean, variance = tf.nn.moments(xn, 0)
        assert_near(mean, tf.zeros_like(x[0, ...]))
        assert_near(variance, tf.ones_like(x[0, ...]), atol=1e-2)

        # test with tf.float64
        xc = tf.cast(x, tf.float64)
        mean, variance = tf.nn.moments(tf.cast(xn, tf.float64), 0)
        assert_near(mean, tf.zeros_like(xc[0, ...]), atol=1e-9)
        assert_near(variance, tf.ones_like(xc[0, ...]), atol=1e-8)

        y = normaliser.inverse_transform(xn)
        assert_near(y, x)
