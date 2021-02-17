# -*- coding: utf-8 -*-
import tensorflow as tf

from unittest import TestCase
from tensorflow.debugging import assert_near

from utils import MeanVarianceNormaliser
from constants import FLOAT_DTYPE

class test_MeanVarianceNormaliser(TestCase):
    def test_transform(self):
        batch_size, information_dim, num_steps = 6, 7, 8
        degenerate_col = 4
        x = tf.random.uniform((batch_size, information_dim, num_steps), dtype=FLOAT_DTYPE)

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