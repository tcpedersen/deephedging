# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

import hedge_models
import approximators
from constants import FLOAT_DTYPE


class test_OCERiskMeasure(unittest.TestCase):
    def test_ExpectedShortfall(self):
        alpha = 0.95
        risk_measure = hedge_models.ExpectedShortfall(alpha)

        # ES is known in closed form for uniform.
        loss = tf.math.sobol_sample(1, 2**19, dtype=FLOAT_DTYPE)
        var = alpha
        risk_measure.w.assign(var)

        result_fitted = risk_measure(-loss)
        result_evaluate = risk_measure.evaluate(-loss)
        expected = tf.constant([(1 + var) / 2.], FLOAT_DTYPE)

        tf.debugging.assert_near(result_fitted, expected, atol=1e-3)
        tf.debugging.assert_near(result_evaluate, expected, atol=1e-3)
