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


    def test_EntropicRisk(self):
        aversion = 2.
        risk_measure = hedge_models.EntropicRisk(aversion)

        # Entropic risk is just log of MGF divided by aversion
        loss = tf.math.sobol_sample(1, 2**19, dtype=FLOAT_DTYPE)
        risk_measure.w.assign(0.)

        result = risk_measure(-loss)
        expected = tf.math.log((tf.exp(aversion) - 1) / aversion) / aversion

        tf.debugging.assert_near(result, expected)


class test_FeatureMap(unittest.TestCase):
    def test_GaussianFeatureMap(self):
        batch, height = 100, 3
        x = tf.random.normal((batch, height))
        layer = approximators.GaussianFeatureMap()

        result = layer(x)

        zipped = zip(layer.center.numpy(), layer.scale.numpy())
        for idx, (c, gamma) in enumerate(zipped):
            expected = tf.exp(-gamma * tf.linalg.norm(x - c, ord=2, axis=-1)**2)
            tf.debugging.assert_near(result[..., idx], expected)