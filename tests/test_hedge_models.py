# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf
import books

import hedge_models
import approximators
from constants import FLOAT_DTYPE


class test_LinearFeatureHedge(unittest.TestCase):
    def test_cost_gradient(self):
        timesteps = 3
        dimension = 2
        num = 1
        cost = True

        init_instruments, init_numeraire, book = books.random_put_call_book(
            timesteps / 250, dimension * 2, dimension, dimension, num)
        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, int(2**10), timesteps, True)

        martingales = instruments / numeraire
        features = tf.unstack(book.delta(time, instruments, numeraire) \
                              * numeraire,
                              axis=-1)[:-1]
        payoff = book.payoff(time, instruments, numeraire)

        model = hedge_models.LinearFeatureHedge(
            timesteps=timesteps,
            instrument_dim=book.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] * (1 + cost)
            )
        model.add_cost(1 / 100)
        model.compile(risk_measure=hedge_models.ExpectedShortfall(0.95))

        with tf.GradientTape() as tape:
            value, costs = model([features, martingales], training=True)
            wealth = value - costs - payoff
            loss = model.risk_measure(wealth)

        trainable_vars = [model.risk_measure.w] + model.trainable_variables
        gradient_expected = tape.gradient(loss, trainable_vars)
        gradient_result, wealth = model.cost_gradient(
            ([features, martingales, payoff],))


        for x1, x2 in zip(gradient_result, gradient_expected):
            tf.debugging.assert_near(x1, x2)


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
