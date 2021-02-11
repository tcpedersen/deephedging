# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import math

from unittest import TestCase
from numpy.testing import assert_array_almost_equal
from tensorflow.debugging import assert_near

from derivative_books import BlackScholesPutCallBook, black_price, black_delta, random_black_scholes_put_call_book
from constants import FLOAT_DTYPE, NP_FLOAT_DTYPE

# ==============================================================================
# === Blacks formulas
class test_black(TestCase):
    def test_black_univariate(self):
        params = [0.25,
                  np.array([110.]),
                  np.array([90.]),
                  0.05,
                  np.array([0.2]),
                  np.array([1])]

        price_expected = np.array([[[21.1765104079965]]])
        delta_expected = np.array([[[0.985434416336097]]])

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)

# ==============================================================================
# === BlackScholesPutCallBook
class test_BlackScholesPutCallBook(TestCase):
    def lazy_marginal(self, samples, time, book):
        num_samples = samples.shape[0]
        book_size = book.book_size
        num_steps = len(time) - 1

        marginal_prices = np.zeros((num_samples, book_size, num_steps + 1), NP_FLOAT_DTYPE)
        marginal_deltas = np.zeros((num_samples, book_size, num_steps + 1), NP_FLOAT_DTYPE)

        for path_idx in range(num_samples):
            for book_idx in range(book_size):
                for time_idx in range(num_steps + 1):
                    params = [book.maturity - time[time_idx, tf.newaxis],
                              samples[path_idx, book.linker[book_idx], time_idx],
                              book.strike[book_idx, tf.newaxis],
                              book.rate,
                              book.volatility[book.linker[book_idx], tf.newaxis],
                              book.put_call[book_idx, tf.newaxis]
                          ]

                    sign = book.exposure[book_idx]
                    marginal_prices[path_idx, book_idx, time_idx] = sign * tf.squeeze(black_price(*params))
                    marginal_deltas[path_idx, book_idx, time_idx] = sign * tf.squeeze(black_delta(*params))

        prices = tf.reduce_sum(marginal_prices, axis=1)
        deltas = np.zeros((num_samples, book.market_size, num_steps + 1))
        for path_idx in range(num_samples):
            for time_idx in range(num_steps + 1):
                deltas[path_idx, :, time_idx] = np.bincount(
                    book.linker,
                    marginal_deltas[path_idx, :, time_idx],
                    book.market_size)

        out = marginal_prices, marginal_deltas, prices, deltas
        out = [tf.convert_to_tensor(x, FLOAT_DTYPE) for x in out]

        return out


    def test_book_value_multivariate(self):
        init_state, book = random_black_scholes_put_call_book(4, 4, 4, 1)
        time, samples = book.sample_paths(init_state, 3, 5, True)

        marginal_prices_expected, marginal_deltas_expected, prices_expected, deltas_expected = \
            self.lazy_marginal(samples, time, book)

        # test marginals
        tradables = book._get_tradables(samples)
        marginal_prices_result = book._marginal_book_value(tradables, time)
        marginal_deltas_result = book._marginal_book_delta(tradables, time)

        assert_near(marginal_prices_result, marginal_prices_expected)
        assert_near(marginal_deltas_result, marginal_deltas_expected)

        # test book value
        price_result = book.book_value(samples, time)
        delta_result = book.book_delta(samples, time)

        assert_near(price_result, prices_expected)
        assert_near(delta_result, deltas_expected)


    def test_sample_paths(self):
        init_state, book = random_black_scholes_put_call_book(4, 4, 4, 1)

        num_paths, num_steps = 2**21, 2
        time, samples = book.sample_paths(init_state, num_paths, num_steps, True)

        expected_dims = (num_paths,
                         book.market_size + 1,
                         num_steps + 1)
        self.assertTupleEqual(tuple(samples.shape), expected_dims)

        payoff = book.payoff(samples)
        self.assertTupleEqual(tuple(payoff.shape), (num_paths, ))

        deflator = math.exp(-book.rate * book.maturity)
        price_result = deflator * tf.reduce_mean(payoff, axis=0)
        price_expected = book.book_value(init_state[tf.newaxis, :, tf.newaxis], 0)

        assert_near(price_result[tf.newaxis, tf.newaxis], price_expected, atol=2)