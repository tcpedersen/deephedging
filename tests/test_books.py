# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import math

from unittest import TestCase
from numpy.testing import assert_array_almost_equal
from tensorflow.debugging import assert_near

from books import black_price, black_delta, random_black_scholes_put_call_book, random_simple_put_call_book
from constants import FLOAT_DTYPE, NP_FLOAT_DTYPE

# ==============================================================================
# === Blacks formulas
class test_black(TestCase):
    def get_formatted_params(self, time, spot, strike, rate, volatility, theta):
        params = [time,
                  np.array([spot], NP_FLOAT_DTYPE),
                  np.array([strike], NP_FLOAT_DTYPE),
                  rate,
                  np.array([volatility], NP_FLOAT_DTYPE),
                  np.array([theta], NP_FLOAT_DTYPE)]

        return params


    def test_black_univariate(self):
        time, spot, strike, rate, volatility, theta = 0.25, 110., 90., 0.05, 0.2, 1
        params = self.get_formatted_params(time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[[21.1765104079965]]])
        delta_expected = np.array([[[0.985434416336097]]])

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)


    def test_black_call_at_maturity(self):
        time, spot, strike, rate, volatility, theta = 0.0, 110., 90., 0.05, 0.2, 1
        params = self.get_formatted_params(time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[[max(spot - strike, 0)]]], NP_FLOAT_DTYPE)
        delta_expected = np.array([[[(spot - strike) > 0]]], NP_FLOAT_DTYPE)

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)


    def test_black_put_at_maturity(self):
        time, spot, strike, rate, volatility, theta = 0.0, 110., 90., 0.05, 0.2, -1
        params = self.get_formatted_params(time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[[max(strike - spot, 0)]]], NP_FLOAT_DTYPE)
        delta_expected = np.array([[[(strike - spot) > 0]]], NP_FLOAT_DTYPE)

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
        deltas = np.zeros((num_samples, book.instrument_dim, num_steps + 1))
        for path_idx in range(num_samples):
            for time_idx in range(num_steps + 1):
                deltas[path_idx, :, time_idx] = np.bincount(
                    book.linker,
                    marginal_deltas[path_idx, :, time_idx],
                    book.instrument_dim)

        out = marginal_prices, marginal_deltas, prices, deltas
        out = [tf.convert_to_tensor(x, FLOAT_DTYPE) for x in out]

        return out


    def test_book_value_multivariate(self):
        init_state, book = random_black_scholes_put_call_book(4, 4, 4, 1)
        time, samples = book.sample_paths(init_state, 3, 5, True)

        marginal_prices_expected, marginal_deltas_expected, prices_expected, deltas_expected = \
            self.lazy_marginal(samples, time, book)

        # test marginals
        instruments = book._get_instruments(samples)
        marginal_prices_result = book._marginal_book_value(instruments, time)
        marginal_deltas_result = book._marginal_book_delta(instruments, time)

        assert_near(marginal_prices_result, marginal_prices_expected)
        assert_near(marginal_deltas_result, marginal_deltas_expected)

        # test book value
        price_result = book.book_value(samples, time)
        delta_result = book.book_delta(samples, time)

        assert_near(price_result, prices_expected)
        assert_near(delta_result, deltas_expected)


    def test_sample_paths(self):
        for init_state, book in [random_black_scholes_put_call_book(4, 4, 4, 1),
                                 random_simple_put_call_book(3.)]:

            num_paths, num_steps = 2**21, 2
            time, samples = book.sample_paths(init_state, num_paths, num_steps, True)

            expected_dims = (num_paths,
                             book.instrument_dim + 1,
                             num_steps + 1)
            self.assertTupleEqual(tuple(samples.shape), expected_dims)

            payoff = book.payoff(samples)
            self.assertTupleEqual(tuple(payoff.shape), (num_paths, ))

            deflator = math.exp(-book.rate * book.maturity)
            price_result = deflator * tf.reduce_mean(payoff, axis=0)
            price_expected = book.book_value(init_state[tf.newaxis, :, tf.newaxis], 0)

            assert_near(price_result[tf.newaxis, tf.newaxis], price_expected, atol=2)