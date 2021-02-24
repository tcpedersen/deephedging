# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from unittest import TestCase
from numpy.testing import assert_array_almost_equal
from tensorflow.debugging import assert_near

from books import black_price, black_delta, \
    random_black_scholes_put_call_book, DerivativeBook, PutCall
from simulators import GBM, ConstantBankAccount
from constants import FLOAT_DTYPE, NP_FLOAT_DTYPE

# ==============================================================================
# === Blacks formulas
class test_black(TestCase):
    def get_formatted_params(self, time, spot, strike, rate, volatility, theta):
        params = [time,
                  np.array([[spot]], NP_FLOAT_DTYPE),
                  np.array([strike], NP_FLOAT_DTYPE),
                  rate,
                  np.array([volatility], NP_FLOAT_DTYPE),
                  np.array([theta], NP_FLOAT_DTYPE)]

        return params


    def test_black_univariate(self):
        time, spot, strike, rate, volatility, theta = 0.25, 110., 90., 0.05, 0.2, 1
        params = self.get_formatted_params(time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[21.1765104079965]])
        delta_expected = np.array([[0.985434416336097]])

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)


    def test_black_call_at_maturity(self):
        time, spot, strike, rate, volatility, theta = 0.0, 110., 90., 0.05, 0.2, 1
        params = self.get_formatted_params(time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[max(spot - strike, 0)]], NP_FLOAT_DTYPE)
        delta_expected = np.array([[(spot - strike) > 0]], NP_FLOAT_DTYPE)

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)


    def test_black_put_at_maturity(self):
        time, spot, strike, rate, volatility, theta = \
            0.0, 110., 90., 0.05, 0.2, -1
        params = self.get_formatted_params(
            time, spot, strike, rate, volatility, theta)

        price_expected = np.array([[max(strike - spot, 0)]], NP_FLOAT_DTYPE)
        delta_expected = np.array([[(strike - spot) > 0]], NP_FLOAT_DTYPE)

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)


# ==============================================================================
# === BlackScholesPutCallBook
class test_BlackScholesPutCallBook(TestCase):
    def lazy_marginal(self, instruments, numeraire, time, book):
        num_samples = instruments.shape[0]
        book_size = book.book_size
        num_steps = len(time) - 1

        marginal_prices = np.zeros((num_samples, book_size, num_steps + 1),
                                   NP_FLOAT_DTYPE)
        marginal_deltas = np.zeros((num_samples, book_size, num_steps + 1),
                                   NP_FLOAT_DTYPE)

        for path_idx in range(num_samples):
            for book_idx in range(book_size):
                for time_idx in range(num_steps + 1):
                    entry = book.derivatives[book_idx]
                    params = [book.maturity - time[time_idx],
                              instruments[path_idx, entry["link"], time_idx],
                              entry["derivative"].strike,
                              entry["derivative"].rate,
                              entry["derivative"].volatility,
                              entry["derivative"].theta
                          ]

                    sign = entry["exposure"]
                    marginal_prices[path_idx, book_idx, time_idx] = sign \
                        * tf.squeeze(black_price(*params))
                    marginal_deltas[path_idx, book_idx, time_idx] = sign \
                        * tf.squeeze(black_delta(*params))

        prices = tf.reduce_sum(marginal_prices, axis=1)

        deltas = np.zeros((num_samples, book.instrument_dim, num_steps + 1))
        links = [entry["link"] for entry in book.derivatives]
        for path_idx in range(num_samples):
            for time_idx in range(num_steps + 1):
                deltas[path_idx, :, time_idx] = np.bincount(
                    links,
                    marginal_deltas[path_idx, :, time_idx],
                    book.instrument_dim)

        out = marginal_prices, marginal_deltas, prices, deltas
        out = [tf.convert_to_tensor(x / numeraire, FLOAT_DTYPE) for x in out]

        return out


    def test_value_delta_univariate(self):
        init_instruments = tf.constant((100., ), FLOAT_DTYPE)
        init_numeraire = tf.constant((1., ), FLOAT_DTYPE)

        instrument_simulator = GBM(0.01, 0.05, [[0.2]])
        numeraire_simulator = ConstantBankAccount(0.01)
        book = DerivativeBook(1.25, instrument_simulator, numeraire_simulator)
        derivative = PutCall(100., 0.01, 0.2, 1.)
        book.add_derivative(derivative, 0, 1)

        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, 1, 1, True)

        value_result = book.value(time, instruments, numeraire)
        delta_result = book.delta(time, instruments, numeraire)

        value_expected = derivative.value(time, instruments, numeraire)
        delta_expected = derivative.delta(time, instruments, numeraire)

        assert_near(value_result, value_expected)
        assert_near(delta_result, delta_expected)

    def test_value_delta_multivariate(self):
        init_instruments, init_numeraire, book = \
            random_black_scholes_put_call_book(1.25, 10, 4, 3, 69)
        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, 3, 5, True)

        marginal_prices_expected, marginal_deltas_expected, prices_expected, \
            deltas_expected = self.lazy_marginal(
                instruments, numeraire, time, book)

        price_result = book.value(time, instruments, numeraire)
        delta_result = book.delta(time, instruments, numeraire)

        assert_near(price_result, prices_expected)
        assert_near(delta_result, deltas_expected)


    def test_sample_paths(self):
        one_dim = random_black_scholes_put_call_book(2.5, 1, 1, 1, 56)
        multi_dim = random_black_scholes_put_call_book(0.5, 4, 4, 1, 69)

        for init_instruments, init_numeraire, book in [one_dim, multi_dim]:
            num_paths, num_steps = 2**20, 2
            time, instruments, numeraire = book.sample_paths(
                init_instruments, init_numeraire, num_paths, num_steps, True)

            expected_dims = (num_paths, book.instrument_dim, num_steps + 1)
            self.assertTupleEqual(tuple(instruments.shape), expected_dims)

            expected_dims = (num_steps + 1, )
            self.assertTupleEqual(tuple(numeraire.shape), expected_dims)

            payoff = book.payoff(instruments, numeraire)
            self.assertTupleEqual(tuple(payoff.shape), (num_paths, ))

            price_result = tf.reduce_mean(payoff)
            price_expected = book.value(time, instruments, numeraire)[0, 0]

            # convergence very slow
            assert_near(price_result, price_expected, atol=1e-1)