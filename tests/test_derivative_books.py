# -*- coding: utf-8 -*-
import numpy as np

from unittest import TestCase, skip
from numpy.testing import assert_array_almost_equal

from derivative_books import BlackScholesPutCallBook, black_price, black_delta

# ==============================================================================
# === Blacks formulas
class test_black(TestCase):
    def test_black_univariate(self):
        params = [0.25,
                  np.array([[110]]),
                  np.array([90]),
                  0.05,
                  np.array([0.2]),
                  np.array([1])]

        price_expected = np.array([[21.1765104079965]])
        delta_expected = np.array([[0.985434416336097]])

        price_result = black_price(*params)
        delta_result = black_delta(*params)

        assert_array_almost_equal(price_result, price_expected)
        assert_array_almost_equal(delta_result, delta_expected)

# ==============================================================================
# === BlackScholesPutCallBook
class test_BlackScholesPutCallBook(TestCase):
    def setUp(self):
        maturity = 2.5
        strike = np.array([80, 90, 100])
        drift = np.array([0.02, 0.03])
        rate = 0.01
        diffusion = np.array([[0.15, 0.2, 0.25], [0.2, 0.45, 0.05]])
        put_call = np.array([-1, 1, 1])
        exposure = np.array([1, 1, -1])
        linker = np.array([0, 1, 0])

        self.book = BlackScholesPutCallBook(
            maturity, strike, drift, rate, diffusion, put_call, exposure,
            linker)


    def test_book_value_multivariate(self):
        spot = np.array([[80, 140, 1.23],
                         [90, 60, 1.24],
                         [100, 80, 1.0],
                         [110, 100, 1.84]])
        time = 0.25

        prices_expected = np.zeros((spot.shape[0], self.book.get_book_size()))
        deltas_expected = np.zeros((spot.shape[0], self.book.get_book_size()))

        for book_idx in range(self.book.get_book_size()):
            for path_idx in range(spot.shape[0]):
                params = [self.book.maturity - time,
                          spot[path_idx,
                               self.book.linker[book_idx]][np.newaxis, np.newaxis],
                          self.book.strike[book_idx][np.newaxis],
                          self.book.rate,
                          self.book.volatility[self.book.linker[book_idx]][np.newaxis],
                          self.book.put_call[book_idx][np.newaxis]
                          ]

                sign = self.book.exposure[book_idx]
                prices_expected[path_idx, book_idx] \
                    = sign * black_price(*params)
                deltas_expected[path_idx, book_idx] \
                    = sign * black_delta(*params)

        prices_result = self.book.marginal_book_value(spot, time)
        deltas_result = self.book.marginal_book_delta(spot, time)

        assert_array_almost_equal(prices_result, prices_expected)
        assert_array_almost_equal(deltas_result, deltas_expected)


    def test_sample_paths(self):
        spot = np.array([85, 95, 1])
        num_paths, num_steps = 2**21, 2
        sample = self.book.sample_paths(spot, num_paths, num_steps, True)

        expected_dims = (num_paths,
                         self.book.get_market_size() + 1,
                         num_steps + 1)
        self.assertTupleEqual(sample.shape, expected_dims)

        payoff = self.book.payoff(sample[:, :, -1])
        self.assertTupleEqual(payoff.shape, (num_paths, ))

        deflator = np.exp(-self.book.rate * self.book.maturity)
        price_result = deflator * payoff.mean(axis=0)
        price_expected = self.book.book_value(spot[np.newaxis, :], 0)

        assert_array_almost_equal(price_result, price_expected, decimal=1)