# -*- coding: utf-8 -*-
import tensorflow as tf
import itertools

from unittest import TestCase
from tensorflow.debugging import assert_near

from books import random_put_call_book, simple_put_call_book, random_barrier_book
from constants import FLOAT_DTYPE, INT_DTYPE

# ==============================================================================
# === BlackScholesPutCallBook
class test_BlackScholesPutCallBook(TestCase):
    def lazy_marginal(self, instruments, numeraire, time, book):
        batch_size = instruments.shape[0]
        book_size = book.book_size
        timesteps = len(time) - 1

        marginal_prices = tf.zeros((batch_size, book_size, timesteps + 1),
                                   FLOAT_DTYPE)
        marginal_deltas = tf.zeros((batch_size, book_size, timesteps + 1),
                                   FLOAT_DTYPE)
        grid = tf.constant(list(itertools.product(
            range(batch_size), range(book_size), range(timesteps + 1))))

        for book_idx in tf.range(book_size):
            entry = book.derivatives[book_idx]
            instrument = instruments[:, entry["link"], :]
            sign = entry["exposure"]
            value = sign * entry["derivative"].value(time, instrument, numeraire)
            delta = sign * entry["derivative"].delta(time, instrument, numeraire)

            indices = tf.boolean_mask(grid, grid[:, 1] == book_idx)
            marginal_prices = tf.tensor_scatter_nd_add(
                marginal_prices, indices, tf.reshape(value, -1))
            marginal_deltas = tf.tensor_scatter_nd_add(
                marginal_deltas, indices, tf.reshape(delta, -1))

        prices = tf.reduce_sum(marginal_prices, axis=1)

        deltas = tf.zeros((batch_size, book.instrument_dim, timesteps + 1))
        links = [entry["link"] for entry in book.derivatives]
        for path_idx in range(batch_size):
            for time_idx in range(timesteps + 1):
                indices = tf.stack([
                    path_idx * tf.ones(book.instrument_dim, INT_DTYPE),
                    tf.range(book.instrument_dim),
                    time_idx * tf.ones(book.instrument_dim, INT_DTYPE)], 1)

                updates = tf.math.bincount(
                    links,
                    marginal_deltas[path_idx, :, time_idx],
                    book.instrument_dim)

                deltas = tf.tensor_scatter_nd_add(deltas, indices, updates)

        return prices, deltas


    def test_value_delta_univariate(self):
        init_instruments, init_numeraire, book = simple_put_call_book(
            1., 100., 105., 0.05, 0.1, 0.2, 1.)

        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, 1, 1, True)

        value_result = book.value(time, instruments, numeraire)
        delta_result = book.delta(time, instruments, numeraire)

        value_expected, delta_expected = self.lazy_marginal(
            instruments, numeraire, time, book)

        assert_near(value_result, value_expected)
        assert_near(delta_result, delta_expected)


    def test_value_delta_multivariate(self):
        init_instruments, init_numeraire, book = \
            random_put_call_book(1.25, 10, 4, 3, 69)
        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, 3, 5, True)

        value_expected, delta_expected = self.lazy_marginal(
            instruments, numeraire, time, book)

        value_result = book.value(time, instruments, numeraire)
        delta_result = book.delta(time, instruments, numeraire)

        assert_near(value_result, value_expected)
        assert_near(delta_result, delta_expected)


    def test_sample_paths(self):
        one_dim = random_put_call_book(2.5, 1, 1, 1, 56)
        multi_dim = random_put_call_book(0.5, 4, 4, 1, 69)

        for init_instruments, init_numeraire, book in [one_dim, multi_dim]:
            num_paths, num_steps = 2**20, 2
            time, instruments, numeraire = book.sample_paths(
                init_instruments, init_numeraire, num_paths, num_steps, True)

            expected_dims = (num_paths, book.instrument_dim, num_steps + 1)
            self.assertTupleEqual(tuple(instruments.shape), expected_dims)

            expected_dims = (num_steps + 1, )
            self.assertTupleEqual(tuple(numeraire.shape), expected_dims)

            payoff = book.payoff(time, instruments, numeraire)
            self.assertTupleEqual(tuple(payoff.shape), (num_paths, ))

            price_result = tf.reduce_mean(payoff)
            price_expected = book.value(time, instruments, numeraire)[0, 0]

            # convergence very slow
            assert_near(price_result, price_expected, atol=1e-1)


class test_random_books(TestCase):
    def test_random_barrier_book(self):
        init_instruments, init_numeraire, book = random_barrier_book(
            1.25, 1000, 100, 100, 69)
        time, instruments, numeraire = book.sample_paths(
            init_instruments, init_numeraire, 3, 2, False)

        for entry in book.derivatives:
            linked = instruments[:, entry["link"], :]
            marginal = entry["derivative"].value(time, linked, numeraire)

            message = f"entry: {entry}"
            tf.debugging.assert_positive(marginal[..., 0], message)