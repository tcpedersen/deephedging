# -*- coding: utf-8 -*-
import tensorflow as tf

import books
import simulators
import derivatives

from constants import FLOAT_DTYPE

# =============================================================================
# === helpers
def _fit_to_dimension(dimension, vector):
    if tf.size(vector) == 1:
        vector = tf.tile([float(vector)], (dimension, ))
    elif tf.size(vector) != dimension:
        raise ValueError(f"dimension {dimension} and dimension of vector "
                         "{tf.size(vector)} is not equal.")

    return vector


def random_empty_book(maturity, dimension, rate, drift, volatility, seed=None):
    tf.random.set_seed(seed)
    rate = float(rate)
    drift = _fit_to_dimension(dimension, drift)
    volatility = _fit_to_dimension(dimension, volatility)

    # scale diffusion to have row norms equal to volatility
    diffusion = tf.random.uniform((dimension, dimension), minval=-1, maxval=1)
    norm = tf.linalg.norm(diffusion, axis=1)
    diffusion *= (volatility / norm)[:, tf.newaxis]

    init_instruments = _fit_to_dimension(dimension, 100.0)
    init_numeraire = tf.constant([1.0], FLOAT_DTYPE)

    instrument_simulator = simulators.GBM(rate, drift, diffusion)
    numeraire_simulator = simulators.ConstantBankAccount(rate)
    book = books.DerivativeBook(
        maturity, instrument_simulator, numeraire_simulator)

    return init_instruments, init_numeraire, book


def random_basket_book(maturity, dimension, volatility, seed=None):
    tf.random.set_seed(seed)
    volatility = _fit_to_dimension(dimension, volatility)
    spot = strike = 1.0

    # scale diffusion to have row norms equal to volatility
    diffusion = tf.random.uniform((dimension, dimension), minval=-1, maxval=1)
    norm = tf.linalg.norm(diffusion, axis=1)
    diffusion *= (volatility / norm)[:, tf.newaxis]

    init_instruments = _fit_to_dimension(dimension, spot)
    init_numeraire = tf.constant([1.0], FLOAT_DTYPE)

    instrument_simulator = simulators.BrownianMotion(diffusion)
    numeraire_simulator = simulators.ConstantBankAccount(0.0)
    book = books.BasketBook(
        maturity, instrument_simulator, numeraire_simulator)

    weights = tf.random.uniform((dimension, ))
    derivative = derivatives.BachelierBasketCall(
        maturity, strike, diffusion, weights)
    book.add_derivative(derivative, 0, 1.0)

    return init_instruments, init_numeraire, book


def add_butterfly(init_instruments, book, spread):
    dimension = book.instrument_dim
    if dimension != len(init_instruments):
        raise ValueError("wrong dimension of init_instruments.")
    for link, spot in enumerate(init_instruments):
        itm = derivatives.PutCall(
            book.maturity,
            spot - spread / 2,
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            1)
        atm = derivatives.PutCall(
            book.maturity,
            spot,
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            1)
        otm = derivatives.PutCall(
            book.maturity,
            spot + spread / 2,
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            1)

        book.add_derivative(itm, link, 1 / dimension)
        book.add_derivative(atm, link, -2 / dimension)
        book.add_derivative(otm, link, 1 / dimension)


def add_dga_calls(init_instruments, book):
    dimension = book.instrument_dim
    if dimension != len(init_instruments):
        raise ValueError("wrong dimension of init_instruments.")
    strikes = tf.pow(init_instruments, book.maturity)

    for link, k in enumerate(strikes):
        atm = derivatives.DiscreteGeometricPutCall(
            book.maturity,
            k,
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            1
            )

        book.add_derivative(atm, link, 1 / dimension)


def add_rko(init_instruments, book, spread):
    dimension = book.instrument_dim
    if dimension != len(init_instruments):
        raise ValueError("wrong dimension of init_instruments.")

    for link, spot in enumerate(init_instruments):
        rko = derivatives.BarrierCall(
            book.maturity,
            spot,
            spot + spread,
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            "out",
            "up"
            )

        book.add_derivative(rko, link, 1 / dimension)


def add_calls(init_instruments, book):
    dimension = book.instrument_dim
    if dimension != len(init_instruments):
        raise ValueError("wrong dimension of init_instruments.")

    for link in tf.range(dimension):
        derivative = derivatives.PutCall(
            book.maturity,
            init_instruments[link],
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[link],
            1
            )

        book.add_derivative(derivative, link, 1 / dimension)


def add_random_putcalls(init_instruments, book, number_of_derivatives=3):
    dimension = book.instrument_dim
    if dimension != len(init_instruments):
        raise ValueError("wrong dimension of init_instruments.")

    for link in tf.range(dimension):
        for _ in tf.range(number_of_derivatives):
            spot = init_instruments[link]
            strike = tf.random.uniform((1, ), minval=spot - 10.0,
                                       maxval=spot + 10.0)
            theta = tf.sign(tf.random.uniform((1, )) - 0.5)

            derivative = derivatives.PutCall(
                book.maturity,
                strike,
                book.numeraire_simulator.rate,
                book.instrument_simulator.volatility[link],
                theta
                )

            book.add_derivative(derivative, link, 1 / dimension)
