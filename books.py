# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.random import uniform

import derivatives
from constants import FLOAT_DTYPE, INT_DTYPE
from simulators import Simulator, GBM, ConstantBankAccount

# ==============================================================================
# === Base
class DerivativeBook(object):
    def __init__(self,
                 maturity: float,
                 instrument_simulator: Simulator,
                 numeraire_simulator: Simulator):
        self.maturity = float(maturity)
        for simulator in [instrument_simulator, numeraire_simulator]:
            assert issubclass(type(simulator), Simulator)

        self.instrument_simulator = instrument_simulator
        self.numeraire_simulator = numeraire_simulator

        self.derivatives = []


    @property
    def book_size(self) -> int:
        """Number of derivatives in the book."""
        return len(self.derivatives)


    @property
    def instrument_dim(self) -> int:
        """Number of underlying risky instruments processes."""
        return self.instrument_simulator.dimension


    def add_derivative(self,
                       derivative: derivatives.Derivative,
                       link: int,
                       exposure: float):
        assert issubclass(type(derivative), derivatives.Derivative)
        assert 0 <= int(link) < self.instrument_dim

        entry = {
            "derivative": derivative,
            "link": int(link),
            "exposure": float(exposure)
            }

        self.derivatives.append(entry)


    def link_apply(self, attr, time, instruments, numeraire):
        """Computes attr of derivative for each derivative according to link.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            marginals: (batch_size, book_size, ...)
        """
        marginals = []
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            func = getattr(entry["derivative"], attr)
            marginal = func(time, linked, numeraire)
            marginals.append(marginal * entry["exposure"])

        return tf.stack(marginals, axis=1)


    def bucket(self, marginals):
        """Sums each marginal into a bucket according to link.
        Args:
            marginals: (batch_size, book_size, ...)
        Returns:
            bucket: (batch_size, instrument_dim, ...)
        """
        links = tf.constant([entry["link"] for entry in self.derivatives],
                            INT_DTYPE)

        v = []
        for k in range(self.instrument_dim):
            mask = tf.squeeze(tf.where(links == k), axis=1)
            v.append(tf.reduce_sum(
                tf.gather(marginals, mask, axis=1), axis=1, keepdims=True))

        return tf.concat(v, axis=1)


    def payoff(self, time: tf.Tensor, instruments: tf.Tensor,
               numeraire: tf.Tensor):
        """Computes payoff of book in terms of numeraire for each sample
        in batch
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            payoff: (batch_size, )
        """
        marginals = self.link_apply("payoff", time, instruments, numeraire)

        return tf.reduce_sum(marginals, axis=1)


    def adjoint(self, time: tf.Tensor, instruments: tf.Tensor,
                numeraire: tf.Tensor):
        """Computes adjoint of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, instrument_dim, timesteps + 1)
        """
        marginals = self.link_apply("adjoint", time, instruments, numeraire)
        bucket = self.bucket(marginals)

        return bucket

    def value(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes value of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instrument: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, timesteps + 1)
        """
        marginals = self.link_apply("value", time, instruments, numeraire)

        return tf.reduce_sum(marginals, axis=1)


    def delta(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes delta of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, instrument_dim, timesteps + 1)
        """
        marginals = self.link_apply("delta", time, instruments, numeraire)
        bucket = self.bucket(marginals)

        return bucket


    def discretise_time(self, timesteps):
        return tf.cast(tf.linspace(0., self.maturity, timesteps + 1),
                       FLOAT_DTYPE)


    def sample_numeraire(self, time, init_numeraire, risk_neutral):
        numeraire = self.numeraire_simulator.simulate(
            time, init_numeraire, 1, risk_neutral)

        return tf.squeeze(numeraire)


    def sample_paths(
            self,
            init_instruments: tf.Tensor,
            init_numeraire: tf.Tensor,
            batch_size: int,
            timesteps: int,
            risk_neutral: bool,
            use_sobol: bool=False,
            skip: int=0,
            exploring_loc: float=None,
            exploring_scale: float=None
            ) -> tf.Tensor:
        """Simulate sample paths.
        Args:
            init_instruments: (state_dim, )
            init_numeraire: (1, )
            batch_size: int
            timesteps: int
            risk_neutral: bool
        Returns:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        """
        time = self.discretise_time(timesteps)

        if exploring_loc is not None and exploring_scale is not None:
            exploring = self.exploring_start(
                init_instruments,
                batch_size,
                exploring_loc,
                exploring_scale
                )
        else:
            exploring = init_instruments

        instruments = self.instrument_simulator.simulate(
            time=time,
            init_state=exploring,
            batch_size=batch_size,
            risk_neutral=risk_neutral,
            use_sobol=use_sobol,
            skip=skip)
        numeraire = self.sample_numeraire(time, init_numeraire, risk_neutral)

        return time, instruments, numeraire


    def exploring_start(self, state, batch_size, loc, scale):
        rvs = tf.random.truncated_normal(
            shape=(batch_size, tf.shape(state)[-1]),
            mean=tf.math.log(loc**2 / tf.sqrt(loc**2 + scale**2)),
            stddev=tf.sqrt(tf.math.log(scale**2 / loc**2 + 1)),
            dtype=FLOAT_DTYPE
            )

        return tf.exp(rvs)


# =============================================================================
# === random generation of books
def simple_empty_book(
        maturity: float,
        spot: float,
        rate: float,
        drift: float,
        volatility: float):
    init_instruments = tf.constant((spot, ), FLOAT_DTYPE)
    init_numeraire = tf.constant((1., ), FLOAT_DTYPE)

    instrument_simulator = GBM(rate, drift, [[volatility]])
    numeraire_simulator = ConstantBankAccount(rate)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)

    return init_instruments, init_numeraire, book

def random_empty_book(
        maturity: float,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int,
        sign_prob: float=0.5):
    tf.random.set_seed(seed)

    maturity = float(maturity)
    drift = tf.random.uniform((instrument_dim, ), 0.05, 0.1, dtype=FLOAT_DTYPE)
    rate = tf.random.uniform((1, ), 0.0, 0.05, dtype=FLOAT_DTYPE)

    scale = tf.cast(tf.sqrt(float(num_brownian_motions)), FLOAT_DTYPE)
    size = (instrument_dim, num_brownian_motions)
    diffusion = random_sign(size, sign_prob) \
        * tf.random.uniform(size, 0.15 / scale, 0.4 / scale, FLOAT_DTYPE)

    init_instruments = uniform((instrument_dim, ), 95, 105, FLOAT_DTYPE)
    init_numeraire = uniform((1, ), 0.75, 1.25, FLOAT_DTYPE)

    instrument_simulator = GBM(rate, drift, diffusion)
    numeraire_simulator = ConstantBankAccount(rate)
    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)

    return init_instruments, init_numeraire, book


def random_linker(book_size, instrument_dim):
    if instrument_dim  > 1:
        one_of_each = tf.range(instrument_dim)
        other_random = tf.random.uniform((book_size - instrument_dim, ),
                                         0, instrument_dim - 1, INT_DTYPE)
        return tf.random.shuffle(tf.concat([one_of_each, other_random], 0))
    else:
        return tf.convert_to_tensor((0, ), INT_DTYPE)


def random_sign(size, p):
    unscaled = tfp.distributions.Binomial(1, probs=p).sample(size)

    return tf.cast(2 * unscaled - 1, FLOAT_DTYPE)


def pct_interval(x, p):
    return (1 - p) * x, (1 + p) * x


def random_put_call_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int,
        **kwargs):
    assert book_size >= instrument_dim, "book_size smaller than instrument_dim."

    init_instruments, init_numeraire, book = random_empty_book(
        maturity, instrument_dim, num_brownian_motions, seed)

    strike = uniform((book_size, ), 95, 105, FLOAT_DTYPE)
    put_call = random_sign(book_size, 1 / 2)
    exposure = random_sign(book_size, 3 / 4)

    linker = random_linker(book_size, instrument_dim)

    for idx, link in enumerate(linker):
        vol = book.instrument_simulator.volatility[link]
        derivative = derivatives.PutCall(
            maturity,
            strike[idx],
            book.numeraire_simulator.rate,
            vol,
            put_call[idx]
            )
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def simple_put_call_book(maturity, spot, strike, rate, drift, sigma, theta):
    init_instruments, init_numeraire, book = simple_empty_book(
        maturity, spot, rate, drift, sigma)

    derivative = derivatives.PutCall(maturity, strike, rate, sigma, theta)
    book.add_derivative(derivative, 0, 1)

    return init_instruments, init_numeraire, book


def random_barrier_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int):
    assert book_size >= instrument_dim, "book_size smaller than instrument_dim."

    init_instruments, init_numeraire, book = random_empty_book(
        maturity, instrument_dim, num_brownian_motions, seed)

    strike = uniform((book_size, ), 95, 105, FLOAT_DTYPE)
    exposure = random_sign(book_size, 3 / 4)

    outin = random_sign(book_size, 1 / 2)
    updown = random_sign(book_size, 1 / 2)

    linker = random_linker(book_size, instrument_dim)

    for idx, link in enumerate(linker):
        call_min_barrier = init_instruments[link] - 10
        call_max_barrier = init_instruments[link] + 10
        if outin[idx] == 1:
            if updown[idx] == 1:
                lower = max(strike[idx], init_instruments[link]) + 1
                upper = call_max_barrier
            elif updown[idx] == -1:
                lower = call_min_barrier
                upper = min(strike[idx], init_instruments[link]) - 1
        elif outin[idx] == -1:
            if updown[idx] == 1:
                lower = max(strike[idx], init_instruments[link]) + 1
                upper = call_max_barrier
            elif updown[idx] == -1:
                lower = call_min_barrier
                upper = init_instruments[link]

        barrier = uniform((1, ), lower, upper, FLOAT_DTYPE)

        vol = book.instrument_simulator.volatility[link]
        derivative = derivatives.BarrierCall(
            maturity, strike[idx], barrier, book.rate, vol, outin[idx],
            updown[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def simple_barrier_book(maturity, spot, strike, barrier, rate, drift, vol,
                        outin, updown):
    init_instruments, init_numeraire, book = simple_empty_book(
        maturity, spot, rate, drift, vol)

    derivative = derivatives.BarrierCall(
        maturity, strike, barrier, rate, vol, outin, updown)
    book.add_derivative(derivative, 0, 1)

    return init_instruments, init_numeraire, book


def random_dga_putcall_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int,
        **kwargs):
    assert book_size >= instrument_dim, "book_size smaller than instrument_dim."

    init_instruments, init_numeraire, book = random_empty_book(
        maturity, instrument_dim, num_brownian_motions, seed, **kwargs)

    put_call = random_sign(book_size, 1 / 2)
    exposure = random_sign(book_size, 3 / 4)

    linker = random_linker(book_size, instrument_dim)

    for idx, link in enumerate(linker):
        expected = init_instruments[link]**maturity
        strike = uniform((1, ), *pct_interval(expected, 0.1), FLOAT_DTYPE)
        vol = book.instrument_simulator.volatility[link]
        derivative = derivatives.DiscreteGeometricPutCall(
            maturity, strike, book.numeraire_simulator.rate, vol, put_call[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def simple_dga_putcall_book(maturity, spot, strike, rate, drift, sigma, theta):
    init_instruments, init_numeraire, book = simple_empty_book(
        maturity, spot, rate, drift, sigma)

    derivative = derivatives.DiscreteGeometricPutCall(
        maturity, strike, rate, sigma, theta)
    book.add_derivative(derivative, 0, 1)

    return init_instruments, init_numeraire, book


def random_mean_putcall_book(maturity, dimension, seed):
    init_instruments, init_numeraire, book = random_empty_book(
        maturity, dimension, dimension, seed)

    init_instruments = tf.ones_like(init_instruments) * 100.
    init_numeraire = tf.ones_like(init_numeraire)
    book.numeraire_simulator.rate = 0.

    for dim in tf.range(dimension):
        derivative = derivatives.PutCall(
            maturity,
            init_instruments[dim],
            book.numeraire_simulator.rate,
            book.instrument_simulator.volatility[dim],
            1
            )
        book.add_derivative(derivative, dim, 1 / dimension)

    return init_instruments, init_numeraire, book
