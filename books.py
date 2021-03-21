# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import abc

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
        payoff = tf.zeros_like(instruments[:, 0, 0], FLOAT_DTYPE)
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            marginal = entry["derivative"].payoff(time, linked, numeraire)
            payoff += marginal * entry["exposure"]

        return payoff


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
        value = tf.zeros_like(instruments[:, 0, :], FLOAT_DTYPE)
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            marginal = entry["derivative"].value(time, linked, numeraire)
            value += marginal * entry["exposure"]

        return value


    def delta(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes value of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, instrument_dim, timesteps + 1)
        """
        marginals = []
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            marginal = entry["derivative"].delta(time, linked, numeraire)
            marginals.append(marginal * entry["exposure"])

        marginals = tf.stack(marginals, axis=1)
        links = tf.constant([entry["link"] for entry in self.derivatives],
                            INT_DTYPE)

        v = []
        for k in range(self.instrument_dim):
            mask = tf.squeeze(tf.where(links == k), axis=1)
            v.append(tf.reduce_sum(
                tf.gather(marginals, mask, axis=1), axis=1, keepdims=True))

        return tf.concat(v, axis=1)


    def discretise_time(self, timesteps):
        return tf.cast(tf.linspace(0., self.maturity, timesteps + 1),
                       FLOAT_DTYPE)


    def sample_numeraire(self, time, init_numeraire, risk_neutral):
        numeraire = self.numeraire_simulator.simulate(
            time, init_numeraire, 1, risk_neutral)

        return tf.squeeze(numeraire)

    def sample_paths(self,
                     init_instruments: tf.Tensor,
                     init_numeraire: tf.Tensor,
                     batch_size: int,
                     timesteps: int,
                     risk_neutral: bool) -> tf.Tensor:
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
        instruments = self.instrument_simulator.simulate(
            time, init_instruments, batch_size, risk_neutral)
        numeraire = self.sample_numeraire(time, init_numeraire, risk_neutral)

        return time, instruments, numeraire


    def _scale_lognormally(self, state, batch_size, scale):
        rvs = tf.random.normal((batch_size, tf.shape(state)[-1]))

        return state * tf.exp(-scale**2 / 2. + scale * rvs)


    def gradient_payoff(self,
                     init_instruments: tf.Tensor,
                     init_numeraire: tf.Tensor,
                     batch_size: int,
                     timesteps: int,
                     frequency: int,
                     risk_neutral: bool,
                     exploring_scale: float=1/5) -> tf.Tensor:
        """Simulate sample paths and compute payoffs with gradients."""
        time = self.discretise_time(timesteps * (1 + frequency))
        numeraire = self.sample_numeraire(time, init_numeraire, risk_neutral)
        init_scaled = self._scale_lognormally(
            init_instruments, batch_size, exploring_scale)
        skip = frequency + 1

        with tf.GradientTape() as tape:
            tape.watch(init_scaled)
            instruments = self.instrument_simulator.simulate(
                time, init_scaled, batch_size, risk_neutral, as_list=True)
            payoff = self.payoff(time, tf.stack(instruments, -1), numeraire)

        gradient = tf.stack(tape.gradient(payoff, instruments[::skip]), -1)

        return time[::skip], tf.stack(instruments[::skip], -1), numeraire[::skip],\
            payoff, gradient


# =============================================================================
# === random generation of books
def random_black_scholes_parameters(
        maturity: int,
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

    return init_instruments, init_numeraire, drift, rate, diffusion


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


def random_put_call_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int,
        **kwargs):
    assert book_size >= instrument_dim, "book_size smaller than instrument_dim."

    init_instruments, init_numeraire, drift, rate, diffusion = \
        random_black_scholes_parameters(
            maturity, instrument_dim, num_brownian_motions, seed, **kwargs)

    instrument_simulator = GBM(rate, drift, diffusion)
    numeraire_simulator = ConstantBankAccount(rate)

    strike = uniform((book_size, ), 95, 105, FLOAT_DTYPE)
    put_call = random_sign(book_size, 1 / 2)
    exposure = random_sign(book_size, 3 / 4)

    linker = random_linker(book_size, instrument_dim)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
    for idx, link in enumerate(linker):
        vol = instrument_simulator.volatility[link]
        derivative = derivatives.PutCall(
            maturity, strike[idx], rate, vol, put_call[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def simple_put_call_book(maturity, spot, strike, rate, drift, sigma, theta):
    init_instruments = tf.constant((spot, ), FLOAT_DTYPE)
    init_numeraire = tf.constant((1., ), FLOAT_DTYPE)

    instrument_simulator = GBM(rate, drift, [[sigma]])
    numeraire_simulator = ConstantBankAccount(rate)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
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

    init_instruments, init_numeraire, drift, rate, diffusion = \
        random_black_scholes_parameters(
            maturity, instrument_dim, num_brownian_motions, seed)

    instrument_simulator = GBM(rate, drift, diffusion)
    numeraire_simulator = ConstantBankAccount(rate)

    strike = uniform((book_size, ), 95, 105, FLOAT_DTYPE)
    exposure = random_sign(book_size, 3 / 4)

    outin = random_sign(book_size, 1 / 2)
    updown = random_sign(book_size, 1 / 2)

    linker = random_linker(book_size, instrument_dim)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
    for idx, link in enumerate(linker):
        call_min_barrier = init_instruments[link] - 10
        call_max_barrier = init_instruments[link] + 10
        if outin[idx] == 1:
            if updown[idx] == 1:
                lower = max(strike[idx], init_instruments[link]) + 1
                upper = call_max_barrier
            elif updown[idx] == -1:
                lower = call_min_barrier
                upper = init_instruments[link]
        elif outin[idx] == -1:
            if updown[idx] == 1:
                lower = init_instruments[link]
                upper = call_max_barrier
            elif updown[idx] == -1:
                lower = call_min_barrier
                upper = init_instruments[link]

        barrier = uniform((1, ), lower, upper, FLOAT_DTYPE)

        vol = instrument_simulator.volatility[link]
        derivative = derivatives.BarrierCall(
            maturity, strike[idx], barrier, rate, vol, outin[idx], updown[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def random_geometric_asian_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int):

    assert book_size >= instrument_dim, "book_size smaller than instrument_dim."

    init_instruments, init_numeraire, drift, rate, diffusion = \
        random_black_scholes_parameters(
            maturity, instrument_dim, num_brownian_motions, seed)

    instrument_simulator = GBM(rate, drift, diffusion)
    numeraire_simulator = ConstantBankAccount(rate)

    linker = random_linker(book_size, instrument_dim)
    exposure = random_sign(book_size, 3 / 4)
    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)

    for idx, link in enumerate(linker):
        vol = instrument_simulator.volatility[link]
        derivative = derivatives.GeometricAverage(maturity, vol)
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book