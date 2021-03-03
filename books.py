# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import abc

from tensorflow.random import uniform

from derivatives import Derivative, PutCall, BarrierCall
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


    def add_derivative(self, derivative: Derivative, link: int, exposure: float):
        assert issubclass(type(derivative), Derivative)
        assert 0 <= int(link) < self.instrument_dim

        entry = {
            "derivative": derivative,
            "link": int(link),
            "exposure": float(exposure)
            }

        self.derivatives.append(entry)


    def payoff(self, instruments: tf.Tensor, numeraire: tf.Tensor):
        """Computes payoff of book in terms of numeraire for each sample
        in batch
        Args:
            instruments: (batch_size, instrument_dim, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            payoff: (batch_size, )
        """
        payoff = tf.zeros_like(instruments[:, 0, 0], FLOAT_DTYPE)
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            marginal = entry["derivative"].payoff(linked, numeraire)
            payoff += marginal * entry["exposure"]

        return payoff


    def value(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes value of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (time_steps + 1, )
            instrument: (batch_size, instrument_dim, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            value: (batch_size, time_steps + 1)
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
            time: (time_steps + 1, )
            instruments: (batch_size, instrument_dim, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            value: (batch_size, instrument_dim, time_steps + 1)
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


    @abc.abstractmethod
    def sample_paths(self,
                     init_instruments: tf.Tensor,
                     init_numeraire: tf.Tensor,
                     batch_size: int,
                     time_steps: int,
                     risk_neutral: bool) -> tf.Tensor:
        """Simulate sample paths.
        Args:
            init_instruments: (state_dim, )
            init_numeraire: (1, )
            batch_size: int
            time_steps: int
            risk_neutral: bool
        Returns:
            time: (time_steps + 1, )
            instruments: (batch_size, instrument_dim, time_steps + 1)
            numeraire: (time_steps + 1, )
        """
        time = tf.cast(tf.linspace(0., self.maturity, time_steps + 1), FLOAT_DTYPE)
        instruments = self.instrument_simulator.simulate(
            time, init_instruments, batch_size, risk_neutral)
        numeraire = self.numeraire_simulator.simulate(
            time, init_numeraire, 1, risk_neutral)

        return time, instruments, tf.squeeze(numeraire)


# =============================================================================
# === random generation of books
def random_black_scholes_parameters(
        maturity: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int):
    tf.random.set_seed(seed)

    maturity = float(maturity)
    drift = tf.random.uniform((instrument_dim, ), 0.05, 0.1, dtype=FLOAT_DTYPE)
    rate = tf.random.uniform((1, ), 0.0, 0.05, dtype=FLOAT_DTYPE)

    scale = tf.cast(tf.sqrt(float(num_brownian_motions)), FLOAT_DTYPE)
    diffusion = tf.random.uniform(
        (instrument_dim, num_brownian_motions),
        0.1 / scale, 0.3 / scale, FLOAT_DTYPE)

    init_instruments = uniform((instrument_dim, ), 95, 105, FLOAT_DTYPE)
    init_numeraire = uniform((1, ), 0.75, 1.25, FLOAT_DTYPE)

    return init_instruments, init_numeraire, drift, rate, diffusion


def random_put_call_book(
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
    put_call = tf.cast(2 * tfp.distributions.Binomial(1, probs=0.5).sample(
        book_size) - 1, FLOAT_DTYPE)
    exposure = tf.cast(2 * tfp.distributions.Binomial(1, probs=0.5).sample(
        book_size) - 1, FLOAT_DTYPE)

    if instrument_dim  > 1:
        one_of_each = tf.range(instrument_dim)
        other_random = tf.random.uniform((book_size - instrument_dim, ),
                                         0, instrument_dim - 1, INT_DTYPE)
        linker = tf.random.shuffle(tf.concat([one_of_each, other_random], 0))
    else:
        linker = tf.convert_to_tensor((0, ), INT_DTYPE)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
    for idx, link in enumerate(linker):
        vol = instrument_simulator.volatility[link]
        derivative = PutCall(maturity, strike[idx], rate, vol, put_call[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


def simple_put_call_book(maturity, spot, strike, rate, drift, sigma, theta):
    init_instruments = tf.constant((spot, ), FLOAT_DTYPE)
    init_numeraire = tf.constant((1., ), FLOAT_DTYPE)

    instrument_simulator = GBM(rate, drift, [[sigma]])
    numeraire_simulator = ConstantBankAccount(rate)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
    derivative = PutCall(maturity, strike, rate, sigma, theta)
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
    exposure = tf.cast(2 * tfp.distributions.Binomial(1, probs=0.5).sample(
        book_size) - 1, FLOAT_DTYPE)

    outin = tf.cast(2 * tfp.distributions.Binomial(1, probs=0.5).sample(
        book_size) - 1, FLOAT_DTYPE)
    updown = tf.cast(2 * tfp.distributions.Binomial(1, probs=0.5).sample(
        book_size) - 1, FLOAT_DTYPE)

    if instrument_dim  > 1:
        one_of_each = tf.range(instrument_dim)
        other_random = tf.random.uniform((book_size - instrument_dim, ),
                                         0, instrument_dim - 1, INT_DTYPE)
        linker = tf.random.shuffle(tf.concat([one_of_each, other_random], 0))
    else:
        linker = tf.convert_to_tensor((0, ), INT_DTYPE)

    book = DerivativeBook(maturity, instrument_simulator, numeraire_simulator)
    for idx, link in enumerate(linker):
        if link == 15:
            break

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
        derivative = BarrierCall(
            maturity, strike[idx], barrier, rate, vol, outin[idx], updown[idx])
        book.add_derivative(derivative, link, exposure[idx])

    return init_instruments, init_numeraire, book


# linked = instruments[:, link, :]
# marginal = derivative.value(time, linked, numeraire)
