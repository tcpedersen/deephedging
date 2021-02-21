# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import abc
import math

from tensorflow.random import uniform

from utils import norm_cdf
from constants import FLOAT_DTYPE, INT_DTYPE

# ==============================================================================
# === Base
class DerivativeBook(abc.ABC):
    @property
    @abc.abstractmethod
    def book_size(self) -> int:
        """Number of derivatives in the book."""


    @property
    @abc.abstractmethod
    def instrument_dim(self) -> int:
        """Number of underlying risky instruments processes."""


    @property
    @abc.abstractmethod
    def non_instrument_dim(self) -> int:
        """Number of underlying non-instruments processes."""


    @property
    def state_dim(self) -> int:
        """Returns the dimensionalility of the state."""
        return self.instrument_dim + 1 + self.non_instrument_dim


    def _force_state_shape(self, state: tf.Tensor) -> tf.Tensor:
        """Reformats input based on dimensionality.
        Args:
            state: state_like
        Returns:
            state: (num_samples, state_dim, num_steps + 1)
        """
        state = tf.convert_to_tensor(state, FLOAT_DTYPE)

        dimension = len(state.shape)
        if dimension == 1:
            return state[tf.newaxis, :, tf.newaxis]
        elif dimension == 2:
            return state[:, :, tf.newaxis]
        elif dimension == 3:
            return state
        else:
            raise ValueError(f"dimensionality {dimension} > 3.")


    def payoff(self, state: tf.Tensor) -> tf.Tensor:
        """Wrapper for DerivativeBook.payoff."""
        return self._payoff(self._force_state_shape(state))


    @abc.abstractmethod
    def _payoff(self, state: tf.Tensor) -> tf.Tensor:
        """Compute payoff from terminal state.
        Args:
            state: (num_samples, state_dim, num_steps + 1)
        Returns:
            payoff: (num_samples, )
        """


    def book_value(self, state: tf.Tensor, time:float) -> tf.Tensor:
        """Wrapper for DerivativeBook._book_value."""
        return self._book_value(self._force_state_shape(state), time)


    @abc.abstractmethod
    def _book_value(self, state: tf.Tensor, time: tf.Tensor) -> tf.Tensor:
        """Compute value of book at each point in time.
        Args:
            state: (num_samples, state_dim, num_steps + 1)
            time: (num_steps + 1, )
        Returns:
            value: (num_samples, num_steps + 1)
        """


    @abc.abstractmethod
    def sample_paths(self,
                     init_state: tf.Tensor,
                     num_paths: int,
                     num_steps: int,
                     risk_neutral: bool) -> tf.Tensor:
        """Simulate sample paths of risky assets.
        Args:
            init_state: (state_dim, )
            num_paths: int
            num_steps: int
            risk_neutral: bool
        Returns:
            time: (num_steps + 1, )
            sample paths: (num_samples, state_dim, num_steps + 1)
        """


    def gradient_paths(self, init_state, num_paths, num_steps):
        """Simulates paths, gradients and payoffs.
        Args:
            init_state: (state_dim, )
            num_paths: int
            num_steps: int
        Returns:
            time: (num_steps + 1, )
            paths: (num_samples, state_dim, num_steps + 1)
            grads: (num_samples, num_steps + 1)
            payoff: (num_samples, )
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(init_state)
            time, paths = self.sample_paths(init_state, num_paths, num_steps, True)
            payoff = self.payoff(paths)
            price = tf.reduce_mean(payoff)
        grads = tape.batch_jacobian(price, paths)

        return time, paths, grads, payoff


# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns price in Black's model.
    Args:
        time_to_maturity: (num_steps + 1, )
        spot: (num_samples, book_size, num_steps + 1)
        strike: (book_size, )
        rate: float
        volatility: (book_size, )
        theta: (book_size, )
    Returns:
        price: (num_samples, book_size, num_steps + 1)
    """
    _strike = strike[tf.newaxis, :, tf.newaxis]
    _theta = theta[tf.newaxis, :, tf.newaxis]

    deflator = tf.math.exp(-rate * time_to_maturity)
    forward = spot / deflator
    m = tf.math.log(forward / _strike)
    v = volatility[tf.newaxis, :, tf.newaxis] * tf.math.sqrt(time_to_maturity)
    m_over_v = m / v
    v_over_2 = v / 2.

    value = deflator * _theta \
            * (forward * norm_cdf(_theta * (m_over_v + v_over_2)) \
               - _strike * norm_cdf(_theta * (m_over_v - v_over_2)))

    return value


def black_delta(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns delta in Black's model.
    Args:
        see black_price
    Returns:
        delta: (num_samples, book_size, num_steps + 1)
    """
    _strike = strike[tf.newaxis, :, tf.newaxis]
    _theta = theta[tf.newaxis, :, tf.newaxis]

    deflator = tf.math.exp(-rate * time_to_maturity)
    forward = spot / deflator
    m = tf.math.log(forward / _strike)
    v = volatility[tf.newaxis, :, tf.newaxis] * tf.math.sqrt(time_to_maturity)

    return _theta * norm_cdf(_theta * (m / v + v / 2.))


def simulate_geometric_brownian_motion(maturity: float,
                                       init_state: tf.Tensor,
                                       drift: tf.Tensor,
                                       volatility: tf.Tensor,
                                       correlation: tf.Tensor,
                                       num_paths: int,
                                       num_steps: int) -> tf.Tensor:
    """Simulate a multivariate GBM.
    Args:
        maturity: float
        init_state: (instrument_dim, )
        drift : (instrument_dim, )
        volatility: (instrument_dim, )
        correlation : (instrument_dim, instrument_dim)
        num_paths : int
        num_steps : int
    Returns:
        Sample paths: (num_paths, instrument_dim, num_steps + 1)
    """
    zero_mean = tf.zeros_like(drift)
    size = (num_paths, num_steps)
    rvs = np.random.multivariate_normal(zero_mean, correlation, size)
    rvs = tf.convert_to_tensor(rvs, FLOAT_DTYPE)

    dt = maturity / num_steps
    m = (drift - volatility * volatility / 2.) * dt
    v = volatility * math.sqrt(dt)
    rvs = tf.exp(m + v * rvs)

    # paths = tf.zeros((num_paths, len(init_state), num_steps + 1))
    # paths[:, :, 0] = init_state

    state = tf.tile(init_state[tf.newaxis, :], (num_paths, 1))
    paths = [state]

    for idx in range(num_steps):
        state = state * rvs[:, idx]
        paths.append(state)

    return tf.stack(paths, axis=-1)


class BlackScholesPutCallBook(DerivativeBook):
    def __init__(self,
                 maturity: float,
                 strike: tf.Tensor,
                 drift: tf.Tensor,
                 rate: float,
                 diffusion: tf.Tensor,
                 put_call: tf.Tensor,
                 exposure: tf.Tensor,
                 linker: tf.Tensor) -> None:
        """ Initialisation of BlackScholesPutCallBook
        Args:
            maturity: float
            strike: (book_size, )
            drift: (instrument_dim, )
            rate: float
            diffusion: (instrument_dim, num_brownian_motions)
            put_call: 1 or -1 for call / put (book_size, )
            exposure: n or -n for long / short (book_size, )
            linker: {0, ..., instrument_dim-1} indicates what asset the
                strike is associated with (book_size, )
        """
        self.maturity = float(maturity)
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.drift = tf.convert_to_tensor(drift, FLOAT_DTYPE)
        self.rate = float(rate)
        self.diffusion = tf.convert_to_tensor(diffusion, FLOAT_DTYPE)
        self.put_call = tf.convert_to_tensor(put_call, FLOAT_DTYPE)
        self.exposure = tf.convert_to_tensor(exposure, FLOAT_DTYPE)
        self.linker = tf.convert_to_tensor(linker, INT_DTYPE)

        self.volatility = tf.linalg.norm(self.diffusion, axis=1)
        self.correlation = (self.diffusion @ tf.transpose(self.diffusion)) \
            / (self.volatility[:, tf.newaxis] @ self.volatility[tf.newaxis, :])

    # === abstract base class implementations
    @property
    def book_size(self):
        return len(self.strike)


    @property
    def instrument_dim(self):
        return len(self.drift)


    @property
    def non_instrument_dim(self):
        return 0


    def _payoff(self, state: tf.Tensor) -> tf.Tensor:
        """Implementation of DerivativeBook._payoff."""
        instruments = self._get_instruments(state)[..., -1]
        diff = self.put_call * (tf.gather(instruments, self.linker, axis=1) - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)
        return tf.squeeze((diff * itm) @ self.exposure[:, tf.newaxis])


    def _book_value(self, state: tf.Tensor, time: tf.Tensor) -> tf.Tensor:
        """Implementation of DerivativeBook.book_value."""
        instruments = self._get_instruments(state)
        values = self._marginal_book_value(instruments, time)
        return tf.reduce_sum(values, axis=1)


    def sample_paths(self,
                     init_state: tf.Tensor,
                     num_paths: int,
                     num_steps: int,
                     risk_neutral: bool) -> tf.Tensor:
        """Implementation of sample_paths from DerivativeBook"""
        measure_drift = tf.ones_like(self.drift) * self.rate if risk_neutral \
            else self.drift

        # simulate risky paths
        instruments = init_state[:self.instrument_dim]
        risk = simulate_geometric_brownian_motion(
            self.maturity,
            instruments,
            measure_drift,
            self.volatility,
            self.correlation,
            num_paths,
            num_steps)


        # simulate riskless path
        time_grid = tf.linspace(0., self.maturity, num_steps + 1)
        single_path = init_state[self.instrument_dim] \
            * tf.exp(self.rate * time_grid)
        riskless = tf.tile(single_path[tf.newaxis, :], (num_paths, 1))

        return time_grid, tf.concat((risk, riskless[:, tf.newaxis, :]), axis=1)


    # === other
    def _get_instruments(self, state: tf.Tensor) -> tf.Tensor:
        """Extract instruments assets from state.
        Args:
            state: see DerivativeBook._force_state_shape
        Returns:
            instruments: (num_samples, instrument_dim, num_steps + 1)
        """
        return state[:, :self.instrument_dim, :]


    def _marginal_book_value(
            self, instruments: tf.Tensor, time: tf.Tensor) -> tf.Tensor:
        """Computes value of each individual option.
            Args:
                instruments: see _get_instruments.
                time: (num_steps + 1, )
            Returns:
                prices: (num_samples, book_size, num_steps + 1)

        """
        value = black_price(
            self.maturity - time,
            tf.gather(instruments, self.linker, axis=1),
            self.strike,
            self.rate,
            tf.gather(self.volatility, self.linker),
            self.put_call
            )

        return self.exposure[tf.newaxis, :, tf.newaxis] * value


    def book_delta(self, state: tf.Tensor, time: float) -> tf.Tensor:
        """Computes gradient of book wrt. underlying instruments
        Args:
            see DerivativeBook._book_value
        Returns:
            gradient: (num_samples, instrument_dim, num_steps + 1)
        """
        state = super()._force_state_shape(state)
        instruments = self._get_instruments(state)
        gradient = self._marginal_book_delta(instruments, time)

        v = []
        for k in range(self.instrument_dim):
            mask = tf.where(self.linker == k)[:, 0]
            v.append(tf.reduce_sum(
                tf.gather(gradient, mask, axis=1), axis=1, keepdims=True))

        return tf.concat(v, axis=1)


    def _marginal_book_delta(
            self, instruments: tf.Tensor, time: tf.Tensor) -> tf.Tensor:
        """Computes delta of each individual option.
            Args:
                see _marginal_book_value
            Returns:
                gradient: (num_samples, book_size, num_steps + 1)
        """
        gradient = black_delta(
            self.maturity - time,
            tf.gather(instruments, self.linker, axis=1),
            self.strike,
            self.rate,
            tf.gather(self.volatility, self.linker),
            self.put_call
            )

        return self.exposure[tf.newaxis, :, tf.newaxis] * gradient


def random_black_scholes_put_call_book(
        maturity: float,
        book_size: int,
        instrument_dim: int,
        num_brownian_motions: int,
        seed: int):
    tf.random.set_seed(seed)

    maturity = float(maturity)
    strike = uniform((book_size, ), 75, 125, FLOAT_DTYPE)
    drift = tf.random.uniform((instrument_dim, ), 0.05, 0.1)
    rate = tf.random.uniform((1, ), 0.0, 0.05)
    diffusion = tf.random.uniform(
        (instrument_dim, num_brownian_motions), 0, 0.25 / num_brownian_motions, FLOAT_DTYPE)
    put_call = 2 * tfp.distributions.Binomial(1, probs=0.5).sample(book_size) - 1
    exposure = 2 * tfp.distributions.Binomial(1, probs=0.5).sample(book_size) - 1

    if instrument_dim  > 1:
        linker = tf.random.uniform((book_size, ), 0, instrument_dim - 1, INT_DTYPE)
    else:
        linker = tf.convert_to_tensor((0, ), INT_DTYPE)

    book = BlackScholesPutCallBook(
        maturity, strike, drift, rate, diffusion, put_call, exposure, linker)

    init_risky = uniform((instrument_dim, ), 75, 125, FLOAT_DTYPE)
    init_riskless = uniform((1, ), 0.75, 1.25, FLOAT_DTYPE)
    init_state = tf.concat([init_risky, init_riskless], axis=0)

    return init_state, book


def random_simple_put_call_book(maturity):
    strike = init_risky = tf.convert_to_tensor([1], FLOAT_DTYPE)
    drift = tf.convert_to_tensor([0.0], FLOAT_DTYPE)
    rate = 0.0
    diffusion = tf.convert_to_tensor([[0.2]], FLOAT_DTYPE)
    put_call = tf.convert_to_tensor([1], FLOAT_DTYPE)
    exposure = tf.convert_to_tensor([1], FLOAT_DTYPE)
    linker = tf.convert_to_tensor([0], INT_DTYPE)

    book = BlackScholesPutCallBook(
        maturity, strike, drift, rate, diffusion, put_call, exposure, linker)

    init_riskless = tf.convert_to_tensor([1], FLOAT_DTYPE)
    init_state = tf.concat([init_risky, init_riskless], axis=0)

    return init_state, book