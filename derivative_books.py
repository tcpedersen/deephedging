# -*- coding: utf-8 -*-
import numpy as np
import abc

from utils import norm_cdf
from constants import NP_FLOAT_DTYPE, NP_INT_DTYPE

# ==============================================================================
# === Base
class DerivativeBook(abc.ABC):
    @abc.abstractmethod
    def get_book_size(self) -> int:
        """Returns the number of derivatives in the book."""

    @abc.abstractmethod
    def get_market_size(self) -> int:
        """Returns the number of underlying risky assets in the book."""

    @abc.abstractmethod
    def get_state_dimension(self) -> int:
        """Returns the dimension of the underlying risky assets. If they have
        different dimensions, return the largest."""

    @abc.abstractmethod
    def payoff(self, terminal_state: np.ndarray) -> np.ndarray:
        """Compute payoff from terminal state.
        Args:
            terminal_state: (num_paths, market_size, state_dimension)
        Returns:
            payoff: (num_paths, )
        """

    @abc.abstractmethod
    def book_value(self, state: np.ndarray, time: float) -> np.ndarray:
        """Compute value of book for m different states.
        Args:
            state: (num_paths, market_size, state_dimension)
            time: float
        Returns:
            value: (num_paths, )
        """

    @abc.abstractmethod
    def sample_risky_assets(self,
                            init_state: np.ndarray,
                            num_paths: int,
                            num_steps: int,
                            risk_neutral: bool) -> np.ndarray:
        """Simulate sample paths of risky assets.
        Args:
            init_state: (market_size, state_dimension)
            num_paths: int
            num_steps: int
            risk_neutral: bool
        Returns:
            sample paths: (num_paths, market_size, state_dimension,
                           num_steps + 1)
        """

    @abc.abstractmethod
    def sample_riskfree_asset(self,
                              init_state: np.array,
                              num_paths: int,
                              num_steps: int) -> np.ndarray:
        """Simulate sample paths of riskless asset.
        Args:
            init_state: (1, )
            num_paths: int
            num_steps: int
        Returns:
            sample paths: (num_paths, num_steps + 1)
        """

# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns price in Black's model.
    Args:
        time_to_maturity: float
        spot: (num_paths, book_size)
        strike: (book_size, )
        rate: float
        volatility: (book_size, )
        theta: (book_size, )
    Returns:
        price: (num_paths, book_size)
    """
    forward = spot * np.exp(rate * time_to_maturity)
    m = np.log(forward / strike)
    v = (volatility * np.sqrt(time_to_maturity))
    delta = np.exp(-rate * time_to_maturity)

    return delta * theta \
        * (forward * norm_cdf(theta * (m / v + v / 2.)) \
           - strike * norm_cdf(theta * (m / v - v / 2.)))


def black_delta(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns delta in Black's model.
    Args:
        time_to_maturity: float
        spot: (num_pats, book_size)
        strike: (book_size, )
        rate: float
        volatility: (book_size, )
        theta: (book_size, )
    Returns:
        delta: (num_paths, book_size)
    """
    forward = spot * np.exp(rate * time_to_maturity)
    m = np.log(forward / strike)
    v = (volatility * np.sqrt(time_to_maturity))

    return theta * norm_cdf(theta * (m / v + v / 2.))


def simulate_geometric_brownian_motion(maturity: float,
                                       init_state: np.ndarray,
                                       drift: np.ndarray,
                                       volatility: np.ndarray,
                                       correlation: np.ndarray,
                                       num_paths: int,
                                       num_steps: int) -> np.ndarray:
    """Simulate a multivariate GBM.
    Args:
        maturity: float
        init_state: (market_size, )
        drift : (market_size, )
        volatility: (market_size, )
        correlation : (market_size, market_size)
        num_paths : int
        num_steps : int
    Returns:
        Sample paths of a multivariate GBM (num_paths, market_size, num_steps + 1)
    """
    dt = maturity / num_steps

    mean = np.zeros_like(init_state)
    rvs = np.random.default_rng().multivariate_normal(
        mean, correlation, size=(num_paths, num_steps))

    paths = np.zeros((num_paths, len(init_state), num_steps + 1))
    paths[:, :, 0] = init_state

    for idx in range(num_steps):
        paths[:, :, idx + 1] = paths[:, :, idx] \
            * np.exp((drift - volatility**2 / 2) * dt \
                     + volatility * np.sqrt(dt) * rvs[:, idx, :])

    return paths


class BlackScholesPutCallBook(DerivativeBook):
    def __init__(self,
                 maturity: float,
                 strike: np.ndarray,
                 drift: np.ndarray,
                 rate: float,
                 diffusion: np.ndarray,
                 put_call: np.ndarray,
                 linker: np.ndarray) -> None:
        """ Initialisation of BlackScholesPutCallBook
        Args:
            maturity: float
            strike: (book_size, )
            drift: (market_size, )
            rate: float
            diffusion: (market_size, num_brownian_motions)
            put_call: 1 or -1 for call / put (book_size, )
            linker: {0, ..., market_size-1} indicates what asset the
                strike is associated with (book_size, )
        """
        self.maturity = float(maturity)
        self.strike = np.array(strike, NP_FLOAT_DTYPE, ndmin=1)
        self.drift = np.array(drift, NP_FLOAT_DTYPE, ndmin=1)
        self.rate = float(rate)
        self.diffusion = np.array(diffusion, NP_FLOAT_DTYPE, ndmin=2)
        self.put_call = np.array(put_call, NP_INT_DTYPE, ndmin=1)
        self.linker = np.array(linker, NP_INT_DTYPE, ndmin=1)

        self.volatility = np.linalg.norm(self.diffusion, axis=1)
        self.correlation = (self.diffusion @ self.diffusion.T) \
            / (self.volatility[:, np.newaxis] @ self.volatility[np.newaxis, :])

    def get_book_size(self):
        return len(self.strike)

    def get_market_size(self):
        return len(self.drift)

    def get_state_dimension(self):
        return 1

    def payoff(self, terminal_state: np.ndarray) -> np.ndarray:
        """Implementation of payoff from DerivativeBook"""
        diff = self.put_call * (terminal_state[:, self.linker, 0] - self.strike)
        return (diff * (diff > 0)).sum(axis=1)

    def marginal_book_value(self, state: np.ndarray, time:float) -> np.ndarray:
        """Computes value of each individual option.
            Args:
                same as DerivativeBook.book_value
            Returns:
                prices: (num_paths, book_size)

        """
        value = black_price(
            self.maturity - time,
            state[:, self.linker, 0],
            self.strike,
            self.rate,
            self.volatility[self.linker],
            self.put_call
            )

        return value


    def book_value(self, state: np.ndarray, time: float) -> np.ndarray:
        """Implementation of book_value from DerivativeBook."""
        return self.marginal_book_value(state, time).sum(axis=1)


    def marginal_book_delta(self, state: np.ndarray, time:float) -> np.ndarray:
        """Computes delta of each individual option.
            Args:
                same as DerivativeBook.book_value
            Returns:
                gradient: (num_paths, book_size)
        """
        gradient = black_delta(
            self.maturity - time,
            state[:, self.linker, 0],
            self.strike,
            self.rate,
            self.volatility[self.linker],
            self.put_call
            )

        return gradient


    def sample_risky_assets(self,
                            init_state: np.ndarray,
                            num_paths: int,
                            num_steps: int,
                            risk_neutral: bool) -> np.ndarray:
        """Implementation of sample_risky_assets from DerivativeBook"""
        measure_drift = np.full_like(self.drift, self.rate) if risk_neutral \
            else self.drift
        paths = simulate_geometric_brownian_motion(
            self.maturity,
            init_state[:, 0],
            measure_drift,
            self.volatility,
            self.correlation,
            num_paths,
            num_steps)

        return paths[:, :, np.newaxis, :]

    def sample_riskfree_asset(self,
                              init_state: np.array,
                              num_paths: int,
                              num_steps: int) -> np.ndarray:
        """Implementation of sample_riskfree_asset from DerivativesBook"""
        time_grid = np.linspace(0., self.maturity, num_steps + 1)
        single_path = init_state * np.exp(self.rate * time_grid)

        return np.tile(single_path, (num_paths, 1))