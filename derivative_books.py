# -*- coding: utf-8 -*-
import numpy as np
import abc
import math

from utils import norm_cdf
from constants import NP_FLOAT_DTYPE, NP_INT_DTYPE

# ==============================================================================
# === Base
class DerivativeBook(abc.ABC):
    def setup_hedge(self,
                    state: np.ndarray,
                    hedge: np.ndarray,
                    value: float) -> np.ndarray:
        """Initialisation of DerivativeBook
        Args:
            state: state_like
            hedge: (num_samples, market_size)
            value: (num_samples, )
        Returns:
            changes in hedge portfolio (num_samples, market_size + 1, )
        """
        state = self._force_state_shape(state)
        self.hedge = np.zeros_like(state[:, :(self.get_market_size() + 1), -1])
        self.hedge[:, -1] = value / state[:, -1, -1]
        self.rebalance_hedge(state, hedge)

        return self.hedge


    def hedge_value(self, state: np.ndarray) -> np.ndarray:
        """Compute value of hedge portfolio. Assumes DerivativeBook.setup_hedge
        has been run at least once beforehand.
        Args:
            state: state_like
        Returns:
            value: (num_samples, )
        """
        state = self._force_state_shape(state)
        return (self.hedge * state[:, :(self.get_market_size() + 1), -1]).sum(axis=1)


    def rebalance_hedge(
            self, state: np.ndarray, hedge: np.ndarray) -> np.ndarray:
        """Rebalance hedge portfolio. Assumes DerivativeBook.setup_hedge has
        been run at least once beforehand.
        Args:
            state: state_like
            hedge: (num_samples, market_size)
        Returns:
            changes in hedge portfolio (num_samples, market_size + 1)
        """
        state = self._force_state_shape(state)
        risk_change = hedge - self.hedge[:, :-1]
        risk_value_change = (risk_change * state[:, :self.get_market_size(), -1]).sum(axis=1)
        riskless_change = -risk_value_change / state[:, self.get_market_size(), -1]
        change = np.hstack([risk_change, riskless_change[:, np.newaxis]])
        self.hedge += change

        return change


    def get_state_dimension(self) -> int:
        """Returns the dimensionalility of the state."""
        return self.get_market_size() + 1 + self.get_non_market_size()


    @abc.abstractmethod
    def get_book_size(self) -> int:
        """Returns the number of derivatives in the book."""


    @abc.abstractmethod
    def get_market_size(self) -> int:
        """Returns the number of underlying risky tradabke processes."""


    @abc.abstractmethod
    def get_non_market_size(self) -> int:
        """Returns the number of underlying non-tradable processes."""


    def _force_state_shape(self, state: np.ndarray) -> np.ndarray:
        """Reformats input based on dimensionality.
        Args:
            state: state_like
        Returns:
            state: (num_samples, state_dimension, num_steps + 1)
        """
        state = np.array(state, NP_FLOAT_DTYPE, ndmin=1)

        dimension = state.ndim
        if dimension == 1:
            return state[np.newaxis, :, np.newaxis]
        elif dimension == 2:
            return state[:, :, np.newaxis]
        elif dimension == 3:
            return state
        else:
            raise ValueError(f"dimensionality {dimension} > 3.")


    def payoff(self, state: np.ndarray) -> np.ndarray:
        """Wrapper for DerivativeBook.payoff."""
        return self._payoff(self._force_state_shape(state))


    @abc.abstractmethod
    def _payoff(self, state: np.ndarray) -> np.ndarray:
        """Compute payoff from terminal state.
        Args:
            state: (num_samples, state_dimension, num_steps + 1)
        Returns:
            payoff: (num_samples, )
        """


    def book_value(self, state: np.ndarray, time:float) -> np.ndarray:
        """Wrapper for DerivativeBook._book_value."""
        return self._book_value(self._force_state_shape(state), time)


    @abc.abstractmethod
    def _book_value(self, state: np.ndarray, time: float) -> np.ndarray:
        """Compute value of book.
        Args:
            state: (num_samples, state_dimension, num_steps + 1)
            time: float
        Returns:
            value: (num_samples, )
        """


    @abc.abstractmethod
    def sample_paths(self,
                     init_state: np.ndarray,
                     num_paths: int,
                     num_steps: int,
                     risk_neutral: bool) -> np.ndarray:
        """Simulate sample paths of risky assets.
        Args:
            init_state: (state_dimension, )
            num_paths: int
            num_steps: int
            risk_neutral: bool
        Returns:
            sample paths: (num_paths, state_dimension, num_steps + 1)
        """

# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns price in Black's model.
    Args:
        time_to_maturity: float
        spot: (num_samples, book_size)
        strike: (book_size, )
        rate: float
        volatility: (book_size, )
        theta: (book_size, )
    Returns:
        price: (num_samples, book_size)
    """
    if time_to_maturity > 0:
        deflator = math.exp(-rate * time_to_maturity)
        forward = spot / deflator
        m = np.log(forward / strike)
        v = volatility * math.sqrt(time_to_maturity)
        m_over_v = m / v
        v_over_2 = v / 2.

        return deflator * theta \
            * (forward * norm_cdf(theta * (m_over_v + v_over_2)) \
               - strike * norm_cdf(theta * (m_over_v - v_over_2)))
    else:
        diff = theta * (spot - strike)
        return diff * (diff > 0)


def black_delta(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns delta in Black's model.
    Args:
        time_to_maturity: float
        spot: (num_samples, book_size)
        strike: (book_size, )
        rate: float
        volatility: (book_size, )
        theta: (book_size, )
    Returns:
        delta: (num_samples, book_size)
    """
    if time_to_maturity > 0:
        forward = spot * math.exp(rate * time_to_maturity)
        m = np.log(forward / strike)
        v = volatility * math.sqrt(time_to_maturity)

        return theta * norm_cdf(theta * (m / v + v / 2.))
    else:
        return theta * (spot - strike) > 0


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
        Sample paths of a multivariate GBM (num_paths, market_size,
                                            num_steps + 1)
    """
    zero_mean = np.zeros_like(drift)
    rvs = np.random.default_rng().multivariate_normal(
        zero_mean, correlation, size=(num_paths, num_steps))

    dt = maturity / num_steps
    m = (drift - volatility * volatility / 2.) * dt
    v = volatility * math.sqrt(dt)
    rvs = np.exp(m + v * rvs)

    paths = np.zeros((num_paths, len(init_state), num_steps + 1))
    paths[:, :, 0] = init_state

    for idx in range(num_steps):
        paths[:, :, idx + 1] = paths[:, :, idx] * rvs[:, idx]

    return paths


class BlackScholesPutCallBook(DerivativeBook):
    def __init__(self,
                 maturity: float,
                 strike: np.ndarray,
                 drift: np.ndarray,
                 rate: float,
                 diffusion: np.ndarray,
                 put_call: np.ndarray,
                 exposure: np.ndarray,
                 linker: np.ndarray) -> None:
        """ Initialisation of BlackScholesPutCallBook
        Args:
            maturity: float
            strike: (book_size, )
            drift: (market_size, )
            rate: float
            diffusion: (market_size, num_brownian_motions)
            put_call: 1 or -1 for call / put (book_size, )
            exposure: n or -n for long / short (book_size, )
            linker: {0, ..., market_size-1} indicates what asset the
                strike is associated with (book_size, )
        """
        self.maturity = float(maturity)
        self.strike = np.array(strike, NP_FLOAT_DTYPE, ndmin=1)
        self.drift = np.array(drift, NP_FLOAT_DTYPE, ndmin=1)
        self.rate = float(rate)
        self.diffusion = np.array(diffusion, NP_FLOAT_DTYPE, ndmin=2)
        self.put_call = np.array(put_call, NP_INT_DTYPE, ndmin=1)
        self.exposure = np.array(exposure, NP_INT_DTYPE, ndmin=1)
        self.linker = np.array(linker, NP_INT_DTYPE, ndmin=1)

        self.volatility = np.linalg.norm(self.diffusion, axis=1)
        self.correlation = (self.diffusion @ self.diffusion.T) \
            / (self.volatility[:, np.newaxis] @ self.volatility[np.newaxis, :])

        self._book_size = len(self.strike)
        self._market_size = len(self.drift)

    # === abstract base class implementations
    def get_book_size(self):
        return self._book_size


    def get_market_size(self):
        return self._market_size


    def get_non_market_size(self):
        return 0


    def _payoff(self, state: np.ndarray) -> np.ndarray:
        """Implementation of DerivativeBook._payoff."""
        tradable = self._get_tradables(state)
        diff = self.put_call * (tradable[:, self.linker] - self.strike)
        return (diff * (diff > 0)) @ self.exposure


    def _book_value(self, state: np.ndarray, time: float) -> np.ndarray:
        """Implementation of DerivativeBook.book_value."""
        tradable = self._get_tradables(state)
        values = self._marginal_book_value(tradable, time)
        return values.sum(axis=1)


    def sample_paths(self,
                     init_state: np.ndarray,
                     num_paths: int,
                     num_steps: int,
                     risk_neutral: bool) -> np.ndarray:
        """Implementation of sample_paths from DerivativeBook"""
        measure_drift = np.full_like(self.drift, self.rate) if risk_neutral \
            else self.drift

        # simulate risky paths
        tradable = init_state[:self.get_market_size()]
        risk = simulate_geometric_brownian_motion(
            self.maturity,
            tradable,
            measure_drift,
            self.volatility,
            self.correlation,
            num_paths,
            num_steps)

        # simulate riskless path
        time_grid = np.linspace(0., self.maturity, num_steps + 1)
        single_path = init_state[self.get_market_size()] \
            * np.exp(self.rate * time_grid)
        riskless = np.tile(single_path, (num_paths, 1))

        return np.concatenate((risk, riskless[:, np.newaxis, :]), axis=1)


    # === other
    def _get_tradables(self, state: np.ndarray) -> np.ndarray:
        """Extract tradable assets from state.
        Args:
            state: see DerivativeBook._force_state_shape
        Returns:
            tradable: (num_samples, market_size)
        """
        return state[:, :self.get_market_size(), -1]


    def _marginal_book_value(self, tradables: np.ndarray, time: float) -> np.ndarray:
        """Computes value of each individual option.
            Args:
                tradables: see _get_tradables.
            Returns:
                prices: (num_samples, book_size)

        """
        value = black_price(
            self.maturity - time,
            tradables[:, self.linker],
            self.strike,
            self.rate,
            self.volatility[self.linker],
            self.put_call
            )

        return self.exposure * value


    def book_delta(self, state: np.ndarray, time: float) -> np.ndarray:
        """Computes gradient of book wrt. underlying tradables
        Args:
            state_like
        Returns:
            gradient: (num_samples, market_size)
        """
        state = super()._force_state_shape(state)
        tradable = self._get_tradables(state)
        gradient = self._marginal_book_delta(tradable, time)
        return np.apply_along_axis(
            lambda x: np.bincount(self.linker, x), 1, gradient)


    def _marginal_book_delta(self,
                             tradables: np.ndarray, time:float) -> np.ndarray:
        """Computes delta of each individual option.
            Args:
                see _get_tradables
            Returns:
                gradient: (num_paths, book_size)
        """
        gradient = black_delta(
            self.maturity - time,
            tradables[:, self.linker],
            self.strike,
            self.rate,
            self.volatility[self.linker],
            self.put_call
            )

        return gradient * self.exposure