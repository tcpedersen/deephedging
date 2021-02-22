# -*- coding: utf-8 -*-
import abc
import tensorflow as tf
import tensorflow_probability as tfp

from constants import FLOAT_DTYPE

class Simulator(abc.ABC):
    @property
    @abc.abstractmethod
    def correlation(self) -> tf.Tensor:
        """Returns the correlation matrix."""

    def rvs(self, batch_size, num_steps):
        """Returns samples of normal random variables with correlation matrix.
        Args:
            batch_size: int
            num_steps: int
        Returns:
            rvs: (batch_size, num_steps, state_dim)
        """
        size = (batch_size, num_steps)
        scale_tril = tf.linalg.cholesky(self.correlation)
        normal = tfp.distributions.MultivariateNormalTriL

        return normal(scale_tril=scale_tril).sample(size)

    @abc.abstractmethod
    def advance(self, state, rvs):
        """Advance the state.
        Args:
            state: (batch_size, state_dim)
            rvs: (batch_size, state_dim)
        Returns:
            new_state: (batch_size, state_dim)
        """


    def simulate(self, maturity, init_state, batch_size, num_steps):
        """Simulates paths.
        Args:
            init_state: (state_dim, )
            batch_size: int
            num_steps: int
        Returns:
            paths: (batch_size, state_dim, num_steps + 1)
        """
        dt = maturity / num_steps
        rvs = self.rvs(batch_size, num_steps)
        paths = [tf.tile(init_state[tf.newaxis, :], (batch_size, 1))]

        for idx in range(num_steps):
            paths.append(self.advance(paths[-1], rvs[:, idx, :], dt))

        return paths


class GBM(Simulator):
    def __init__(self, maturity, drift, volatility, correlation):
        self.maturity = float(maturity)
        self.drift = tf.convert_to_tensor(drift, FLOAT_DTYPE)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)
        self._correlation = tf.convert_to_tensor(correlation, FLOAT_DTYPE)


    @property
    def correlation(self) -> tf.Tensor:
        return self._correlation


    def advance(self, state, rvs, dt):
        m = (self.drift - self.volatility * self.volatility / 2.) * dt
        v = self.volatility * dt**(1 / 2)
        rvs = tf.exp(m + v * rvs)

        return state * rvs


class ConstantBankAccount(Simulator):
    def __init__(self, rate):
        self.rate = tf.constant((float(rate), ), FLOAT_DTYPE)


    @property
    def correlation(self) -> tf.Tensor:
        return tf.constant([[1.]], FLOAT_DTYPE)


    def advance(self, state, rvs, dt):
        return state + state * self.rate * dt