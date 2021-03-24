# -*- coding: utf-8 -*-
import abc
import tensorflow as tf
import numpy as np

import utils
from constants import FLOAT_DTYPE

class Simulator(abc.ABC):
    @property
    def correlation(self) -> tf.Tensor:
        """Returns the correlation matrix."""
        return self._correlation


    @correlation.setter
    def correlation(self, val):
        self._correlation = val


    @property
    def dimension(self) -> int:
        return len(self.correlation)


    def rvs(self, batch_size, timesteps, use_sobol=False, skip=0):
        """Returns samples of normal random variables with correlation matrix.
        Args:
            batch_size: int
            timesteps: int
            use_sobol: bool
            skip: int
        Returns:
            rvs: (batch_size, timesteps, dimension)
        """
        if not use_sobol and skip > 0:
            raise ValueError("skip must be 0 when use_sobol is false.")

        if use_sobol:
            chol = tf.linalg.cholesky(self.correlation)
            sobol = tf.math.sobol_sample(
                dim=self.dimension * timesteps,
                num_results=batch_size,
                skip=skip,
                dtype=FLOAT_DTYPE)
            sobol = tf.stack(tf.split(sobol, self.dimension, 1), -1)
            rvs = utils.norm_qdf(sobol)
            rvs = tf.matmul(chol, rvs, transpose_b=True)
            rvs = tf.transpose(rvs, [0, 2, 1])
        else:
            size = (batch_size, timesteps)
            loc = tf.zeros((self.dimension, ), FLOAT_DTYPE)
            rvs = np.random.multivariate_normal(loc, self.correlation, size)

        return tf.convert_to_tensor(rvs, FLOAT_DTYPE)


    @abc.abstractmethod
    def advance(self, state, rvs, dt, risk_neutral):
        """Advance the state.
        Args:
            state: (batch_size, state_dim)
            rvs: (batch_size, state_dim)
            risk_neutral: bool
        Returns:
            new_state: (batch_size, state_dim)
        """


    def simulate(self, time, init_state, batch_size, risk_neutral,
                 as_list=False, use_sobol=False, skip=0):
        """Simulates paths.
        Args:
            time: (time_steps + 1, )
            init_state: (dimension, ) or (batch_size, dimension)
            batch_size: int
            risk_neutral: bool
        Returns:
            paths: (batch_size, state_dim, timesteps + 1)
        """
        increment = time[1:] - time[:-1]
        rvs = self.rvs(batch_size, len(time) - 1,
                       use_sobol=use_sobol, skip=skip)

        if tf.equal(len(tf.shape(init_state)), 1):
            paths = [tf.tile(init_state[tf.newaxis, :], (batch_size, 1))]
        elif tf.equal(len(tf.shape(init_state)), 2):
            paths = [init_state]
        else:
            raise ValueError("dimension of init_state > 2.")

        for idx, dt in enumerate(increment):
            paths.append(
                self.advance(paths[-1], rvs[:, idx, :], dt, risk_neutral))

        return paths if as_list else tf.stack(paths, 2)


class GBM(Simulator):
    def __init__(self, rate, drift, diffusion):
        """Initialisation of GBM.
        Args:
            rate: float
            drift: (dimension, )
            diffusion: (dimension, None)
        Returns:
            None
        """
        self.rate = float(rate)
        self.drift = tf.convert_to_tensor(drift, FLOAT_DTYPE)
        self.diffusion = tf.convert_to_tensor(diffusion, FLOAT_DTYPE)

        self.volatility = tf.linalg.norm(self.diffusion, axis=1)
        self.correlation = (self.diffusion @ tf.transpose(self.diffusion)) \
            / (self.volatility[:, tf.newaxis] @ self.volatility[tf.newaxis, :])
        self.correlation = utils.near_positive_definite(self.correlation)


    def advance(self, state, rvs, dt, risk_neutral):
        drift = self.rate if risk_neutral else self.drift

        m = (drift - self.volatility * self.volatility / 2.) * dt
        v = self.volatility * dt**(1 / 2)
        rvs = tf.exp(m + v * rvs)

        return state * rvs


class ConstantBankAccount(Simulator):
    def __init__(self, rate):
        self.rate = float(rate)
        self.correlation = tf.constant([[1.]], FLOAT_DTYPE)


    def advance(self, state, rvs, dt, risk_neutral):
        return state + state * self.rate * dt