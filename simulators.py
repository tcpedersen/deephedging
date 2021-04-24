# -*- coding: utf-8 -*-
import abc
import tensorflow as tf
import numpy as np

import utils
from constants import FLOAT_DTYPE

class Simulator(abc.ABC):
    @property
    def max_size(self) -> int:
        return int(2**25)

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


    def _split_with_remainder(self, tensor, size):
        return tf.split(tensor, tf.arange)


    def _simulate_batch(self, time, init_state, batch_size, risk_neutral,
                        use_sobol, skip):
        increment = time[1:] - time[:-1]
        timesteps = len(time) - 1
        rvs = self.rvs(batch_size, timesteps, use_sobol=use_sobol, skip=skip)

        paths = [init_state]
        for idx, dt in enumerate(increment):
            v = self.advance(paths[-1], rvs[:, idx, :], dt, risk_neutral)
            paths.append(v)

        return tf.stack(paths, 2)

    def simulate(self, time, init_state, batch_size, risk_neutral,
                 use_sobol=False, skip=0):
        """Simulates paths.
        Args:
            time: (time_steps + 1, )
            init_state: (dimension, ) or (batch_size, dimension)
            batch_size: int
            risk_neutral: bool
        Returns:
            paths: (batch_size, state_dim, timesteps + 1)
        """
        if tf.equal(len(tf.shape(init_state)), 1):
            init_state = tf.tile(init_state[tf.newaxis, :], (batch_size, 1))

        state_dim = tf.shape(init_state)[-1]
        mini_batch_size = min(
            self.max_size // (len(time) * state_dim), batch_size)
        num_splits = int(batch_size // mini_batch_size)
        size_split = [mini_batch_size] * num_splits
        size_split += [batch_size - sum(size_split)]

        init_state_split = tf.split(init_state, size_split)
        skip_values = skip + tf.cumsum(size_split) - mini_batch_size
        paths = []

        for init, skip in zip(init_state_split, skip_values):
            if tf.equal(tf.size(init), 0):
                continue

            size = tf.shape(init)[0]
            path = self._simulate_batch(time, init, size, risk_neutral,
                                        use_sobol=use_sobol, skip=skip)
            paths.append(path)

        return tf.concat(paths, 0)


class GBM(Simulator):
    def __init__(self, rate, drift, diffusion):
        """Initialisation of GBM.
        Args:
            rate: float
            drift: (dimension, )
            diffusion: (dimension, None)
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


class JumpGBM(GBM):
    def __init__(self, rate, drift, diffusion, intensity, jumpsize, jumpvol):
        """Initialisation of GBM.
        Args:
            rate: float
            drift: (dimension, )
            diffusion: (dimension, None)
            intensity: float
            jumpsize: float
            jumpvol: jumpvol
        """
        if len(drift) > 1:
            raise NotImplementedError("multivariate not implemented.")
        super().__init__(rate, drift, diffusion)

        self.intensity = float(intensity)
        self.jumpsize = float(jumpsize)
        self.jumpvol = float(jumpvol)

        m = self.jumpsize + tf.square(self.jumpvol) / 2.
        self.comp = self.intensity * (tf.math.exp(m) - 1)


    def advance(self, state, rvs, dt, risk_neutral):
        # TODO make jump distribution depend on risk_neutral
        if not risk_neutral:
            raise NotImplementedError("does not support P-measure.")

        nonjump = super().advance(state, rvs, dt, risk_neutral)
        poisson = tf.random.poisson(rvs.shape, self.intensity * dt, FLOAT_DTYPE)
        normals = tf.random.normal(rvs.shape, dtype=FLOAT_DTYPE)
        logjumps = poisson * self.jumpsize + tf.sqrt(poisson) \
            * self.jumpvol * normals

        return nonjump * tf.exp(logjumps - self.comp * dt)


class ConstantBankAccount(Simulator):
    def __init__(self, rate):
        self.rate = float(rate)
        self.correlation = tf.constant([[1.]], FLOAT_DTYPE)


    def advance(self, state, rvs, dt, risk_neutral):
        return state + state * self.rate * dt