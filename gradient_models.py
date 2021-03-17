# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

import approximators
from constants import FLOAT_DTYPE

class SemiRecurrentApproximator(tf.keras.models.Model, abc.ABC):
    @property
    @abc.abstractmethod
    def approximators(self) -> list:
        """Returns the approximator layers."""


    @abc.abstractmethod
    def observation(self, step, features, internal) -> tf.Tensor:
        """Returns the input to the approximators."""


    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.internal_dim = self.approximators[0].internal_dim


    def inner_call(self, inputs, training=False):
        """The martingales and payoff are assumed to be expressed in terms of
        the numeraire.

        Args:
            inputs: instruments (batch_size, input_dim, timesteps + 1)
        Returns:
            output: (batch_size, timesteps)
        """
        output = []
        batch_size = tf.shape(inputs)[0]
        internal = tf.zeros((batch_size, self.internal_dim), FLOAT_DTYPE)

        for step, h in enumerate(self.approximators):
            observation = self.observation(step, inputs, internal)
            approx, internal = h(observation)
            output.append(approx)

        return output


    def call(self, inputs, training=False):
        """Implementation of tf.keras.Model.call."""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            y = self.inner_call(inputs, training)
        dydx = tape.gradient(y, inputs)

        return tf.concat(y, -1), dydx


class MemorylessSemiRecurrent(SemiRecurrentApproximator):
    def __init__(self, timesteps, instrument_dim, num_layers, num_units):
        super().__init__()

        self._approximators = []
        for _ in range(timesteps):
            self._approximators.append(approximators.DenseApproximator(
                instrument_dim=instrument_dim,
                num_layers=num_layers,
                num_units=num_units,
                internal_dim=0,
                activation=tf.keras.activations.softplus))


    @property
    def approximators(self) -> list:
        return self._approximators


    def observation(self, step, features, internal):
        return features[..., step]


class MemorySemiRecurrent(SemiRecurrentApproximator):
    def __init__(self,
                 timesteps,
                 internal_dim,
                 num_layers,
                 num_units):
        super().__init__()

        self._approximators = []
        for _ in range(timesteps):
            self._approximators.append(approximators.DenseApproximator(
                num_layers=num_layers,
                num_units=num_units,
                output_dim=1,
                internal_dim=internal_dim,
                activation=tf.keras.activations.softplus))


    @property
    def approximators(self) -> list:
        return self._approximators


    def observation(self, step, features, internal):
        return tf.concat([features[..., step], internal], 1)
