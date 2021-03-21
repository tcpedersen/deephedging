# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense, BatchNormalization

import approximators
from constants import FLOAT_DTYPE

class MemoryDifferentialNetwork(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, internal_dim):
        super().__init__()
        self.internal_dim = int(internal_dim)
        self.output_dim = 1

        self.activation = tf.keras.activations.softplus

        self.input_layer = BatchNormalization()
        self.dense_layers = []
        self.batch_layers = []

        for _ in range(num_layers - 1):
            self.dense_layers.append(
                Dense(units=num_units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=self.output_dim + internal_dim)

    def call(self, inputs, training=False):
        feature, internal = inputs

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(feature)
            x = tf.concat([feature, internal], -1)
            x = self.input_layer(x, training=training)
            for dense, batch in zip(self.dense_layers, self.batch_layers):
                x = dense(x)
                x = batch(x, training=training)
                x = self.activation(x)
            output, internal = tf.split(
                self.output_layer(x), [self.output_dim, self.internal_dim], 1)
        gradient = tape.gradient(output, feature)

        return [output, internal, gradient]


class SemiRecurrentDifferentialNetworkTest(tf.keras.models.Model):
    def __init__(self, timesteps, num_layers, num_units, internal_dim):
        super().__init__()
        self.approximators = []
        for _ in range(timesteps):
            self.approximators.append(
                MemoryDifferentialNetwork(
                    num_layers=num_layers,
                    num_units=num_units,
                    internal_dim=internal_dim
                    ))

        self.internal_dim = int(internal_dim)


    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        internal = tf.zeros((batch_size, self.internal_dim), FLOAT_DTYPE)
        output_func = []
        output_grad = []

        for step, h in enumerate(self.approximators):
            observation = [inputs[..., step], internal]
            output, internal, gradient = h(observation)
            output_func.append(output)
            output_grad.append(gradient)

        return tf.concat(output_func, -1), tf.stack(output_grad, -1)


class SemiRecurrentDifferentialNetwork(tf.keras.models.Model, abc.ABC):
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


class MemorylessSRDN(SemiRecurrentDifferentialNetwork):
    def __init__(self, timesteps, num_layers, num_units):
        super().__init__()

        self._approximators = []
        for _ in range(timesteps):
            self._approximators.append(approximators.DenseApproximator(
                num_layers=num_layers,
                num_units=num_units,
                output_dim=1,
                internal_dim=0,
                activation=tf.keras.activations.softplus))


    @property
    def approximators(self) -> list:
        return self._approximators


    def observation(self, step, features, internal):
        return features[..., step]


class MemorySRDN(SemiRecurrentDifferentialNetwork):
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
