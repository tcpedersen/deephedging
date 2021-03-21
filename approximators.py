# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu, softplus

class Approximator(tf.keras.layers.Layer, abc.ABC):
    def __init__(self, output_dim, internal_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.internal_dim = int(internal_dim)


    @abc.abstractmethod
    def _call(self, inputs, training=False):
        """Returns the output from a call.
        Args:
            inputs: ...
            training: bool
        Returns:
            output: (batch_size, output_dim + internal_dim)
        """


    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.layers.Layer."""
        full_output = self._call(inputs, training)
        output, internal = tf.split(
            full_output, [self.output_dim, self.internal_dim], 1)

        return output, internal


class DenseApproximator(Approximator):
    def __init__(self,
                 num_layers,
                 num_units,
                 output_dim,
                 internal_dim,
                 activation=softplus,
                 **kwargs):
        super().__init__(output_dim, internal_dim)
        self.activation = activation

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(num_layers - 1):
            self.dense_layers.append(
                Dense(units=num_units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=output_dim + internal_dim)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, None)
            training: bool
        Returns:
            output: see Strategy._call
        """
        for dense, batch in zip(self.dense_layers, self.batch_layers):
            inputs = dense(inputs, training=training)
            inputs = batch(inputs, training=training)
            inputs = self.activation(inputs)
        output = self.output_layer(inputs, training=training)

        return output


class FeatureApproximator(Approximator):
    def __init__(self, instrument_dim, **kwargs):
        super().__init__(instrument_dim, 0, trainable=False)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: features (batch_size, instrument_dim)
            training: bool (not used)
        Returns:
            output: see Strategy._call
        """

        return inputs


class IdentityFeatureMap(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        return inputs


class GaussianFeatureMap(tf.keras.layers.Layer):
    def build(self, input_shape):
        shape = (input_shape[-1], )
        self.center = self.add_weight(
            shape=shape, initializer="glorot_uniform", trainable=True)
        self.scale = self.add_weight(
            shape=shape,
            initializer=tf.keras.initializers.constant(1 / 2),
            trainable=True)
        super().build(input_shape)


    def call(self, inputs, training=False):
        centered = inputs[..., tf.newaxis] - self.center

        return tf.exp(- self.scale * tf.reduce_sum(tf.square(centered), 1))


class LinearFeatureApproximator(Approximator):
    def __init__(self, instrument_dim, mappings, **kwargs):
        super().__init__(instrument_dim, 0, **kwargs)

        self.mappings = [mapping() for mapping in mappings]
        self.bias = self.add_weight(shape=(instrument_dim, ),
                                    initializer="zeros",
                                    trainable=True)


    def build(self, input_shape):
        self.kernels = []
        for shape in input_shape:
            kernel = self.add_weight(shape=(shape[-1], ),
                                     initializer="glorot_uniform",
                                     trainable=True)
            self.kernels.append(kernel)

        assert len(self.kernels) == len(self.mappings), \
            "length of kernels and mappings unequal: " \
                + f"{len(self.kernels)} != {len(self.mappings)}"

        super().build(input_shape)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args / Returns:
            see FeatureStrategy._call
        """
        output = 0.
        for feature, kernel, mapping in zip(inputs, self.kernels, self.mappings):
            output += tf.multiply(kernel, mapping(feature))

        return output + self.bias
