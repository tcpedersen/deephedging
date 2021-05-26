# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense, BatchNormalization

class Approximator(tf.keras.layers.Layer, abc.ABC):
    def __init__(self, output_dim, internal_dim):
        super().__init__()
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
                 activation):
        super().__init__(output_dim, internal_dim)
        self.activation = activation

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(num_layers - 1):
            self.dense_layers.append(
                Dense(units=num_units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=output_dim + internal_dim)


    def _evaluate_layer(self, x, dense, batch, training):
        x = dense(x, training=training)
        x = batch(x, training=training)
        return self.activation(x)

    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, None)
            training: bool
        Returns:
            output: see Strategy._call
        """
        for dense, batch in zip(self.dense_layers, self.batch_layers):
            inputs = self._evaluate_layer(inputs, dense, batch, training)
        output = self.output_layer(inputs, training=training)

        return output


    def initialise(self, inputs, sample_size):
        batch_size = tf.shape(inputs)[0]

        iterator = zip(self.dense_layers, self.batch_layers)
        for k, (dense, batch) in enumerate(iterator):
            sample_idx = tf.random.shuffle(tf.range(batch_size))[:sample_size]
            sample = tf.gather(inputs, sample_idx, axis=0)

            for i in tf.range(k + 1):
                sample = self._evaluate_layer(
                    sample,
                    self.dense_layers[i],
                    self.batch_layers[i],
                    False)
            mean, variance = tf.nn.moments(sample, 0)

            # dense.set_weights([dense.get_weights()[0] / tf.sqrt(variance)])
            batch.set_weights([
                batch.get_weights()[0] / tf.sqrt(variance),
                (batch.get_weights()[1] - mean) / tf.sqrt(variance),
                batch.get_weights()[2],
                batch.get_weights()[3]])

        sample_idx = tf.random.shuffle(tf.range(batch_size))[:sample_size]
        sample = tf.gather(inputs, sample_idx, axis=0)
        sample = self._call(sample, False)
        mean, variance = tf.nn.moments(sample, 0)
        self.output_layer.set_weights(
            [self.output_layer.get_weights()[0] / tf.sqrt(variance),
             self.output_layer.get_weights()[1]])


class FeatureApproximator(Approximator):
    def __init__(self, instrument_dim, **kwargs):
        super().__init__(instrument_dim, 0)


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


class LinearFeatureApproximator(Approximator):
    def __init__(self, instrument_dim, mappings):
        super().__init__(instrument_dim, 0)

        self.mappings = [mapping() for mapping in mappings]
        self.bias = self.add_weight(name="bias",
                                    shape=(instrument_dim, ),
                                    initializer="zeros",
                                    trainable=True)


    def build(self, input_shape):
        self.kernels = []
        for i in range(len(self.mappings)):
            kernel = self.add_weight(name=f"kernel_{i}",
                                     shape=(self.output_dim, ),
                                     initializer="glorot_uniform",
                                     trainable=True)
            self.kernels.append(kernel)

        super().build(input_shape)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args / Returns:
            see FeatureStrategy._call
        """
        inputs = tf.split(inputs, len(self.mappings), 1)
        output = 0.0
        iterator = zip(inputs, self.kernels, self.mappings)
        for feature, kernel, mapping in iterator:
            output += tf.multiply(kernel, mapping(feature))

        return output + self.bias


class MatrixFeatureApproximator(Approximator):
    def __init__(self, instrument_dim):
        super().__init__(instrument_dim, 0)
        self.bias = self.add_weight(name="bias",
                                    shape=(instrument_dim, ),
                                    initializer="zeros",
                                    trainable=True)

        self.matrix = self.add_weight(name="matrix",
                                      shape=(instrument_dim, instrument_dim),
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.use_cost = False

    def build(self, input_shape):
        if input_shape[1] == self.output_dim:
            pass
        elif input_shape[1] == 2 * self.output_dim:
            # add cost weight
            self.costkernel = self.add_weight(name="cost_kernel",
                                              shape=(self.output_dim, ),
                                              initializer="glorot_uniform",
                                              trainable=True)
            self.use_cost = True
        else:
            raise NotImplementedError


    def _call(self, inputs, training=False):
        inputs = tf.split(inputs, 2 if self.use_cost else 1, 1)
        output = tf.matmul(inputs[0], self.matrix, transpose_b=True) + self.bias
        if self.use_cost:
            output += tf.multiply(self.costkernel, inputs[1])

        return output
