# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense

class FeedForwardNeuralNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, units, output_dim, activation):
        super().__init__()

        self.hidden_layers = []
        for _ in range(layers - 1):
            layer = Dense(units, activation=activation)
            self.hidden_layers.append(layer)

        self.output_layer = Dense(output_dim)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        y = self.output_layer(x)

        return y


class Polynomial(tf.keras.layers.Layer):
    def __init__(self, degree):
        super().__init__()
        self.degree = int(degree)
        self.kernel = self.add_weight("kernel", (degree + 1, 1))


    def expand_feature(self, x):
        return tf.concat([tf.pow(x, float(n)) \
                          for n in tf.range(self.degree + 1)], 1)

    def call(self, inputs, training=False):
        x = self.expand_feature(inputs)

        return x @ self.kernel

# ==============================================================================
# ===
class SequenceNetwork(tf.keras.Model, abc.ABC):
    def __init__(self, layers, units, activation):
        super().__init__()

        self.num_layers = layers
        self.units = units
        self.activation = activation


    def build(self, input_shape, output_dim):
        """Implementation of build."""
        timesteps = input_shape[-1]

        self.networks = []
        for _ in range(timesteps):
            network = FeedForwardNeuralNetwork(
                layers=self.num_layers,
                units=self.units,
                output_dim=output_dim,
                activation=self.activation
                )
            self.networks.append(network)


    @abc.abstractmethod
    def gradient(self, inputs, training=False):
        """Implementation of call.
        Args:
            inputs: (batch, None, timesteps)
        Returns:
            output: (batch, None, timesteps)
        """


class SequenceValueNetwork(SequenceNetwork):
    def __init__(self, layers, units, activation):
        super().__init__(layers, units, activation)


    def build(self, input_shape):
        super().build(input_shape, 1)


    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        """Implementation of call.
        Args:
            inputs: (batch, None, timesteps)
        Returns:
            output: (batch, timesteps)
        """
        outputs = []

        for step, network in enumerate(self.networks):
            x = inputs[..., step]
            y = network(x, training=training)

            outputs.append(y)

        return tf.concat(outputs, -1)


    def gradient(self, inputs, training=False):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            y = self(inputs, training=training)
        dydx = tape.gradient(y, inputs)

        return dydx


class SequenceTwinNetwork(SequenceNetwork):
    def __init__(self, layers, units, activation):
        super().__init__(layers, units, activation)


    def build(self, input_shape):
        super().build(input_shape, 1)


    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        """Implementation of call.
        Args:
            inputs: (batch, dimension, timesteps)
        Returns:
            output: (batch, timesteps), (batch, None, timesteps)
        """
        outputs = []
        gradients = []

        for step, network in enumerate(self.networks):
            x = inputs[..., step]

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                y = network(x, training=training)
            dydx = tape.gradient(y, x)

            outputs.append(y)
            gradients.append(dydx)

        return tf.concat(outputs, -1), tf.stack(gradients, -1)

    def gradient(self, inputs, training=False):
        return self(inputs, training=training)[1]


class SequenceDeltaNetwork(SequenceNetwork):
    def __init__(self, layers, units, activation):
        super().__init__(layers, units, activation)


    def build(self, input_shape):
        super().build(input_shape, input_shape[1])


    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        """Implementation of call.
        Args:
            inputs: (batch, dimension, timesteps)
        Returns:
            output: (batch, None, timesteps)
        """
        outputs = []

        for step, network in enumerate(self.networks):
            x = inputs[..., step]
            y = network(x, training=training)

            outputs.append(y)

        return tf.stack(outputs, -1)


    def gradient(self, inputs, training=False):
        return self(inputs, training=training)


class SequencePolynomial(tf.keras.Model):
    def __init__(self, degree):
        super().__init__()
        self.degree = int(degree)

    def build(self, input_shape):
        """Implementation of build."""
        timesteps = input_shape[-1]

        self.polynomials = []
        for _ in range(timesteps):
            self.polynomials.append(Polynomial(self.degree))


    def fit(self, x, y, weight, **kwargs):
        raise NotImplementedError()
