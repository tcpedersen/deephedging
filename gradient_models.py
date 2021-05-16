# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc

from tensorflow.keras.layers import Dense

class FeedForwardNeuralNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, units, output_dim, activation,
                 output_activation=None):
        super().__init__()

        self.hidden_layers = []
        for _ in range(layers - 1):
            layer = Dense(units, activation=activation)
            self.hidden_layers.append(layer)

        self.output_layer = Dense(output_dim, activation=output_activation)

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


    def basis(self, x):
        e = []
        exponent = tf.cast(tf.range(self.degree + 1), x.dtype)
        for v in tf.unstack(x, axis=-1):
            e.append(tf.pow(v[:, tf.newaxis], exponent))

        return tf.concat(e, -1)


    def basis_jacobian(self, x):
        exponent = tf.cast(tf.range(self.degree + 1), x.dtype)
        diffexpo = exponent - 1.0
        adjphi = exponent * tf.pow(x, diffexpo)
        # if x is zero adjphi is 0 * 0**(-1)
        return tf.where(tf.math.is_nan(adjphi), 0.0, adjphi)


    def fit(self, x, y, dydx):
        xc = tf.cast(x, tf.float64)
        yc = tf.cast(y, tf.float64)
        dydxc = tf.cast(dydx, tf.float64)

        phi = self.basis(xc)
        adjphi = self.basis_jacobian(xc)
        w = tf.linalg.norm(yc, axis=0)**2 / tf.linalg.norm(dydxc, axis=0)**2
        matrix = w * tf.transpose(phi) @ phi \
            + (1 - w) * tf.transpose(adjphi) @ adjphi
        rhs = w * tf.transpose(phi) @ yc \
            + (1 - w) * tf.transpose(adjphi) @ dydxc

        self.kernel.assign(tf.cast(np.linalg.solve(matrix, rhs), x.dtype))


    def call(self, inputs, training=False):
        phi = self.basis(inputs)
        return phi @ self.kernel


    def gradient(self, inputs, training=False):
        adjphi = self.basis_jacobian(inputs)
        return adjphi @ self.kernel


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
            inputs:
                x: (batch, input_dim, timesteps)
        Returns:
            output:
                y: (batch, output_dim, timesteps)
                dydx: (batch, output_dim, input_dim, timesteps)
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


    def fit(self, inputs, outputs, **kwargs):
        x, (y, dydx) = inputs, outputs
        for step, poly in enumerate(self.polynomials):
            poly.fit(x[..., step], y[..., tf.newaxis], dydx[..., step])


    def call(self, inputs, training=False):
        outputs = []
        for step, poly in enumerate(self.polynomials):
            outputs.append(poly(inputs[..., step]))

        return tf.stack(outputs, -1)


    def gradient(self, inputs, training=False):
        outputs = []
        for step, poly in enumerate(self.polynomials):
            outputs.append(poly.gradient(inputs[..., step]))

        return tf.stack(outputs, -1)
