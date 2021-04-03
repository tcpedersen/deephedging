# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization

class MemoryTwinNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, units, internal_dim):
        super().__init__()
        self.internal_dim = int(internal_dim)
        self.output_dim = 1

        self.activation = tf.keras.activations.softplus

        self.input_layer = BatchNormalization()
        self.dense_layers = []
        self.batch_layers = []

        for _ in range(layers - 1):
            self.dense_layers.append(Dense(units=units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=self.output_dim + self.internal_dim)

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


class SemiRecurrentTwinNetwork(tf.keras.models.Model):
    def __init__(self, timesteps, layers, units, internal_dim):
        super().__init__()
        self.approximators = []

        for step in range(timesteps + 1):
            approximator = MemoryTwinNetwork(
                layers=layers,
                units=units,
                internal_dim=internal_dim if step < timesteps else 0
                )

            self.approximators.append(approximator)


    def observation(self, step, inputs, internal) -> tf.Tensor:
        observation = inputs[..., step]
        if step == 0:
            observation, internal = tf.split(
                observation, [tf.shape(observation)[-1], 0], 1)

        return [observation, internal]


    def call(self, inputs, training=False):
        output_func = []
        output_grad = []
        internal = 0. # not used for step == 0

        for step, approx in enumerate(self.approximators):
            observation = self.observation(step, inputs, internal)
            output, internal, gradient = approx(observation, training=training)
            output_func.append(output)
            output_grad.append(gradient)

        return tf.concat(output_func, -1), tf.stack(output_grad, -1)
