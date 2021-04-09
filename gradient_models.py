# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization

class FeedForwardNeuralNetwork(tf.keras.Model):
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


class MemoryTwinNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, units, internal_dim):
        super().__init__()
        self.internal_dim = int(internal_dim)
        self.output_dim = 1

        self.activation = tf.keras.activations.softplus

        self.input_layer = BatchNormalization(trainable=False) # TODO remove
        self.dense_layers = []
        self.batch_layers = []

        for _ in range(layers - 1):
            self.dense_layers.append(Dense(units=units, use_bias=False))
            self.batch_layers.append(BatchNormalization(trainable=False)) # TODO

        self.output_layer = Dense(units=self.output_dim + self.internal_dim)

    def call(self, inputs, training=False):
        feature, internal = inputs

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(feature)
            x = tf.concat([feature, internal], -1)
            # x = self.input_layer(x, training=training)
            for dense, batch in zip(self.dense_layers, self.batch_layers):
                x = dense(x)
                # x = batch(x, training=training)
                x = self.activation(x)
            output, internal = tf.split(
                self.output_layer(x), [self.output_dim, self.internal_dim], 1)
        gradient = tape.gradient(output, feature)

        return [output, internal, gradient]


class SemiRecurrentTwinNetwork(tf.keras.models.Model):
    def __init__(self, timesteps, layers, units, internal_dim, use_batchnorm):
        super().__init__()
        self.approximators = []
        self.timesteps = timesteps

        for step in range(timesteps + 1):
            approximator = MemoryTwinNetwork(
                layers=layers,
                units=units,
                internal_dim=internal_dim if step < timesteps else 0
                )

            self.approximators.append(approximator)


        self.internal_batch = []
        if use_batchnorm:
            if internal_dim > 0:
                for _ in range(timesteps):
                    batch = tf.keras.layers.BatchNormalization(trainable=True)
                    self.internal_batch.append(batch)
            else:
                raise ValueError("cannot use batchnorm if internal_dim <= 0.")

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

            if self.internal_batch and step < self.timesteps:
                internal = self.internal_batch[step](
                    internal,
                    training=training
                    )

            output_func.append(output)
            output_grad.append(gradient)

        return tf.concat(output_func, -1), tf.stack(output_grad, -1)


class TwinLSTMNetwork(tf.keras.models.Model):
    def __init__(self, lstm_cells, lstm_units, network_layers, network_units):
        super().__init__()
        self.lstm_cells = int(lstm_cells)
        self.lstm_units = int(lstm_units)
        self.network_layers = int(network_layers)
        self.network_units = int(network_units)
        self.output_dim = 1


    def build(self, input_shape):
        """Implementation of build.
        Args:
            input_shape: (batch, input_dim, timesteps + 1)
        """
        timesteps = input_shape[2] - 1

        self.networks = []
        self.cells = []

        for step in tf.range(timesteps + 1):

            stack = [tf.keras.layers.LSTMCell(self.lstm_units) \
                     for _ in range(self.lstm_cells)]
            cell = tf.keras.layers.StackedRNNCells(stack)

            network = FeedForwardNeuralNetwork(
                layers=self.network_layers,
                units=self.network_units,
                output_dim=self.output_dim,
                activation=tf.keras.activations.softplus
                )

            self.cells.append(cell)
            self.networks.append(network)


    def observation(self, step, feature, memory) -> tf.Tensor:
        if step == 0:
            return feature

        return tf.concat([feature, memory], axis=-1)


    def warmup(self, x, y, **kwargs):
        self.lstm.compile(optimizer="Adam", loss="mean_squared_error")
        return self.lstm.fit(x, y, **kwargs)

    def call(self, inputs, training=False):
        """Implementation of call.
        Args:
            x: (batch, input_dim, timesteps + 1)
        Returns:
            y: (batch, timesteps + 1)
            dydx: (batch, input_dim, timesteps + 1)
        """
        output_lst = []
        gradient_lst = []

        batch = tf.shape(inputs)[0]
        output = tf.zeros((batch, self.output_dim), inputs.dtype)
        states = self.cells[0].get_initial_state(inputs[..., 0])

        for step, (cell, network) in enumerate(zip(self.cells, self.networks)):
            feature = inputs[..., step]

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(feature)
                memory, states = cell(feature, states)
                # observation = self.observation(step, feature, memory)
                output = network(memory, training=training)
            gradient = tape.gradient(output, feature)

            output_lst.append(output)
            gradient_lst.append(gradient)

        return tf.concat(output_lst, -1), tf.stack(gradient_lst, -1)
