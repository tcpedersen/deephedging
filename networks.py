# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import sigmoid, linear, relu
from tensorflow.keras.initializers import VarianceScaling

# ============================================================================
# === Sequential
class SequentialNeuralNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_layers, num_units, output_dim,
                 output_activation, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_units = num_units
        self.output_dim = output_dim

        self.hidden_layers = \
            [Dense(self.num_units, use_bias=True, activation=relu,
                   kernel_initializer=VarianceScaling())
             for _ in range(self.num_layers - 1)]

        self.batch_norm = \
            [BatchNormalization() for _ in range(self.num_layers - 1)]

        self.output_layer = Dense(self.output_dim, use_bias=True,
                                  activation=output_activation,
                                  kernel_initializer=VarianceScaling())

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for layer, normalisation in zip(self.hidden_layers, self.batch_norm):
            x = layer(x)
            x = normalisation(x, training)
        y = self.output_layer(x)
        return y