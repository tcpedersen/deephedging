# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softplus, linear
from tensorflow.keras.initializers import VarianceScaling

# ============================================================================
# === Sequential
class SequentialNeuralNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_layers, num_units, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.num_units = num_units
        self.output_dim = output_dim

        self.hidden_layers = \
            [Dense(self.num_units, use_bias=True, activation=softplus,
                   kernel_initializer=VarianceScaling())
             for _ in range(self.num_layers - 1)]

        self.output_layer = Dense(self.output_dim, use_bias=True,
                                  activation=linear,
                                  kernel_initializer=VarianceScaling())

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        y = self.output_layer(x)
        return y