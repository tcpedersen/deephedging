# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import softplus

from constants import FLOAT_DTYPE

# ==============================================================================
# === Layers
class Strategy(tf.keras.layers.Layer):
    def __init__(self, instrument_dim, num_layers, num_units, **kwargs):
        super(Strategy, self).__init__(**kwargs)

        self.num_layers = int(num_layers)
        self.num_units = int(num_units)
        self.instrument_dim = int(instrument_dim)

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(self.num_layers - 1):
            self.dense_layers.append(Dense(self.num_units, softplus)) # TODO change activation
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(self.instrument_dim)

    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, instrument_dim)
            training: # TODO
        Returns:
            output: (batch_size, instrument_dim)
        """
        for dense, batch in zip(self.dense_layers, self.batch_layers):
            inputs = dense(inputs)
            inputs = batch(inputs, training=training)
        output = self.output_layer(inputs)

        return output


# ==============================================================================
# === Models
class SimpleHedge(tf.keras.models.Model):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super(SimpleHedge, self).__init__(**kwargs)

        self.instrument_dim = int(instrument_dim)
        self.num_steps = int(num_steps)
        self.num_layers = int(num_layers)
        self.num_units = int(num_units)

        self.strategy_layers = []
        for num in range(self.num_steps):
            self.strategy_layers.append(Strategy(
                self.instrument_dim, self.num_layers, self.num_units))


    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for SimpleHedge.
        Args:
            inputs: [information, payoff]
                information: (batch_size, instrument_dim, num_steps + 1)
                payoff: (batch_size, ) payoff at time num_steps + 1
        Returns:
            output: # TODO
        """
        information, payoff = inputs
        value = tf.zeros_like(payoff, FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            info = information[..., step]
            holdings = strategy(info, training)
            increment = tf.math.subtract(information[..., step + 1], info)
            value += tf.reduce_sum(tf.math.multiply(holdings, increment), -1)
        value += payoff

        return value

# =============================================================================
# ===
class EntropicRisk(tf.keras.losses.Loss):
    def __init__(self, risk_aversion, **kwargs):
        super(EntropicRisk, self).__init__(**kwargs)
        self.risk_aversion = float(risk_aversion)

    def call(self, y_true, y_pred):
        #return tf.exp(-self.risk_aversion * y_pred) # TODO
        return tf.math.log(y_pred)