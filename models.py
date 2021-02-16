# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu, sigmoid

from constants import INT_DTYPE, FLOAT_DTYPE

# ==============================================================================
# === Layers
class Strategy(tf.keras.layers.Layer):
    def __init__(self, instrument_dim, num_layers, num_units, **kwargs):
        super(Strategy, self).__init__(**kwargs)

        self.instrument_dim = int(instrument_dim)
        self.num_layers = int(num_layers)
        self.num_units = int(num_units)

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(self.num_layers - 1):
            self.dense_layers.append(
                Dense(units=self.num_units, use_bias=False)) # TODO add bias
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=self.instrument_dim)

    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, instrument_dim)
            training: bool
        Returns:
            output: (batch_size, instrument_dim)
        """
        for dense, batch in zip(self.dense_layers, self.batch_layers):
            inputs = dense(inputs)
            inputs = batch(inputs, training=training)
            inputs = relu(inputs)
        output = self.output_layer(inputs)

        return output


# ==============================================================================
# === Models
class SimpleHedge(tf.keras.models.Model):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super(SimpleHedge, self).__init__(**kwargs)

        self.num_steps = int(num_steps)
        self.instrument_dim = int(instrument_dim)
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
            inputs: [information, trade, payoff]
                information: (batch_size, ?, num_steps)
                trade: (batch_size, instrument_dim, num_steps + 1)
                payoff: (batch_size, ) payoff at time num_steps + 1
        Returns:
            value: (batch_size, )
        """
        information, trade, payoff = inputs
        wealth = tf.zeros_like(payoff, FLOAT_DTYPE)
        holdings = tf.zeros_like(trade[..., 0], FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            helper = holdings
            holdings = strategy(information[..., step], training)
            wealth -= tf.reduce_sum(
                tf.multiply(holdings - helper, trade[..., step]), 1)

        wealth += tf.reduce_sum(tf.multiply(holdings, trade[..., -1]), 1)
        wealth -= payoff # TODO minus instead

        return wealth


# =============================================================================
# ===
class EntropicRisk(tf.keras.losses.Loss):
    def __init__(self, risk_aversion, **kwargs):
        super(EntropicRisk, self).__init__(**kwargs)
        self.aversion = float(risk_aversion)

    @tf.function
    def call(self, y_true, y_pred):
        return tf.exp(-self.aversion * y_pred)