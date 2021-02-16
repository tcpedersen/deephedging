# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

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


    def compile(self, loss_fn, **kwargs):
        super(SimpleHedge, self).compile(**kwargs)
        self.loss_fn = loss_fn


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
        wealth = -payoff

        for step, strategy in enumerate(self.strategy_layers):
            holdings = strategy(information[..., step], training)
            increment = trade[:, :, step + 1] - trade[:, :, step]
            wealth += tf.reduce_sum(tf.multiply(holdings, increment), 1)

        return wealth


    @tf.function
    def train_step(self, data):
        (x, ) = data
        with tf.GradientTape() as tape:
            y = self(x, training=True)
            loss = self.loss_fn(y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}


# =============================================================================
# ===
class RiskMeasure(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: tf.Tensor):
        """Returns the associated risk of x.
        Args:
            x: (batch_size, )
        Returns:
            :math: \rho(x) (1, )
        """


class EntropicRisk(RiskMeasure):
    def __init__(self, risk_aversion):
        self.aversion = float(risk_aversion)

    @tf.function
    def __call__(self, x):
        exp = tf.exp(-self.aversion * x)
        return tf.math.log(tf.reduce_mean(exp, -1)) / self.aversion