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
class Hedge(tf.keras.models.Model, abc.ABC):
    def __init__(self, **kwargs):
        super(Hedge, self).__init__(**kwargs)


    @property
    @abc.abstractmethod
    def instrument_dim(self) -> int:
        """Returns number of hedge instruments."""


    @property
    @abc.abstractmethod
    def num_steps(self) -> int:
        """Returns number of steps in hedge."""


    @property
    @abc.abstractmethod
    def strategy_layers(self) -> list:
        """Returns the strategy layers."""


    def compile(self, risk_measure, **kwargs):
        super(Hedge, self).compile(**kwargs)
        self.risk_measure = risk_measure


    @tf.function
    def train_step(self, data):
        (x, ) = data
        with tf.GradientTape() as tape:
            y = self(x, training=True)
            loss = self.risk_measure(y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}


    @abc.abstractmethod
    def observation(self, step, information, holdings) -> tf.Tensor:
        """Returns the input to the strategy layers.
        Args:
            see Hedge.call
        Returns:
            input: (batch_size, None)
        """


    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.models.Model.
        Args:
            inputs: [information, trade, payoff]
                information: (batch_size, None, num_steps)
                trade: (batch_size, instrument_dim + 1, num_steps + 1)
                payoff: (batch_size, ) payoff at time num_steps + 1
        Returns:
            value: (batch_size, )
        """
        information, trade, payoff = inputs
        wealth = -payoff
        holdings = tf.zeros_like(trade[:, :self.instrument_dim, 0], FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, information, holdings)
            holdings = strategy(observation, training)
            increment = trade[:, :, step + 1] - trade[:, :, step]
            wealth += tf.reduce_sum(tf.multiply(holdings, increment), 1)

        return wealth


class SimpleHedge(Hedge):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super(SimpleHedge, self).__init__(**kwargs)

        self._num_steps = int(num_steps)
        self._instrument_dim = int(instrument_dim)

        self._strategy_layers = []
        for _ in range(self.num_steps):
            self._strategy_layers.append(Strategy(
                self.instrument_dim, num_layers, num_units))

    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    @property
    def instrument_dim(self) -> int:
        return self._instrument_dim


    @property
    def num_steps(self) -> int:
        return self._num_steps


    @tf.function
    def observation(self, step, information, holdings):
        """Implementation of observation for SimpleHedge."""
        return information[..., step]

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