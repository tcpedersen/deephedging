# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu

from constants import FLOAT_DTYPE

# ==============================================================================
# === Strategy layers
class DenseStrategy(tf.keras.layers.Layer):
    def __init__(self, instrument_dim, num_layers, num_units, **kwargs):
        super(DenseStrategy, self).__init__(**kwargs)

        self.instrument_dim = int(instrument_dim)

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(num_layers - 1):
            self.dense_layers.append(
                Dense(units=num_units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=self.instrument_dim)

    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, None)
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
# === Cost Layers
class ProportionalCost(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        """Implementation of call for ProportionalCost.
        Args:
            inputs: [delta_holdings, instruments]
                delta_holdings: (batch_size, instrument_dim)
                instruments: (batch_size, instrument_dim)
        Returns:
            output: (batch_size, )
        """
        delta_holdings, instruments = inputs
        return tf.reduce_sum(tf.multiply(delta_holdings * instruments), 1)

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


    @property
    def cost_layers(self) -> list:
        """Returns the cost layers or None if no transaction costs."""
        return None


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
    def observation(self, step, information, hedge) -> tf.Tensor:
        """Returns the input to the strategy layers.
        Args:
            see Hedge.call
        Returns:
            input: (batch_size, )
        """


    def costs(self, step, new_hedge, old_hedge, instruments):
        """Returns the costs associated with a trade.
        Args:
            step: int
            new_hedge / old_hedge: (batch_size, instrument_dim)
            instruments: (batch_size, instrument_dim)
        Returns:
            (batch_size, )
        """
        if self.cost_layers is not None:
            delta = new_hedge - old_hedge
            return self.cost_layers[step]([delta, instruments[..., step]])

        return tf.zeros_like(new_hedge[..., 0], FLOAT_DTYPE)


    @tf.function
    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.models.Model.
        Args:
            inputs: [information, instruments, payoff]
                information: (batch_size, None, num_steps)
                instruments: (batch_size, instrument_dim, num_steps + 1)
                payoff: (batch_size, ) payoff at time num_steps + 1
        Returns:
            value: (batch_size, )
        """
        information, instruments, payoff = inputs
        wealth = -payoff
        hedge = tf.zeros_like(instruments[..., 0], FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, information, hedge)
            old_hedge = hedge
            hedge = strategy(observation, training)
            costs = self.costs(step, hedge, old_hedge, instruments)

            increment = instruments[:, :, step + 1] - instruments[:, :, step]
            wealth += tf.reduce_sum(tf.multiply(hedge, increment), 1)
            wealth -= costs

        return wealth


class SimpleHedge(Hedge):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super(SimpleHedge, self).__init__(**kwargs)

        self._num_steps = int(num_steps)
        self._instrument_dim = int(instrument_dim)

        self._strategy_layers = []
        for _ in range(self.num_steps):
            self._strategy_layers.append(DenseStrategy(
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
    def observation(self, step, information, hedge):
        """Implementation of observation for SimpleHedge."""
        return information[..., step]

# =============================================================================
# ===
class RiskMeasure(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Returns the associated risk of x.
        Args:
            x: (batch_size, )
        Returns:
            (1, )
        """


class EntropicRisk(RiskMeasure):
    def __init__(self, risk_aversion):
        self.aversion = float(risk_aversion)

    @tf.function
    def __call__(self, x):
        exp = tf.exp(-self.aversion * x)
        return tf.math.log(tf.reduce_mean(exp, 0)) / self.aversion


class MeanSquareRisk(RiskMeasure):
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(x), 0)