# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc

import approximators
from constants import FLOAT_DTYPE

# ==============================================================================
# === abstract base class of hedge model
class BaseHedge(tf.keras.models.Model, abc.ABC):
    def __init__(self):
        super().__init__()
        self.cost = 0.


    def add_cost(self, cost: float):
        if self.cost > 0:
            raise ValueError("cost already exists.")
        self.cost = float(cost)


    def compile(self, risk_measure, **kwargs):
        super().compile(**kwargs)

        assert issubclass(type(risk_measure), OCERiskMeasure)
        self.risk_measure = risk_measure


    def train_step(self, data):
        (x, ) = data

        with tf.GradientTape() as tape:
            value, costs = self(x[:-1], training=True)
            wealth = value - costs - x[-1]
            loss = self.risk_measure(wealth)

        trainable_vars = [self.risk_measure.w] + self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}


    @abc.abstractmethod
    def strategy(self, features, training=False):
        """The strategy of the agent.
        Args:
            features: list of (batch_size, instrument_dim) of len timesteps
        Returns:
            strategy: (batch_size, instrument_dim, timesteps)
        """


    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.models.Model.
        The martingales and payoff are assumed to be expressed in terms of the
        numeraire.

        Args:
            inputs: [features, martingales, payoff]
                features: list of (batch_size, instrument_dim) of len timesteps
                martingales: (batch_size, instrument_dim, timesteps + 1)
                payoff: (batch_size, ) payoff at time timesteps + 1
        Returns:
            value: (batch_size, )
            costs: (batch_size, )
        """
        features, martingales = inputs
        strategy = self.strategy(features, training=training)

        increment = martingales[..., 1:] - martingales[..., :-1]
        value = tf.reduce_sum(tf.multiply(strategy, increment), [1, 2])

        if self.cost > 0:
            pad = [[0, 0], [0, 0], [1, 0]]
            increment = tf.abs(strategy - tf.pad(strategy[..., :-1], pad))
            proportions = tf.reduce_sum(
                tf.multiply(martingales[..., :-1], increment), [1, 2])
            costs = self.cost * proportions
        else:
            costs = tf.zeros((tf.shape(martingales)[0], ), FLOAT_DTYPE)

        return value, costs


class FeatureHedge(BaseHedge):
    """Simply returns the features as strategy."""
    def strategy(self, features, training=False):
        return tf.stack(features, -1)


# ==============================================================================
# === semi-recurrent hedge strategies
class SemiRecurrentHedge(BaseHedge):
    def __init__(self, append_internal, append_hedge):
        super().__init__()
        self.append_internal = bool(append_internal)
        self.append_hedge = bool(append_hedge)


    @property
    @abc.abstractmethod
    def strategy_layers(self) -> list:
        """Returns the strategy layers."""


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        """Returns the input to the strategy layers.
        Args:
            see Hedge.call
        Returns:
            input: (batch_size, )
        """
        observation = features[step]

        if step > 0:
            if self.append_internal:
                observation = tf.concat([observation, internal], 1)
            if self.append_hedge and self.cost > 0:
                observation = tf.concat([observation, hedge], 1)

        return observation


    def strategy(self, features, training):
        strategies = []
        hedge = 0.
        internal = 0.

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, features, hedge, internal)
            hedge, internal = strategy(observation, training=training)
            strategies.append(hedge)

        return tf.stack(strategies, -1)


class LinearFeatureHedge(SemiRecurrentHedge):
    def __init__(self, timesteps, instrument_dim, mappings):
        super().__init__(append_internal=False,
                         append_hedge=(len(mappings) > 1))

        self._strategy_layers = []
        for step in range(timesteps):
            approximator = approximators.LinearFeatureApproximator(
                instrument_dim=instrument_dim,
                mappings=[mappings[0]] if step == 0 else mappings)
            self._strategy_layers.append(approximator)


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


class NeuralHedge(SemiRecurrentHedge):
    def __init__(self,
                 timesteps,
                 instrument_dim,
                 internal_dim,
                 num_layers,
                 num_units,
                 activation):
        super().__init__(append_internal=(internal_dim > 0), append_hedge=True)

        self._strategy_layers = []

        for step in range(timesteps):
            self._strategy_layers.append(
                approximators.DenseApproximator(
                    num_layers=1 if step == 0 else num_layers,
                    num_units=1 if step == 0 else num_units,
                    output_dim=instrument_dim,
                    internal_dim=internal_dim if step < timesteps - 1 else 0,
                    activation=activation)
                )

    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers

# ==============================================================================
# === fully recurrent hedge strategies
class RecurrentHedge(BaseHedge):
    def __init__(self, timesteps, rnn, cells, units, instrument_dim):
        super().__init__()
        self.rnn = []
        for _ in range(cells):
            cell = rnn(units, return_sequences=True)
            self.rnn.append(cell)

        self.output_layers = []
        for _ in range(timesteps):
            layer = tf.keras.layers.Dense(instrument_dim)
            self.output_layers.append(layer)


    def strategy(self, features, training=False):
        sequence = tf.stack(features, 1)
        for network in self.rnn:
            sequence = network(sequence)

        strategies = []
        for step, layer in enumerate(self.output_layers):
            hedge = layer(sequence[:, step, :], training=training)
            strategies.append(hedge)

        return tf.stack(strategies, -1)


# =============================================================================
# === risk measures
class OCERiskMeasure(abc.ABC):
    def __init__(self):
        self.w = tf.Variable(0., trainable=True)

    def __call__(self, x: tf.Tensor):
        return self.w + tf.reduce_mean(self.loss(-x - self.w), 0)

    @abc.abstractmethod
    def loss(self, x: tf.Tensor) -> tf.Tensor:
        """Returns the associated loss of x.
        Args:
            x: (batch_size, )
        Returns:
            loss: (batch_size, )
        """


    @abc.abstractmethod
    def evaluate(self, x: tf.Tensor) -> tf.Tensor:
        """Returns the true value of the estimator."""


class EntropicRisk(OCERiskMeasure):
    def __init__(self, risk_aversion):
        super().__init__()
        self.aversion = float(risk_aversion)


    def __call__(self, x: tf.Tensor):
        redundant = tf.square(self.w) # to ensure w = 0 in optimum.
        batch_size = tf.cast(tf.shape(x)[0], FLOAT_DTYPE)

        return (tf.math.reduce_logsumexp(-self.aversion * x) \
            - tf.math.log(batch_size)) / self.aversion + redundant


    def loss(self, x):
        # clip as tf.exp will otherwise overflow.
        xc = tf.clip_by_value(x, x.dtype.min,
                              tf.math.log(FLOAT_DTYPE.max) / self.aversion - 1.)
        return tf.math.exp(self.aversion * xc) \
            - (1. + tf.math.log(self.aversion)) / self.aversion


    def evaluate(self, x):
        raise NotImplementedError("evaluate not implemented for EntropicRisk.")


class ExpectedShortfall(OCERiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)


    def loss(self, x):
        return tf.nn.relu(x) / (1. - self.alpha)


    def evaluate(self, x):
        loss = tf.cast(-x, tf.float64)
        var = np.quantile(loss.numpy(), self.alpha)
        mask = (loss > var)

        return tf.cast(tf.reduce_mean(tf.boolean_mask(loss, mask)), x.dtype)