# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc
import functools
import operator

import utils
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


    def compute_cost(self, martingales, strategy, return_full=False):
        # increment[..., timesteps + 1] is implicitly zero
        increment = strategy[..., 1:] - strategy[..., :-1]
        proportions = self.cost * martingales[..., :-2]
        costs = tf.reduce_sum(proportions * tf.abs(increment), [1, 2])

        return (costs, increment, proportions) if return_full else costs


    def compute_value(self, martingales, strategy, return_full=False):
        increment = martingales[..., 1:] - martingales[..., :-1]
        value = tf.reduce_sum(tf.multiply(strategy, increment), [1, 2])

        return (value, increment) if return_full else value


    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.models.Model.
        The martingales and payoff are assumed to be expressed in terms of the
        numeraire.

        Args:
            inputs: [features, martingales]
                features: list of (batch_size, instrument_dim) of len timesteps
                martingales: (batch_size, instrument_dim, timesteps + 1)
        Returns:
            value: (batch_size, )
            costs: (batch_size, )
        """
        features, martingales = inputs
        strategy = self.strategy(features, training=training)
        value = self.compute_value(martingales, strategy)
        if self.cost > 0:
            costs = self.compute_cost(martingales, strategy)
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
    def __init__(self, append_internal, append_hedge, observation_normalise):
        super().__init__()
        self.append_internal = bool(append_internal)
        self.append_hedge = bool(append_hedge)
        self.observation_normalise = bool(observation_normalise)

        if self.observation_normalise:
            self.batch_layers = []
            for _ in tf.range(len(self.strategy_layers)):
                batch = tf.keras.layers.BatchNormalization()
                self.batch_layers.append(batch)


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
        hedge = 0.0
        internal = 0.0

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, features, hedge, internal)

            if self.observation_normalise:
                observation = self.batch_layers[step](
                    observation, training=training)

            hedge, internal = strategy(observation, training=training)
            strategies.append(hedge)

        return tf.stack(strategies, -1)


class LinearFeatureHedge(SemiRecurrentHedge):
    def __init__(self, timesteps, instrument_dim, mappings):
        super().__init__(append_internal=False,
                         append_hedge=(len(mappings) > 1),
                         observation_normalise=False)

        self._strategy_layers = []
        for step in range(timesteps):
            approximator = approximators.LinearFeatureApproximator(
                instrument_dim=instrument_dim,
                mappings=[mappings[0]] if step == 0 else mappings
                )
            self._strategy_layers.append(approximator)


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


class MatrixFeatureHedge(SemiRecurrentHedge):
    def __init__(self, timesteps, instrument_dim, withcost):
        super().__init__(append_internal=False,
                         append_hedge=withcost,
                         observation_normalise=False)

        self._strategy_layers = []
        for step in range(timesteps):
            approximator = approximators.MatrixFeatureApproximator(
                instrument_dim=instrument_dim)
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
                 activation,
                 observation_normalise=False):
        super().__init__(
            append_internal=(internal_dim > 0),
            append_hedge=True,
            observation_normalise=observation_normalise)

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


    def initialise(self, features, sample_size):
        """Initialise weigths depending on the features.
        Args:
            see BaseHedge.strategy
        """
        self.strategy(features, training=True) # initialise batch normalisation
        batch_size = tf.shape(features[0])[0]

        for step, strategy in enumerate(self.strategy_layers):
            sample_idx = tf.random.shuffle(tf.range(batch_size))[:sample_size]
            sample = [tf.gather(features[k], sample_idx, axis=0) \
                      for k in tf.range(step + 1)]

            hedge = 0.0
            internal = 0.0
            for k in tf.range(step + 1):
                observation = self.observation(k, sample, hedge, internal)
                hedge, internal = self.strategy_layers[k](observation,
                                                          training=False)

            if step > 0:
                strategy.initialise(observation, sample_size)

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
        self.w = tf.Variable(0.0, trainable=True)

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


    @abc.abstractmethod
    def gradient(self, x: tf.Tensor) -> tf.Tensor:
        """Returns derivative of loss."""


class MeanVariance(OCERiskMeasure):
    def __init__(self, aversion):
        super().__init__()
        self.aversion = tf.constant(float(aversion), FLOAT_DTYPE)


    def loss(self, x):
        return x + self.aversion / 2. * tf.square(x)


    def evaluate(self, x):
        mean, variance = utils.cast_apply(tf.nn.moments, x, axes=0)

        return self.aversion / 2. * variance - mean


    def gradient(self, x):
        return 1.0 + self.aversion * x


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


    def gradient(self, x):
        return tf.cast(x > 0, FLOAT_DTYPE) / (1.0 - self.alpha)
