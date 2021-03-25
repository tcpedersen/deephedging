# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc

import approximators
from constants import FLOAT_DTYPE

# ==============================================================================
# === Cost Layers
class ProportionalCost(tf.keras.layers.Layer):
    def __init__(self, cost, **kwargs):
        super().__init__(**kwargs)
        self.cost = tf.constant(float(cost), FLOAT_DTYPE)


    def call(self, inputs):
        """Implementation of call for ProportionalCost.
        Args:
            inputs: [shift, martingales]
                shift: (batch_size, instrument_dim)
                martingales: (batch_size, instrument_dim)
        Returns:
            output: (batch_size, )
        """
        shift, martingales = inputs

        return tf.reduce_sum(
            self.cost * tf.multiply(martingales, tf.abs(shift)), 1)


# ==============================================================================
# === abstract base class of hedge model
class Hedge(tf.keras.models.Model, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cost_layers = []


    @property
    def timesteps(self) -> int:
        """Returns number of steps in hedge."""
        return len(self.strategy_layers)


    @property
    @abc.abstractmethod
    def strategy_layers(self) -> list:
        """Returns the strategy layers."""


    @property
    def cost_layers(self) -> list:
        """Returns the cost layers or None if no transaction costs."""
        return self._cost_layers


    def add_cost_layers(self, cost: float):
        if self._cost_layers:
            raise ValueError("cost layers already exists.")

        for _ in range(self.timesteps):
            self._cost_layers.append(ProportionalCost(float(cost)))


    def compile(self, risk_measure, **kwargs):
        super().compile(**kwargs)

        assert issubclass(type(risk_measure), OCERiskMeasure)
        self.risk_measure = risk_measure

        self.instrument_dim = self.strategy_layers[0].output_dim
        self.internal_dim = self.strategy_layers[0].internal_dim


    def train_step(self, data):
        (x, ) = data

        with tf.GradientTape() as tape:
            value, costs = self(x, training=True)
            wealth = value - costs - x[-1]
            loss = self.risk_measure(wealth)

        trainable_vars = [self.risk_measure.w] + self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}


    @abc.abstractmethod
    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        """Returns the input to the strategy layers.
        Args:
            see Hedge.call
        Returns:
            input: (batch_size, )
        """


    def costs(self, step, new_hedge, old_hedge, martingales):
        """Returns the costs associated with a trade.
        Args:
            step: int
            new_hedge / old_hedge: (batch_size, instrument_dim)
            martingales: (batch_size, instrument_dim)
        Returns:
            (batch_size, )
        """
        if self.cost_layers:
            shift = new_hedge - old_hedge
            return self.cost_layers[step]([shift, martingales[..., step]])

        return tf.zeros_like(new_hedge[..., 0], FLOAT_DTYPE)


    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.models.Model.
        The martingales and payoff are assumed to be expressed in terms of the
        numeraire.

        Args:
            inputs: [features, martingales, payoff]
                features: (batch_size, None, timesteps)
                martingales: (batch_size, instrument_dim, timesteps + 1)
                payoff: (batch_size, ) payoff at time timesteps + 1
        Returns:
            value: (batch_size, )
            costs: (batch_size, )
        """
        features, martingales, payoff = inputs
        batch_size = tf.shape(martingales)[0]

        costs = tf.zeros_like(payoff, FLOAT_DTYPE)
        value = tf.zeros_like(payoff, FLOAT_DTYPE)

        hedge = tf.zeros((batch_size, self.instrument_dim), FLOAT_DTYPE)
        internal = tf.zeros((batch_size, self.internal_dim), FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, features, hedge, internal)
            old_hedge = hedge
            hedge, internal = strategy(observation, training)
            increment = martingales[..., step + 1] - martingales[..., step]

            value += tf.reduce_sum(tf.multiply(hedge, increment), 1)
            costs += self.costs(step, hedge, old_hedge, martingales)

        return value, costs


    @tf.function
    def hedge_ratios(self, inputs):
        features, martingales, payoff = inputs
        batch_size = tf.shape(martingales)[0]

        hedge_ratios = []
        internal_ratios = []

        hedge = tf.zeros((batch_size, self.instrument_dim), FLOAT_DTYPE)
        internal = tf.zeros((batch_size, self.internal_dim), FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, features, hedge, internal)
            hedge, internal = strategy(observation, training=False)
            hedge_ratios.append(hedge)
            internal_ratios.append(internal)

        return tf.stack(hedge_ratios, -1), tf.stack(internal_ratios, -1)


# ==============================================================================
# === hedge models
class DeltaHedge(Hedge):
    def __init__(self, timesteps, instrument_dim, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = [
            approximators.FeatureApproximator(instrument_dim)] * timesteps


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        """Implementation of observation for Hedge. Note features must be
        hedge ratios.
        """
        return features[..., step]


class LinearFeatureHedge(Hedge):
    def __init__(self, timesteps, instrument_dim, mappings, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = []
        for _ in range(timesteps):
            self._strategy_layers.append(
                approximators.LinearFeatureApproximator(
                    instrument_dim, mappings))

    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        """Implementation of observation for Hedge. Note features must be
        hedge ratios.
        """
        observation = [features[..., step]]

        if self.cost_layers:
            observation += [hedge]

        return observation


class SimpleHedge(Hedge):
    def __init__(
            self, timesteps, instrument_dim, num_layers, num_units, **kwargs):
        super().__init__()

        self._strategy_layers = [approximators.DenseApproximator(
                num_layers=1,
                num_units=instrument_dim,
                output_dim=instrument_dim,
                internal_dim=0,
                **kwargs)]
        for _ in range(timesteps - 1):
            self._strategy_layers.append(approximators.DenseApproximator(
                num_layers=num_layers,
                num_units=num_units,
                output_dim=instrument_dim,
                internal_dim=0,
                **kwargs))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        observation = features[..., step]

        if self.cost_layers:
            observation = tf.concat([observation, hedge], 1)

        return observation


class MemoryHedge(Hedge):
    def __init__(self,
                 timesteps,
                 instrument_dim,
                 internal_dim,
                 num_layers,
                 num_units,
                **kwargs):
        super().__init__()

        self._strategy_layers = [approximators.DenseApproximator(
                num_layers=1,
                num_units=num_units,
                output_dim=instrument_dim,
                internal_dim=internal_dim,
                **kwargs)]
        for _ in range(timesteps - 1):
            self._strategy_layers.append(approximators.DenseApproximator(
                num_layers=num_layers,
                num_units=num_units,
                output_dim=instrument_dim,
                internal_dim=internal_dim,
                **kwargs))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        observation = tf.concat([features[..., step], internal], 1)

        if self.cost_layers:
            observation = tf.concat([observation, hedge], 1)

        return observation


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