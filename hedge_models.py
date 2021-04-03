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
    def __init__(self, append_internal, append_hedge):
        super().__init__()
        self.append_internal = bool(append_internal)
        self.append_hedge = bool(append_hedge)
        self.cost_layers = []
        self.internal_batch = []


    @property
    def timesteps(self) -> int:
        """Returns number of steps in hedge."""
        return len(self.strategy_layers)


    @property
    @abc.abstractmethod
    def strategy_layers(self) -> list:
        """Returns the strategy layers."""


    def add_cost_layers(self, cost: float):
        if self.cost_layers:
            raise ValueError("cost layers already exists.")

        for _ in range(self.timesteps):
            self.cost_layers.append(ProportionalCost(float(cost)))


    def add_internal_batch_normalisation(self):
        if self.internal_batch:
            raise ValueError("internal batch normalisation already exists.")
        elif not self.append_internal:
            raise ValueError("cannot use internal batch normalisation ",
                             "when append_internal is False.")

        for _ in range(self.timesteps - 1):
            self.internal_batch.append(tf.keras.layers.BatchNormalization())


    def compile(self, risk_measure, **kwargs):
        super().compile(**kwargs)

        assert issubclass(type(risk_measure), OCERiskMeasure)
        self.risk_measure = risk_measure


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
            if self.append_hedge and self.cost_layers:
                observation = tf.concat([observation, hedge], 1)

        return observation


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
        costs = tf.zeros_like(payoff, FLOAT_DTYPE)
        value = tf.zeros_like(payoff, FLOAT_DTYPE)

        hedge = 0.
        internal = 0.

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, features, hedge, internal)
            old_hedge = hedge
            hedge, internal = strategy(observation, training)
            increment = martingales[..., step + 1] - martingales[..., step]

            value += tf.reduce_sum(tf.multiply(hedge, increment), 1)
            costs += self.costs(step, hedge, old_hedge, martingales)

            if self.internal_batch and step < self.timesteps - 1:
                internal = self.internal_batch[step](internal, training=training)

        return value, costs


# ==============================================================================
# === hedge models
class DeltaHedge(Hedge):
    def __init__(self, timesteps, instrument_dim):
        super().__init__(append_internal=False, append_hedge=False)

        self._strategy_layers = [
            approximators.FeatureApproximator(instrument_dim)] * timesteps


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


class LinearFeatureHedge(Hedge):
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


class NeuralHedge(Hedge):
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


class ConstantHedge(Hedge):
    def __init__(self, timesteps, instrument_dim):
        super().__init__(False, False)
        self._strategy_layers = []
        for step in range(timesteps):
            self._strategy_layers.append(tf.keras.layers.Lambda(self.zeros))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def zeros(self, inputs, *args, **kwargs):
        return [tf.zeros_like(inputs, FLOAT_DTYPE)] * 2

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