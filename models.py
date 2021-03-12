# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu

from constants import FLOAT_DTYPE

# ==============================================================================
# === Strategy layers
class Strategy(tf.keras.layers.Layer, abc.ABC):
    def __init__(self, instrument_dim, internal_dim, **kwargs):
        super().__init__(**kwargs)
        self.instrument_dim = int(instrument_dim)
        self.internal_dim = int(internal_dim)


    @abc.abstractmethod
    def _call(self, inputs, training=False):
        """Returns the output from a call.
        Args:
            inputs: ...
            training: bool
        Returns:
            output: (batch_size, instrument_dim + internal_dim)
        """


    def call(self, inputs, training=False):
        """Implementation of call for tf.keras.layers.Layer."""
        output = self._call(inputs, training)
        hedge, internal = tf.split(
            output, [self.instrument_dim, self.internal_dim], 1)

        return hedge, internal


class DenseStrategy(Strategy):
    def __init__(self,
                 instrument_dim,
                 num_layers,
                 num_units,
                 internal_dim=0,
                 **kwargs):
        super().__init__(instrument_dim, internal_dim)

        self.instrument_dim = int(instrument_dim)
        self.internal_dim = int(internal_dim)

        self.dense_layers = []
        self.batch_layers = []

        for _ in range(num_layers - 1):
            self.dense_layers.append(
                Dense(units=num_units, use_bias=False))
            self.batch_layers.append(BatchNormalization())

        self.output_layer = Dense(units=self.instrument_dim + self.internal_dim)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: (batch_size, None)
            training: bool
        Returns:
            output: see Strategy._call
        """
        for dense, batch in zip(self.dense_layers, self.batch_layers):
            inputs = dense(inputs)
            inputs = batch(inputs, training=training)
            inputs = relu(inputs)
        output = self.output_layer(inputs)

        return output


class FeatureStrategy(Strategy):
    def __init__(self, instrument_dim, **kwargs):
        super().__init__(instrument_dim, 0, trainable=False)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: features (batch_size, instrument_dim)
            training: bool (not used)
        Returns:
            output: see Strategy._call
        """

        return inputs


class IdentityFeatureMap(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        return inputs


class GaussianFeatureMap(tf.keras.layers.Layer):
    def build(self, input_shape):
        shape = (input_shape[-1], )
        self.center = self.add_weight(
            shape=shape, initializer="glorot_uniform", trainable=True)
        self.scale = self.add_weight(
            shape=shape,
            initializer=tf.keras.initializers.constant(1 / 2),
            trainable=True)
        super().build(input_shape)


    def call(self, inputs, training=False):
        centered = inputs[..., tf.newaxis] - self.center

        return tf.exp(- self.scale * tf.reduce_sum(tf.square(centered), 1))


class LinearFeatureStrategy(Strategy):
    def __init__(self, instrument_dim, mappings, **kwargs):
        super().__init__(instrument_dim, 0, **kwargs)

        self.mappings = [mapping() for mapping in mappings]
        self.bias = self.add_weight(shape=(instrument_dim, ),
                                    initializer="zeros",
                                    trainable=True)


    def build(self, input_shape):
        self.kernels = []
        for shape in input_shape:
            kernel = self.add_weight(shape=(shape[-1], ),
                                     initializer="glorot_uniform",
                                     trainable=True)
            self.kernels.append(kernel)

        assert len(self.kernels) == len(self.mappings), \
            "length of kernels and mappings unequal: " \
                + f"{len(self.kernels)} != {len(self.mappings)}"

        super().build(input_shape)


    def _call(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args / Returns:
            see FeatureStrategy._call
        """
        output = 0.
        for feature, kernel, mapping in zip(inputs, self.kernels, self.mappings):
            output += tf.multiply(kernel, mapping(feature))

        return output + self.bias


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
# === Models
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

        self.instrument_dim = self.strategy_layers[0].instrument_dim
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


    @tf.function
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
# === DeltaHedge
class DeltaHedge(Hedge):
    def __init__(self, timesteps, instrument_dim, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = [FeatureStrategy(instrument_dim)] * timesteps


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
                LinearFeatureStrategy(instrument_dim, mappings))

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


# ==============================================================================
# === SimpleHedge
class SimpleHedge(Hedge):
    def __init__(
            self, timesteps, instrument_dim, num_layers, num_units, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = [DenseStrategy(
                instrument_dim, 1, instrument_dim)]
        for _ in range(timesteps - 1):
            self._strategy_layers.append(DenseStrategy(
                instrument_dim, num_layers, num_units))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        observation = features[..., step]

        if self.cost_layers:
            observation = tf.concat([observation, hedge], 1)

        return observation


# ==============================================================================
# === MemoryHedge
class MemoryHedge(Hedge):
    def __init__(self,
                 timesteps,
                 instrument_dim,
                 internal_dim,
                 num_layers,
                 num_units,
                 **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = [DenseStrategy(
                instrument_dim, 1, num_units, internal_dim)]
        for _ in range(timesteps - 1):
            self._strategy_layers.append(DenseStrategy(
                instrument_dim, num_layers, num_units, internal_dim))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, features, hedge, internal) -> tf.Tensor:
        observation = tf.concat([features[..., step], internal], 1)

        if self.cost_layers:
            observation = tf.concat([observation, hedge], 1)

        return observation


# =============================================================================
# ===
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
        xc = tf.clip_by_value(x, x.dtype.min, tf.math.log(FLOAT_DTYPE.max) / self.aversion - 1.)
        return tf.math.exp(self.aversion * xc) - (1. + tf.math.log(self.aversion)) \
            / self.aversion


class ExpectedShortfall(OCERiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)


    def loss(self, x):
        return tf.nn.relu(x) / (1. - self.alpha)
