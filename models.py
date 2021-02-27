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
        super().__init__(**kwargs)

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


class DeltaStrategy(object):
    def __init__(self, book, **kwargs):
        self.book = book


    def __call__(self, inputs, training=False):
        """Implementation of call for Strategy.
        Args:
            inputs: [time, instruments, numeraire]
                time: (1, )
                instruments: (batch_size, instrument_dim)
                numeraire: (1, )
            training: bool
        Returns:
            output: (batch_size, instrument_dim)
        """
        time, instruments, numeraire = inputs
        discounted_delta = self.book.delta(
            time, instruments[..., tf.newaxis], numeraire)

        return discounted_delta[..., -1] * numeraire


# ==============================================================================
# === Cost Layers
class ProportionalCost(tf.keras.layers.Layer):
    def __init__(self, cost, **kwargs):
        super().__init__(**kwargs)
        self.cost = tf.constant(float(cost), FLOAT_DTYPE)


    @tf.function
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
        self._use_gradients = False
        self._cost_layers = []


    @property
    def num_steps(self) -> int:
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

        for _ in range(self.num_steps):
            self._cost_layers.append(ProportionalCost(float(cost)))


    def compile(self, risk_measure, **kwargs):
        super().compile(**kwargs)

        assert issubclass(type(risk_measure), OCERiskMeasure)
        self.risk_measure = risk_measure


    @tf.function
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
    def observation(self, step, information, hedge) -> tf.Tensor:
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
            inputs: [information, martingales, payoff]
                information: (batch_size, None, num_steps)
                martingales: (batch_size, instrument_dim, num_steps + 1)
                payoff: (batch_size, ) payoff at time num_steps + 1
                gradients (optional): (batch_size, instrument_dim, num_steps)
        Returns:
            value: (batch_size, )
            costs: (batch_size, )
        """
        information, martingales, payoff = inputs

        costs = tf.zeros_like(payoff, FLOAT_DTYPE)
        value = tf.zeros_like(payoff, FLOAT_DTYPE)
        hedge = tf.zeros_like(martingales[..., 0], FLOAT_DTYPE)

        for step, strategy in enumerate(self.strategy_layers):
            observation = self.observation(step, information, hedge)
            old_hedge = hedge
            hedge = strategy(observation, training)
            increment = martingales[..., step + 1] - martingales[..., step]

            value += tf.reduce_sum(tf.multiply(hedge, increment), 1)
            costs += self.costs(step, hedge, old_hedge, martingales)

        return value, costs


# ==============================================================================
# === DeltaHedge
class DeltaHedge(Hedge):
    def __init__(self, num_steps, book, numeraire, **kwargs):
        super().__init__(**kwargs)
        self.book = book
        self._strategy_layers = [DeltaStrategy(book)] * num_steps
        self.numeraire = numeraire


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    @property
    def dt(self) -> tf.Tensor:
        return tf.constant([self.book.maturity / self.num_steps], FLOAT_DTYPE)


    def observation(self, step, information, hedge) -> tf.Tensor:
        """Implementation of observation for Hedge. Note information must be
        martingales.
        """
        return [step * self.dt, information[..., step],
                self.numeraire[tf.newaxis, step]]


# ==============================================================================
# === SimpleHedge
class SimpleHedge(Hedge):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = []
        for _ in range(num_steps):
            self._strategy_layers.append(DenseStrategy(
                instrument_dim, num_layers, num_units))


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, information, hedge):
        return information[..., step]


class CostSimpleHedge(SimpleHedge):
    def __init__(self, num_steps, instrument_dim, num_layers, num_units, cost,
                 **kwargs):
        super().__init__(num_steps, instrument_dim, num_layers, num_units,
                         **kwargs)
        super().add_cost_layers(cost)

    def observation(self, step, information, hedge):
        non_cost = super().observation(step, information, hedge)

        return tf.concat([non_cost, hedge], 1)


# ==============================================================================
# === RecurrentHedge
class RecurrentHedge(Hedge):
    def __init__(
            self, num_steps, instrument_dim, num_layers, num_units, **kwargs):
        super().__init__(**kwargs)

        self._strategy_layers = [
            DenseStrategy(
                instrument_dim,
                num_layers,
                num_units)] * num_steps


    @property
    def strategy_layers(self) -> list:
        return self._strategy_layers


    def observation(self, step, information, hedge):
        time = tf.ones_like(information[..., step], FLOAT_DTYPE) * step

        return tf.concat([time, information[..., step]], 1)


class CostRecurrentHedge(RecurrentHedge):
    def __init__(self, num_steps, instrument_dim, num_layers, num_units, cost,
                 **kwargs):
        super().__init__(num_steps, instrument_dim, num_layers, num_units,
                         **kwargs)
        super().add_cost_layers(cost)

    def observation(self, step, information, hedge):
        non_cost = super().observation(step, information, hedge)

        return tf.concat([non_cost, hedge], 1)


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


    def loss(self, x):
        return tf.exp(self.aversion * x) - (1. + tf.math.log(self.aversion)) \
            / self.aversion


class ExpectedShortfall(OCERiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)


    def loss(self, x):
        return tf.nn.relu(x) / (1. - self.alpha)