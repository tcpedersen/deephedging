# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv
from tensorflow_probability.python.internal import special_math

from constants import FLOAT_DTYPE_EPS, FLOAT_DTYPE
import models

ONE_OVER_SQRT_TWO_PI = 1. / tf.sqrt(2. * np.pi)
SQRT_TWO = tf.sqrt(2.)

# ==============================================================================
# === Gaussian
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * tf.exp(-0.5 * x * x)

def norm_cdf(x):
    return special_math.ndtr(x)

def norm_qdf(x):
    return erfinv(2. * x - 1.) * SQRT_TWO

def near_positive_definite(A):
    dtype = tf.float64
    C = (A + tf.transpose(A)) / 2.
    eigval, eigvec = tf.linalg.eig(tf.cast(C, dtype))
    eigval = tf.where(tf.math.real(eigval) < 0, 0, eigval)
    psd = tf.math.real(eigvec @ tf.linalg.diag(eigval) @ tf.transpose(eigvec))
    eps = tf.sqrt(tf.cast(FLOAT_DTYPE_EPS, dtype))

    return tf.cast(psd + tf.eye(psd.shape[0], dtype=dtype) * eps, FLOAT_DTYPE)

# ==============================================================================
# === Training
class PeakSchedule:
    def __init__(self, a, b, n):
        self.a = a
        self.b = b

        self.n1, self.n2, self.n3 = 0, n // 4, n // 2

    def __call__(self, n, alpha):
        if n <= self.n2:
            return (self.a - self.b)/(self.n1 - self.n2) * n \
                - (self.a * self.n2 - self.b * self.n1) / (self.n1 - self.n2)
        elif self.n2 < n < self.n3:
            return -(self.a - self.b) / (self.n2 - self.n3) * n \
                + (self.a * self.n2 - self.b * self.n3) / (self.n2 - self.n3)
        else:
            return self.a


class MeanVarianceNormaliser:
    def fit(self, x):
        '''Fit normaliser to data.
        Args:
            x: (batch_size, information_dim, num_steps + 1)
        '''

        # major numerical imprecision in reduce_mean for tf.float32,
        # so convert to tf.float64 before calculating moments.
        self.mean, self.variance = tf.nn.moments(tf.cast(x, np.float64), 0)
        self.eps = tf.constant(0., tf.float64)

    def transform(self, x):
        '''Normalize data.
        Args:
            x: see LinearNormalizer.fit
        Returns:
            y: same as input.
        '''
        xc = tf.cast(x, tf.float64)
        yc = tf.nn.batch_normalization(
            xc, self.mean, self.variance, None, None, self.eps)
        yc = tf.where(tf.equal(self.variance, 0), 0., yc)

        return tf.cast(yc, FLOAT_DTYPE)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        '''Normalize data.
        Args:
            y: see LinearNormalizer.fit
        Returns:
            x: same as input.
        '''
        yc = tf.cast(y, tf.float64)
        xc = tf.sqrt(self.variance + self.eps) * yc + self.mean
        xc = tf.where(tf.equal(self.variance, 0.), self.mean, xc)

        return tf.cast(xc, FLOAT_DTYPE)


# ==============================================================================
# === other
def expected_shortfall(wealth, alpha):
    """Emperical expected shortfall."""
    loss = -wealth
    var = np.quantile(loss, alpha)
    return tf.reduce_mean(loss[loss > var])


def precise_mean(x, **kwargs):
    return tf.cast(tf.reduce_mean(tf.cast(x, tf.float64)), x.dtype)


# ==============================================================================
# === experiments
def hedge_model_input(time, instruments, numeraire, book):
    information = tf.math.log(instruments / numeraire)
    martingales = instruments / numeraire
    payoff = book.payoff(time, instruments, numeraire)

    return [information, martingales, payoff]


def benchmark_input(time, instruments, numeraire, book):
    information = instruments / numeraire
    martingales = instruments / numeraire
    payoff = book.payoff(time, instruments, numeraire)

    return [information, martingales, payoff]


def no_liability_input(time, instruments, numeraire, book):
    information, martingales, payoff = hedge_model_input(
        time, instruments, numeraire, book)

    return [information, martingales, tf.zeros_like(payoff)]


def train_model(model, inputs, alpha, normalise=True):
    # normalise data
    normaliser = MeanVarianceNormaliser()
    norm_information = normaliser.fit_transform(inputs[0]) if normalise else inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2]]

    # compile model
    risk_measure = models.ExpectedShortfall(alpha)
    optimizer = tf.keras.optimizers.Adam(1e-1)
    model.compile(risk_measure, optimizer=optimizer)

    # define callbacks
    batch_size, epochs = 2**10, 100

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, min_delta=1e-4, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", verbose=1, patience=2)

    callbacks = [early_stopping, reduce_lr]

    # train
    history = model.fit(norm_inputs,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

    return history, norm_inputs, normaliser


def test_model(model, inputs, normaliser=None):
    # normalise data
    if normaliser is not None:
        norm_information =  normaliser.transform(inputs[0])
    else:
        norm_information = inputs[0]
    norm_inputs = [norm_information, inputs[1], inputs[2]]

    # test model
    value, costs = model(norm_inputs)
    risk = model.risk_measure(value - costs - inputs[2])

    return norm_inputs, risk


def plot_distributions(models, inputs, prices):
    data = []
    for model, input, price in zip(models, inputs, prices):
        value, costs = model(input)
        wealth = price + value - costs - input[2]
        data.append([value, costs, wealth])

    sample_size = min(250000, len(value))

    # wealth
    plt.figure()
    for value, costs, wealth in data:
        plot_data = np.random.choice(wealth, sample_size, replace=False)
        plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    plt.show()

    # value
    plt.figure()
    for value, costs, wealth in data:
        plot_data = np.random.choice(value, sample_size, replace=False)
        plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    plt.show()

    # costs
    # plt.figure()
    # for value, costs, wealth in data:
    #     plot_data = np.random.choice(costs, 250000, replace=False)
    #     plt.hist(plot_data, bins=250, density=True, alpha=0.5)
    # plt.show()

    return data


def plot_barrier_payoff(model, norm_inputs, price, instruments, numeraire, book):
    derivative = book.derivatives[0]["derivative"]
    crossed = tf.squeeze(tf.reduce_any(derivative.crossed(instruments), 2))
    payoff = book.payoff(instruments, numeraire)
    xlim = (tf.reduce_min(instruments[:, 0, -1]),
            tf.reduce_max(instruments[:, 0, -1]))

    value, costs = model(norm_inputs)

    for indices in [crossed, ~crossed]:
        m = tf.boolean_mask(instruments[..., 0, -1], indices, 0)

        key = tf.argsort(m, 0)
        x = tf.gather(m, key)

        y1 = tf.gather(payoff[indices], key)
        y2 = tf.gather(tf.boolean_mask(price + value, indices), key)

        plt.figure()
        plt.xlim(*xlim)
        plt.scatter(x, y2, s=0.5)
        plt.plot(x, y1, color="black")
        plt.show()