# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from scipy.special import erfinv
from tensorflow_probability.python.internal import special_math

from constants import FLOAT_DTYPE_EPS, FLOAT_DTYPE

ONE_OVER_SQRT_TWO_PI = 1. / np.sqrt(2. * np.pi)
SQRT_TWO = np.sqrt(2.)

# ==============================================================================
# === Gaussian
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * np.exp(-0.5 * x * x)

def norm_cdf(x):
    return special_math.ndtr(x)

def norm_qdf(x):
    return erfinv(2. * x - 1.) * SQRT_TWO

def near_positive_definite(A):
    C = (A + tf.transpose(A)) / 2.
    eigval, eigvec = tf.linalg.eig(C)
    eigval = tf.where(tf.math.real(eigval) < 0, 0, eigval)
    psd = tf.math.real(eigvec @ tf.linalg.diag(eigval) @ tf.transpose(eigvec))

    return psd + tf.eye(psd.shape[0], dtype=FLOAT_DTYPE) * tf.sqrt(FLOAT_DTYPE_EPS)

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