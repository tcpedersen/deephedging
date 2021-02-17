# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from scipy.special import erfinv
from tensorflow_probability.python.internal import special_math

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
        self.mean = tf.reduce_mean(x, 0)
        self.std = tf.math.reduce_std(x, 0)

    def transform(self, x):
        '''Normalize data.
        Args:
            x: see LinearNormalizer.fit
        Returns:
            y: same as input.
        '''
        y = (x - self.mean) / self.std
        return tf.where(tf.equal(self.std, 0)[tf.newaxis, ...], 0., y)

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
        x = self.std * y + self.mean
        return tf.where(tf.equal(self.std, 0.), self.mean, x)