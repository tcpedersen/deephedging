# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import abc

from constants import FLOAT_DTYPE

class Normaliser(abc.ABC):
    @abc.abstractmethod
    def fit(self, x):
        """Fit normaliser to data.
        Args:
            x: (batch_size, None, timesteps + 1)
        """


    @abc.abstractmethod
    def transform(self, x):
        """Normalise data
        Args:
            x: same dimensions as input to .fit.
        Returns:
            y: same dimensions as x.
        """


    @abc.abstractmethod
    def inverse_transform(self, x):
        """Renormalise data
        Args:
            y: same dimensions as output of .transform.
        Returns:
            x: same dimensions as input to .transform.
        """


    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class MeanVarianceNormaliser(Normaliser):
    def fit(self, x):
        # major numerical imprecision in reduce_mean for tf.float32,
        # so convert to tf.float64 before calculating moments.
        self.mean, self.variance = tf.nn.moments(tf.cast(x, tf.float64), 0)
        self.eps = tf.constant(0., tf.float64)


    def transform(self, x):
        xc = tf.cast(x, tf.float64)
        yc = tf.nn.batch_normalization(
            xc, self.mean, self.variance, None, None, self.eps)
        yc = tf.where(tf.equal(self.variance, 0), 0., yc)

        return tf.cast(yc, x.dtype)


    def inverse_transform(self, y):
        yc = tf.cast(y, tf.float64)
        xc = tf.sqrt(self.variance + self.eps) * yc + self.mean
        xc = tf.where(tf.equal(self.variance, 0.), self.mean, xc)

        return tf.cast(xc, y.dtype)


class ZeroComponentAnalysis(Normaliser):
    def __init__(self):
        self.eps = tf.constant(1e-4, tf.float64)

    def fit(self, x):
        xc = tf.cast(x, tf.float64)
        self.mean = tf.reduce_mean(xc, 0)
        cov = tfp.stats.covariance(xc, sample_axis=0, event_axis=1)

        self.w = []
        self.inv_w = []

        for step in tf.range(tf.shape(xc)[-1]):
            # u @ tf.linalg.diag(s) @ tf.transpose(v) = cov
            s, u, v = tf.linalg.svd(cov[..., step],
                                    full_matrices=True, compute_uv=True)
            s = tf.where(tf.abs(s) < self.eps, 1, s)
            self.w.append(u @ tf.linalg.diag(s**(-1 / 2)) @ tf.transpose(v))
            self.inv_w.append(v @ tf.linalg.diag(s**(1 / 2)) @ tf.transpose(u))

        self.w = tf.convert_to_tensor(self.w, tf.float64)
        self.inv_w = tf.convert_to_tensor(self.inv_w, tf.float64)


    def transform(self, x):
        norm_xc = tf.cast(x, tf.float64) - self.mean
        yc = []
        for step in tf.range(tf.shape(norm_xc)[-1]):
            yc.append(norm_xc[..., step] @ self.w[step, ...])

        return tf.transpose(tf.convert_to_tensor(yc, x.dtype), [1, 2, 0])


    def inverse_transform(self, y):
        yc = tf.cast(y, tf.float64)

        norm_xc = []
        for step in tf.range(tf.shape(yc)[-1]):
            norm_xc.append(yc[..., step] @ self.inv_w[step, ...])

        return tf.cast(tf.stack(norm_xc, -1) + self.mean, y.dtype)


class DifferentialMeanVarianceNormaliser(object):
    def fit(self, x, y):
        """Assumes:
            x: (batch, dimension, timesteps)
            y: (batch, )
        """
        # major numerical imprecision in reduce_mean for tf.float32,
        # so convert to tf.float64 before calculating moments.
        self.xmean, self.xvar = tf.nn.moments(tf.cast(x, tf.float64), 0)
        self.ymean, self.yvar = tf.nn.moments(tf.cast(y, tf.float64), 0)


    def transform(self, x, y, dydx):
        """Assumes:
            x: (batch, dimension, timesteps)
            y: (batch, )
            dydx: (batch, dimension, timesteps)
        """
        norm_x = tf.nn.batch_normalization(
            tf.cast(x, tf.float64), self.xmean, self.xvar, None, None, 0.)
        norm_y = tf.nn.batch_normalization(
            tf.cast(y, tf.float64), self.ymean, self.yvar, None, None, 0.)
        norm_dydx = tf.cast(dydx, tf.float64) * tf.sqrt(self.xvar / self.yvar)

        return (tf.cast(v, FLOAT_DTYPE) for v in [norm_x, norm_y, norm_dydx])


    def inverse_transform(self, norm_x, norm_y, norm_dydx):
        x = tf.sqrt(self.xvar) * tf.cast(norm_x, tf.float64) + self.xmean
        y = tf.sqrt(self.yvar) * tf.cast(norm_y, tf.float64) + self.ymean
        dydx = tf.cast(norm_dydx, tf.float64) * tf.sqrt(self.yvar / self.xvar)

        return (tf.cast(v, FLOAT_DTYPE) for v in [x, y, dydx])


    def fit_transform(self, x, y, dydx):
        self.fit(x, y)

        return self.transform(x, y, dydx)

