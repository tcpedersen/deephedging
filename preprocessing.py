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
            x: list of (batch_size, None)
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
    def fit(self, inputs):
        self.mean = []
        self.variance = []

        for x in inputs:
            # major numerical imprecision in reduce_mean for tf.float32,
            # so convert to tf.float64 before calculating moments.
            m, v = tf.nn.moments(tf.cast(x, tf.float64), 0)
            self.mean.append(m)
            self.variance.append(v)


    def transform(self, inputs):
        outputs = []
        for x, mean, variance in zip(inputs, self.mean, self.variance):
            outputs.append(self._transform(x, mean, variance))

        return outputs


    def _transform(self, x, mean, variance):
        xc = tf.cast(x, tf.float64)
        yc = tf.nn.batch_normalization(xc, mean, variance, None, None, 0.)
        yc = tf.where(tf.equal(variance, 0), 0., yc)

        return tf.cast(yc, x.dtype)


    def inverse_transform(self, inputs):
        outputs = []
        for y, mean, variance in zip(inputs, self.mean, self.variance):
            outputs.append(self._inverse_transform(y, mean, variance))

        return outputs


    def _inverse_transform(self, y, mean, variance):
        yc = tf.cast(y, tf.float64)
        xc = tf.sqrt(variance) * yc + mean
        xc = tf.where(tf.equal(variance, 0.), mean, xc)

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


    def transform_x(self, x):
        norm_x = tf.nn.batch_normalization(
            tf.cast(x, tf.float64), self.xmean, self.xvar, None, None, 0.)

        return tf.cast(norm_x, x.dtype)

    def transform_y(self, y):
        norm_y = tf.nn.batch_normalization(
            tf.cast(y, tf.float64), self.ymean, self.yvar, None, None, 0.)

        return tf.cast(norm_y, y.dtype)

    def transform_dydx(self, dydx):
        norm_dydx = tf.cast(dydx, tf.float64) * tf.sqrt(self.xvar / self.yvar)

        return tf.cast(norm_dydx, dydx.dtype)

    def transform(self, x, y, dydx):
        """Assumes:
            x: (batch, dimension, timesteps)
            y: (batch, )
            dydx: (batch, dimension, timesteps)
        """
        return self.transform_x(x), self.transform_y(y), \
            self.transform_dydx(dydx)


    def inverse_transform_x(self, norm_x):
        x = tf.sqrt(self.xvar) * tf.cast(norm_x, tf.float64) + self.xmean

        return tf.cast(x, norm_x.dtype)


    def inverse_transform_y(self, norm_y):
        y = tf.sqrt(self.yvar) * tf.cast(norm_y, tf.float64) + self.ymean

        return tf.cast(y, norm_y.dtype)


    def inverse_transform_dydx(self, norm_dydx):
        dydx = tf.cast(norm_dydx, tf.float64) * tf.sqrt(self.yvar / self.xvar)

        return tf.cast(dydx, norm_dydx.dtype)


    def inverse_transform(self, norm_x, norm_y, norm_dydx):
        return self.inverse_transform_x(norm_x), \
            self.inverse_transform_y(norm_y), \
                self.inverse_transform_dydx(norm_dydx)


    def fit_transform(self, x, y, dydx):
        self.fit(x, y)

        return self.transform(x, y, dydx)

