# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import abc

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


class IOMeanVarianceNormaliser:
    def __init__(self):
        self.xnormaliser = MeanVarianceNormaliser()
        self.ynormaliser = MeanVarianceNormaliser()


    def fit(self, x, y):
        """Assumes:
            x: (batch, xdim, timesteps)
            y: (batch, ydim, timesteps)
        """
        self.xnormaliser.fit(tf.unstack(x, axis=-1))
        self.ynormaliser.fit(tf.unstack(y, axis=-1))


    def transform_x(self, x):
        norm_x = self.xnormaliser.transform(tf.unstack(x, axis=-1))

        return tf.stack(norm_x, axis=-1)


    def transform(self, x, y):
        """
        Args:
            x: (batch, xdim, timesteps)
            y: (batch, ydim, timesteps)
        Returns:
            norm_x: (batch, xdim, timesteps)
            norm_y: (batch, ydim, timesteps)
        """

        norm_y = self.ynormaliser.transform(tf.unstack(y, axis=-1))

        return self.transform_x(x), tf.stack(norm_y, axis=-1)


    def inverse_transform_y(self, norm_y):
        y = self.ynormaliser.inverse_transform(tf.unstack(norm_y, axis=-1))

        return tf.stack(y, axis=-1)


    def inverse_transform(self, norm_x, norm_y):
        """
        Returns:
            x: (batch, xdim, timesteps)
            y: (batch, ydim, timesteps)
        Args:
            norm_x: (batch, xdim, timesteps)
            norm_y: (batch, ydim, timesteps)
        """
        x = self.xnormaliser.inverse_transform(tf.unstack(norm_x, axis=-1))
        y = self.inverse_transform_y(norm_y)

        return tf.stack(x, axis=-1), y


    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)


class DifferentialMeanVarianceNormaliser(object):
    def fit(self, x, y):
        """Assumes:
            x: (batch, xdim, timesteps)
            y: (batch, ydim)
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
        cdydx = tf.cast(dydx, tf.float64)
        scale = self.xvar / self.yvar[..., tf.newaxis, tf.newaxis]
        norm_dydx = cdydx * tf.sqrt(scale)

        return tf.cast(norm_dydx, dydx.dtype)

    def transform(self, x, y, dydx):
        """Assumes:
            x: (batch, xdim, timesteps)
            y: (batch, ydim)
            dydx: (batch, ydim, xdim, timesteps)
        """
        return self.transform_x(x), self.transform_y(y), \
            self.transform_dydx(dydx)


    def inverse_transform_x(self, norm_x):
        x = tf.sqrt(self.xvar) * tf.cast(norm_x, tf.float64) + self.xmean

        return tf.cast(x, norm_x.dtype)


    def inverse_transform_y(self, norm_y):
        cnorm_y = tf.cast(norm_y, tf.float64)
        y = tf.sqrt(self.yvar[tf.newaxis, :, tf.newaxis]) * cnorm_y \
            + self.ymean[tf.newaxis, :, tf.newaxis]

        return tf.cast(y, norm_y.dtype)


    def inverse_transform_dydx(self, norm_dydx):
        cnorm_dydx = tf.cast(norm_dydx, tf.float64)
        scale = self.yvar[..., tf.newaxis, tf.newaxis] / self.xvar

        dydx = cnorm_dydx * tf.sqrt(scale)

        return tf.cast(dydx, norm_dydx.dtype)


    def inverse_transform(self, norm_x, norm_y, norm_dydx):
        return self.inverse_transform_x(norm_x), \
            self.inverse_transform_y(norm_y), \
                self.inverse_transform_dydx(norm_dydx)


    def fit_transform(self, x, y, dydx):
        self.fit(x, y)

        return self.transform(x, y, dydx)
