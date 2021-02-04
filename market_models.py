# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from constants import FLOAT_PI, FLOAT_DTYPE

ONE_OVER_SQRT_TWO_PI = 1. / tf.sqrt(2. * FLOAT_PI)
SQRT_TWO = tf.sqrt(tf.constant(2., FLOAT_DTYPE))

# ==============================================================================
# ===
@tf.function
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * tf.exp(-0.5 * tf.square(x))

@tf.custom_gradient
def norm_cdf(x):
    y = 0.5 * (1. + tf.math.erf(x / SQRT_TWO))

    def grad(dy):
        return norm_pdf(x) * dy
    return y, grad

@tf.custom_gradient
def norm_qdf(x):
    y = tf.math.erfinv(2 * x - 1) * SQRT_TWO

    def grad(dy):
        return dy / norm_pdf(y)

    return y, grad

# ==============================================================================
# === Back Account
class ConstantRateBankAccount(object):
    def __init__(self, rate):
        self.rate = tf.convert_to_tensor(rate, FLOAT_DTYPE)

    def sample_path(self, maturity, spot, num_paths, num_timesteps):
        time_grid = tf.cast(tf.linspace(0., maturity, num_timesteps + 1),
                            FLOAT_DTYPE)
        return spot * tf.exp(self.rate * time_grid)


# ==============================================================================
# === Black Scholes model
class BlackScholes(object):
    def __init__(self, drift, rate, vol):
        self.drift = tf.convert_to_tensor(drift, FLOAT_DTYPE)
        self.rate = tf.convert_to_tensor(rate, FLOAT_DTYPE)
        self.vol = tf.convert_to_tensor(vol, FLOAT_DTYPE)

    def forward(self, maturity, spot):
        return spot * tf.exp(self.rate * maturity)

    def deflator(self, maturity):
        return tf.exp(-self.rate * maturity)

    # ==========================================================================
    # === pricing
    def black_price(self, maturity, spot, strike, theta):
        forward = self.forward(maturity, spot)
        m = tf.math.log(forward / strike)
        v = self.vol * tf.sqrt(maturity)
        delta = self.deflator(maturity)

        return delta * theta \
            * (forward * norm_cdf(theta * (m / v + v / 2.)) \
               - strike * norm_cdf(theta * (m / v - v / 2.)))

    def black_delta(self, maturity, spot, strike, theta):
        forward = self.forward(maturity, spot)
        m = tf.math.log(forward / strike)
        v = self.vol * tf.sqrt(maturity)
        return theta * norm_cdf(theta * (m / v + v / 2.))

    def call_price(self, maturity, spot, strike):
        return self.black_price(maturity, spot, strike, 1.)

    def call_delta(self, maturity, spot, strike):
        return self.black_delta(maturity, spot, strike, 1.)

    # ==========================================================================
    # === monte carlo
    def sample_path(self, maturity, spot, num_paths, num_timesteps, measure):
        '''
        Returns
        -------
        An num_paths x (num_timesteps + 1) tensor
        '''
        if measure == "q":
            mu = self.rate
        elif measure == "p":
            mu = self.drift
        else:
            raise ValueError(f"measure must be either p or q, not {measure}.")


        # sobol = tf.math.sobol_sample(num_timesteps, num_paths, dtype=FLOAT_DTYPE)
        # rvs = norm_qdf(sobol)

        rvs = tf.random.normal((num_paths, num_timesteps), dtype=FLOAT_DTYPE)

        # use numpy as it needs to be modified
        paths = np.zeros((num_paths, num_timesteps + 1))
        paths[:, 0] = spot

        dt = maturity / num_timesteps

        for idx in range(num_timesteps):
            paths[:, idx + 1] = paths[:, idx] \
                * np.exp((mu - self.vol**2 / 2) * dt \
                         + self.vol * np.sqrt(dt) * rvs[:, idx])

        return tf.convert_to_tensor(paths, FLOAT_DTYPE)