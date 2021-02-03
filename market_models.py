# -*- coding: utf-8 -*-
import tensorflow as tf
from constants import FLOAT_PI, FLOAT_DTYPE

ONE_OVER_SQRT_TWO_PI = 1. / tf.sqrt(2. * FLOAT_PI)
SQRT_TWO = tf.sqrt(tf.constant(2., FLOAT_DTYPE))

# ==============================================================================
# ===
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * tf.exp(-0.5 * tf.square(x))

def norm_cdf(x):
    return 0.5 * (1. + tf.math.erf(x / SQRT_TWO))

# ==============================================================================
# === Black Scholes model
class BlackScholes(object):
    def __init__(self, rate, vol):
        self.rate = tf.convert_to_tensor(rate, FLOAT_DTYPE)
        self.vol = tf.convert_to_tensor(vol, FLOAT_DTYPE)

    def forward(self, maturity, spot):
        return spot * tf.exp(self.rate * maturity)

    def deflator(self, maturity):
        return tf.exp(-self.rate * maturity)

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