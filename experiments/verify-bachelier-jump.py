# -*- coding: utf-8 -*-
import tensorflow as tf
import utils

import derivatives

from constants import FLOAT_DTYPE

timesteps = 14

maturity = timesteps / 250
spot = 100.0
strike = 100.0
rate = 0.0
volatility = 0.4

intensity = 1.3
jumpsize = -0.2
jumpvol = 0.15

option = derivatives.BachelierJumpCall(
    maturity, strike, volatility, intensity, jumpsize, jumpvol)

# simulation
trials = int(2**24)
s0 = tf.ones((trials, ), FLOAT_DTYPE) * spot

dw = tf.random.normal((trials, )) * tf.sqrt(maturity)

normals = tf.random.normal((trials, ))
poisson = tf.random.poisson((trials, ), intensity * maturity, FLOAT_DTYPE)
dn = poisson * jumpsize + tf.sqrt(poisson) * jumpvol * normals

s1 = s0 - intensity * jumpsize * maturity + volatility * dw + dn

time = tf.constant([0.0, maturity], FLOAT_DTYPE)
instrument = tf.stack([s0, s1], 1)
numeraire = tf.ones_like(time)
payoff = option.payoff(time, instrument, numeraire)
adjoint = option.adjoint(time, instrument, numeraire)

print(f"MC price: {utils.precise_mean(payoff):.4f}")
print(f"CF price: {option.value(time, instrument[0, tf.newaxis, ...], numeraire)[0, 0]:4f}")

print(f"MC delta: {utils.precise_mean(adjoint):.4f}")
print(f"CF delta: {option.delta(time, instrument[0, tf.newaxis, ...], numeraire)[0, 0]:4f}")
