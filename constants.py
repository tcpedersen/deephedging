# -*- coding: utf-8 -*-
import tensorflow as tf

FLOAT_DTYPE = tf.float32
INT_DTYPE = tf.int32

STR_FLOAT_DTYPE = 'float64' if FLOAT_DTYPE == tf.float64 else 'float32'

FLOAT_PI = tf.constant(3.1415926535897932384626433832795028841971, FLOAT_DTYPE)
FLOAT_ZERO = tf.constant(0., FLOAT_DTYPE)

# === determine machine epsilon
FLOAT_DTYPE_EPS = tf.constant(1., FLOAT_DTYPE)
while FLOAT_DTYPE_EPS + tf.constant(1., FLOAT_DTYPE) != \
    tf.constant(1., FLOAT_DTYPE):
        FLOAT_DTYPE_EPS /= 2.