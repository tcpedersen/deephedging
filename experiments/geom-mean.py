# -*- coding: utf-8 -*-
import tensorflow as tf

from simulators import GBM
import utils

batch = int(2**20)
gbm = GBM(0.02, [0.05], [[0.2]])

tau = 1.5
init_state = tf.constant([100.], tf.float32)
rate, drift, vol = gbm.rate, gbm.drift, gbm.volatility

time = tf.constant([0, 0.1, 0.3, 0.4, 0.7, 0.75, 1., 1.5], tf.float32)
m = len(time) - 1
dt = time[1:] - time[:-1]

def xi(k):
    mu = (rate - vol**2 / 2) * tf.reduce_sum(
        (time[(k + 1):] - time[k]) * dt[k:])
    vsq = vol**2 * tf.reduce_sum((tau - time[k:-1])**2 * dt[k:])

    return tf.math.exp(mu + vsq / 2.)

paths = gbm.simulate(time, init_state, batch, True)[:, 0, :]

for k in tf.range(m):
    geom = tf.math.reduce_prod(tf.pow(paths[..., (k + 1):], dt[k:]), -1) \
        / tf.pow(paths[..., k], tau - time[k])

    mean = utils.precise_mean(geom)
    std = utils.precise_func(tf.math.reduce_std, geom)
    lci = mean - 1.96 * std / tf.sqrt(float(batch))
    rci = mean + 1.96 * std / tf.sqrt(float(batch))

    print(f"emperical: {utils.precise_mean(geom).numpy(): .7f}")
    print(f"analytical: {xi(k).numpy()[0]: .7f}")
    print(f"CI: ({lci.numpy():.7f}, {rci.numpy():.7f})")
    print()

    assert lci < xi(k) < rci, f"failed for k={k}."


def price_at_t(t, spot, k):
    assert time[k] <= t <= time[k + 1]
    return spot**(tau - time[k]) \
        * tf.math.reduce_prod(tf.pow(paths[..., 1:(k + 1)], dt[:k]), -1) \
            * xi(k + 1) \
                * tf.math.exp(
                    (rate - vol**2 / 2 + vol**2 / 2 * (tau - time[k])) \
                        * (tau - time[k]) * (time[k + 1] - t))

def price_at_k(k):
    return tf.pow(paths[..., k], tau - time[k]) \
        * tf.math.reduce_prod(tf.pow(paths[..., 1:(k + 1)], dt[:k]), -1) \
            * xi(k)


for k in tf.range(m):
    tp = price_at_t(time[k], paths[..., k], k)
    kp = price_at_k(k)

    tf.debugging.assert_near(tp, kp)