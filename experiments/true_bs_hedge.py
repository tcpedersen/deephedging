# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_agents.metrics import py_metrics
from tf_agents.drivers.py_driver import PyDriver

from environments import BlackScholesCallEnv
from policies import BlackScholesDeltaPolicy

maturity, asset_spot, strike = 1.25, 100, 100
drift, rate, vol = 0.1, 0.01, 0.3

num_hedges_each_year = 2**13

env = BlackScholesCallEnv(maturity, asset_spot, strike, drift, rate, vol,
                          num_hedges_each_year)
policy = BlackScholesDeltaPolicy(strike, drift, rate, vol)

replay_buffer = []
metric = py_metrics.AverageReturnMetric()
observers = [replay_buffer.append, metric]
driver = PyDriver(env, policy, observers, max_episodes=1)

initial_time_step = env.reset()
final_time_step, _ = driver.run(initial_time_step)

print('Average Return: ', metric.result())

# === stats
payoff = (asset - strike) * tf.cast(asset > strike, FLOAT_DTYPE)
print(f"mean hedge error: {tf.reduce_mean(V - payoff):.4}.")
print(f"mean absolute hedge error: {tf.reduce_mean(tf.abs(V - payoff)):.4}.")
print(f"mean squared hedge error: {tf.reduce_mean(tf.square(V - payoff)):.4}.")

# === plot
xmin, xmax = 80., 110.
x = tf.linspace(xmin, xmax, 1000)
y = (x - strike) * tf.cast(x > strike, FLOAT_DTYPE)

inplot = (xmin < asset) & (asset < xmax)

plt.figure()
plt.plot(x, y, color="black")
plt.scatter(tf.boolean_mask(asset, inplot), tf.boolean_mask(V, inplot),
            5, alpha=0.5, color="cyan")
plt.show()
