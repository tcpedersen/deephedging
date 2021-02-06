# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.metrics import py_metrics
from tf_agents.drivers.py_driver import PyDriver

from derivative_books import BlackScholesPutCallBook
from environments import DerivativeBookHedgeEnv
from policies import BlackScholesDeltaPolicy

# ==============================================================================
# === hyperparameters
num_hedges = 52

# ==============================================================================
# === define book
maturity = 1.
strike = np.array([80, 90, 100])
drift = np.array([0.02, 0.03])
rate = 0.01
diffusion = np.array([[0.15, 0.2, 0.25], [0.2, 0.45, 0.05]])
put_call = np.array([-1, 1, 1])
exposure = np.array([1, 1, -1])
linker = np.array([0, 1, 0])

book = BlackScholesPutCallBook(
    maturity, strike, drift, rate, diffusion, put_call, exposure, linker)

# ==============================================================================
# === define environment
init_state = np.array([85, 95, 1])
env = DerivativeBookHedgeEnv(book, init_state, num_hedges, 0.)
policy = BlackScholesDeltaPolicy(book)

replay_buffer = []
metric = py_metrics.AverageReturnMetric()
observers = [replay_buffer.append, metric]
driver = PyDriver(env, policy, observers, max_episodes=250)

initial_time_step = env.reset()
final_time_step, _ = driver.run(initial_time_step)

print('Average Return: ', metric.result())

# === plot
# xmin, xmax = 80., 110.
# x = tf.linspace(xmin, xmax, 1000)
# y = (x - strike) * tf.cast(x > strike, FLOAT_DTYPE)
# 
# inplot = (xmin < asset) & (asset < xmax)
# 
# plt.figure()
# plt.plot(x, y, color="black")
# plt.scatter(tf.boolean_mask(asset, inplot), tf.boolean_mask(V, inplot),
#             5, alpha=0.5, color="cyan")
# plt.show()
