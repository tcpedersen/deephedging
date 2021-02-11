# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from derivative_books import random_black_scholes_put_call_book
from environments import DerivativeBookHedgeEnv
from policies import BlackScholesDeltaPolicy
from metrics import CumulativeRewardMetric

# ==============================================================================
# === hyperparameters
num_hedges = 12
batch_size = 1
max_episodes = 1

# ==============================================================================
# === define book
init_state, book = random_black_scholes_put_call_book(20, 7, 7)

# ==============================================================================
# === define environment
env = DerivativeBookHedgeEnv(book, init_state, num_hedges, 0., batch_size)
policy = BlackScholesDeltaPolicy(book)
metrics = [CumulativeRewardMetric()]

num_episode = 0

while num_episode < max_episodes:
    time_step = env.reset()

    while not time_step.terminated:
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step)

        for metric in metrics:
            metric.load(time_step, action_step)

        time_step = next_time_step
    num_episode += 1

plt.figure()
time = np.linspace(0., book.maturity, len(metric.result()))
plt.plot(time, metric.result())
plt.show()

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
