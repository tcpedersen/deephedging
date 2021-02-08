# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt

from derivative_books import BlackScholesPutCallBook
from environments import DerivativeBookHedgeEnv
from policies import BlackScholesDeltaPolicy
from metrics import CumulativeRewardMetric

# ==============================================================================
# === hyperparameters
num_hedges = 52
batch_size = int(10**6 / 250)
max_episodes = 250

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
