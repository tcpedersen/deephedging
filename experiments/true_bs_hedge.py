# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from derivative_books import random_black_scholes_put_call_book
from environments import DerivativeBookHedgeEnv
from policies import BlackScholesDeltaPolicy
from metrics import TrainMetric

# ==============================================================================
# === hyperparameters
batch_size = 250
max_episodes = 100
num_hedges_a_year = 52


# ==============================================================================
# === define book
# seed 23: long call
# seed 27: short call
init_state, book = random_black_scholes_put_call_book(1, 1, 1, 420)
num_hedges = math.ceil(num_hedges_a_year * book.maturity)

# ==============================================================================
# === define environment
env = DerivativeBookHedgeEnv(book, init_state, num_hedges, 0., batch_size)
policy = BlackScholesDeltaPolicy(book)
metrics = [TrainMetric(max_episodes)]

episode = 0

while episode < max_episodes:
    time_step = env.reset()
    cumulative_reward = 0

    while not time_step.terminated:
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step)

        for metric in metrics:
            metric.load(time_step, action_step)
        cumulative_reward += tf.reduce_mean(time_step.reward).numpy()

        time_step = next_time_step

    print(f"episode {episode + 1}".ljust(15) \
          + f"average reward: {cumulative_reward: .3f}")
    episode += 1

# mean, std, left_ci, right_ci = metric.result()


# plt.figure()
# time = np.linspace(0., book.maturity, num_hedges)
# plt.plot(time, mean)
# plt.plot(time, left_ci, "--", color="red")
# plt.plot(time, right_ci, "--", color="red")
# plt.show()
