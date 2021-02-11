# -*- coding: utf-8 -*-
import tensorflow as tf
import abc
import math

from timestep import ActionStep, TimeStep

# ==============================================================================
# === Metrics
class Metric(object):
    @abc.abstractmethod
    def load(self, time_step: TimeStep, action_step: ActionStep):
        """Loads the steps into memory."""

    @abc.abstractmethod
    def result(self):
        """Returns the result of the metric."""

class CumulativeRewardMetric(object):
    def __init__(self):
        self.num_episodes = 0
        self.mean_reward = []

    def load(self, time_step: TimeStep, action_step: ActionStep):
        self.mean_reward.append(tf.reduce_mean(time_step.reward))
        if time_step.terminated:
            self.num_episodes += 1

    def result(self):
        raw = tf.concat(self.mean_reward, 0)
        split = tf.split(raw, self.num_episodes)
        stacked = tf.stack(split)
        summed = tf.cumsum(stacked, 0)

        mean = tf.reduce_mean(summed, 0)
        std = tf.math.reduce_std(summed, 0)

        right_ci = mean + 1.96 * std / math.sqrt(self.num_episodes)
        left_ci = mean - 1.96 * std / math.sqrt(self.num_episodes)

        return mean, std, left_ci, right_ci
