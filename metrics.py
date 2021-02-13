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

class TrainMetric(Metric):
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.mean_reward = []

    def load(self, time_step: TimeStep, action_step: ActionStep):
        self.mean_reward.append(tf.reduce_mean(time_step.reward))

    def result(self):
        raw = tf.concat(self.mean_reward, 0)
        split = tf.split(raw, self.num_episodes)
        stacked = tf.stack(split)
        summed = tf.reduce_sum(stacked, 0)

        mean = tf.reduce_mean(summed, 0)
        std = tf.math.reduce_std(summed, 0)

        right_ci = mean + 1.96 * std / math.sqrt(self.num_episodes)
        left_ci = mean - 1.96 * std / math.sqrt(self.num_episodes)

        return mean, std, left_ci, right_ci


class TestMetric(Metric):
    def __init__(self):
        self.reward = []

    def load(self, time_step: TimeStep, action_step: ActionStep):
        self.reward.append(time_step.reward)

    def result(self):
        stacked = tf.stack(self.reward, 1)
        summed = tf.cumsum(stacked, 0)

        mean = tf.reduce_mean(summed, 0)
        std = tf.math.reduce_std(summed, 0)

        num_samples = stacked.shape[0]
        right_ci = mean + 1.96 * std / math.sqrt(num_samples)
        left_ci = mean - 1.96 * std / math.sqrt(num_samples)

        return mean, std, left_ci, right_ci