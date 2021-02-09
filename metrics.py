# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

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
        self.mean_reward = []

    def load(self, time_step: TimeStep, action_step: ActionStep):
        self.mean_reward.append(tf.reduce_mean(time_step.reward))

    def result(self):
        return tf.cumsum(tf.concat(self.mean_reward, 0))
