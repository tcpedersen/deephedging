# -*- coding: utf-8 -*-
import numpy as np
import abc

from timestep import ActionStep, TimeStep
from constants import NP_FLOAT_DTYPE

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
        self.mean_reward.append(time_step.reward.mean())

    def result(self):
        return np.cumsum(np.array(self.mean_reward, NP_FLOAT_DTYPE))
