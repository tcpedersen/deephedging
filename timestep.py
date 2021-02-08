# -*- coding: utf-8 -*-
import numpy as np
from constants import NP_FLOAT_DTYPE

# ==============================================================================
# === TimeStep
class TimeStep(object):
    def __init__(self,
                 observation: np.ndarray,
                 reward: np.ndarray,
                 terminated: bool):
        self.observation = np.array(observation, NP_FLOAT_DTYPE, ndmin=2)
        self.reward = np.array(reward, NP_FLOAT_DTYPE, ndmin=1)
        self.terminated = bool(terminated)


def restart(observation):
    batch_size, d = observation.shape
    return TimeStep(observation, np.zeros(batch_size, NP_FLOAT_DTYPE), False)

def transition(observation, reward):
    return TimeStep(observation, reward, False)

def termination(observation, reward):
    return TimeStep(observation, reward, True)

# ==============================================================================
# === Action
class ActionStep(object):
    def __init__(self, action):
        self.action = np.array(action, NP_FLOAT_DTYPE, ndmin=2)
