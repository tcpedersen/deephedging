# -*- coding: utf-8 -*-
import tensorflow as tf
from constants import FLOAT_DTYPE

# ==============================================================================
# === TimeStep
class TimeStep(object):
    def __init__(self,
                 observation: tf.Tensor,
                 reward: tf.Tensor,
                 terminated: bool):
        # TODO check if not tensor before converting, otherwise do not convert.
        self.observation = tf.convert_to_tensor(observation, FLOAT_DTYPE)
        self.reward = tf.convert_to_tensor(reward, FLOAT_DTYPE)
        self.terminated = bool(terminated)


def restart(observation):
    batch_size, d = observation.shape
    return TimeStep(observation, tf.zeros(batch_size, FLOAT_DTYPE), False)

def transition(observation, reward):
    return TimeStep(observation, reward, False)

def termination(observation, reward):
    return TimeStep(observation, reward, True)

# ==============================================================================
# === Action
class ActionStep(object):
    def __init__(self, action: tf.Tensor):
        self.action = tf.convert_to_tensor(action, FLOAT_DTYPE)
