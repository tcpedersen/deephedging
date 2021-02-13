# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer():
    def __init__ (self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, time_step, action_step, next_time_step):
        observation, action =  time_step.observation, action_step.action
        reward = next_time_step.reward
        next_observation = next_time_step.observation

        batch_size = observation.shape[0]
        _input = tf.concat((observation, action, reward[:, tf.newaxis], next_observation), 1)

        for val in tf.split(_input, batch_size):
            self.buffer.append(val)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)

        out = []
        for ii in idx:
            out.append(self.buffer[ii])
        out = tf.concat(out, 0)

        return out
