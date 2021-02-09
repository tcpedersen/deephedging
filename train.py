# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from collections import deque

from timestep import TimeStep, ActionStep

class ReplayBuffer():
    def __init__ (self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, time_step, action_step, next_time_step):
        observation, action =  time_step.observation, action_step.action,
        reward = next_time_step.reward
        next_observation = next_time_step.observation

        self.buffer.append((observation, action, reward, next_observation))

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        out = []
        for jj in range(4):
            sample = tf.concat([self.buffer[ii][jj] for ii in idx], 0)
            out.append(sample)

        return out
