# -*- coding: utf-8 -*-
import numpy as np

from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.policy_step import PolicyStep

from derivative_books import BlackScholesPutCallBook
from constants import NP_FLOAT_DTYPE

class BlackScholesDeltaPolicy(PyPolicy):
    def __init__(self, book):
        assert isinstance(book, BlackScholesPutCallBook)
        self.book = book

        state_dimension = self.book.get_state_dimension()
        market_size = self.book.get_market_size()
        input_size = 1 + state_dimension + market_size # +1 for time
        input_array_spec = ArraySpec((input_size, ), NP_FLOAT_DTYPE)
        super().__init__(
            time_step_spec=ts.time_step_spec(input_array_spec),
            action_spec=ArraySpec((market_size, ), NP_FLOAT_DTYPE))

    def _action(self, time_step, policy_state):
        market_size = self.book.get_market_size()
        time = time_step.observation[0]
        state = time_step.observation[1:(market_size + 1)][np.newaxis, :]
        deltas = -self.book.marginal_book_delta(state, time)[0, :]
        action = np.bincount(self.book.linker, deltas)
        return PolicyStep(action)