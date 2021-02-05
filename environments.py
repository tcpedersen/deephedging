# -*- coding: utf-8 -*-
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import time_step as ts
from collections import deque

from constants import NP_FLOAT_DTYPE
from market_models import BlackScholes, ConstantRateBankAccount

class BSCallHedgeEnv(py_environment.PyEnvironment):
    def __init__(self, maturity, spot, strike, drift, rate, vol,
                 num_hedges_each_year):
        self._asset_model = BlackScholes(drift, rate, vol)
        self._bank_model = ConstantRateBankAccount(rate)

        self._maturity = float(maturity)
        self._asset_spot = float(spot)
        self._strike = float(strike)
        self._bank_spot = 1.

        self._num_hedges = int(num_hedges_each_year)
        self._num_paths_in_batch = 1000
        self._time_step_size = self._maturity / self._num_hedges

        self._asset_batch = deque()
        self._bank_path = self._bank_model.sample_path(
            self._maturity, self._bank_spot, 1, self._num_hedges)

        self._action_spec = ArraySpec((), NP_FLOAT_DTYPE, "action")
        self._observation_spec = ArraySpec((), NP_FLOAT_DTYPE, "observation")
        self._episode_ended = False

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._time_idx = 0
        if not self._asset_batch:
            self._fill_batch()
        self._asset_path = self._asset_batch.pop()
        self._asset_value = self._asset_path[self._time_idx]
        self._bank_value = self._bank_path[self._time_idx]
        self._book_value = - self._asset_model.call_price(
            self._maturity, self._asset_value, self._strike)
        self._hedge_value = 0.
        self._asset_holdings = 0.
        self._bank_holdings = 0.
        self._episode_ended = False

        return ts.restart(np.array(self._asset_value, NP_FLOAT_DTYPE))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._time_idx += 1
        self._asset_value = self._asset_path[self._time_idx]
        self._bank_value = self._bank_path[self._time_idx]

        self._hedge_value = self._asset_holdings * self._asset_value \
            + self._bank_holdings * self._bank_value
        self._book_value = - self._asset_model.call_price(
            self._maturity - self._time_step_size * self._time_idx,
            self._asset_value,
            self._strike)

        self._asset_holdings = action
        self._bank_holdings = (self._hedge_value - self._asset_holdings \
                               * self._asset_value) / self._bank_value
        reward = (self._book_value - self._hedge_value)**2

        if self._time_idx + 1 == len(self._asset_path):
            self._episode_ended = True
            return ts.termination(self._asset_value, reward)
        else:
            return ts.transition(self._asset_value, reward)

    def _fill_batch(self):
        paths = self._asset_model.sample_path(
            self._maturity,
            self._asset_spot,
            self._num_paths_in_batch,
            self._num_hedges,
            "p")
        self._asset_batch = deque(paths)