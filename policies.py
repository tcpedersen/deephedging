# -*- coding: utf-8 -*-
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.policy_step import PolicyStep


from market_models import BlackScholes
from constants import NP_FLOAT_DTYPE

class BlackScholesDeltaPolicy(PyPolicy):
    def __init__(self, strike, drift, rate, vol):
        self._asset_model = BlackScholes(drift, rate, vol)
        self._strike = float(strike)

        input_array_spec = ArraySpec((2, ), NP_FLOAT_DTYPE)
        super().__init__(
            time_step_spec=ts.time_step_spec(input_array_spec),
            action_spec=ArraySpec((), NP_FLOAT_DTYPE))

    def _action(self, time_step, policy_state):
        maturity, spot = time_step.observation
        action = self._asset_model.call_delta(maturity, spot, self._strike)
        return PolicyStep(action)