# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import time_step as ts

from constants import NP_FLOAT_DTYPE
from derivative_books import DerivativeBook

class DerivativeBookHedgeEnv(PyEnvironment):
    def __init__(self, book, init_state, num_hedges, transaction_cost_scale):
        """Initialise DerivativeBookHedgeEnv
        Args:
            book: DerivativeBook
            init_state: np.ndarray (market_size, state_dimension)
            num_hedges: int
            transaction_cost_scale: float
        Returns:
            None
        """
        assert issubclass(type(book), DerivativeBook)

        self.book = book
        self.init_state = np.array(init_state, NP_FLOAT_DTYPE)
        self.num_hedges = int(num_hedges)
        self.transaction_cost_scale = float(transaction_cost_scale)

        self.state_dimension = self.book.get_state_dimension()

        self.num_paths_in_batch = 100
        self.time_step_size = self.book.maturity / self.num_hedges

        self.batch = deque()

        self.action_spec = ArraySpec(
            (self.book.get_market_size(), ), NP_FLOAT_DTYPE, "action")
        self.observation_spec = ArraySpec(
            (self.state_dimension, ), NP_FLOAT_DTYPE, "observation")
        self.episode_ended = False

    def observation_spec(self):
        return self.observation_spec

    def action_spec(self):
        return self.action_spec

    def _reset(self):
        self.time_idx = 0
        if not self.batch:
            self.fill_batch()

        self.book_value = self.book.book_value(
            self.init_state[np.newaxis, :], 0.)

        self.hedge = np.zeros(self.state_dimension, NP_FLOAT_DTYPE)
        self.hedge_value = -self.book_value

        self.path = self.batch.pop()
        self.observation = np.hstack([
            0, self.path[:, self.time_idx], self.hedge])
        self._episode_ended = False

        return ts.restart(self.observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # action chosen at time t
        time = self.time_step_size * self.time_idx
        market_state = self.path[True, :, self.time_idx]
        tradable = market_state[0, :(self.book.get_market_size() + 1)]

        # rebalance portfolio
        prior_hedge = self.hedge
        self.hedge_value = tradable @ self.hedge
        self.hedge[:self.book.get_market_size()] = action
        self.hedge[-1] = (self.hedge_value - tradable[:-1] @ self.hedge[:-1]) \
            / tradable[-1]

        # calculate costs
        chg_hedge = abs(self.hedge - prior_hedge)
        transaction_cost = self.transaction_cost_scale * tradable @ chg_hedge

        # move to time t + 1
        self.time_idx += 1
        time = self.time_step_size * self.time_idx
        market_state = self.path[True, :, self.time_idx]
        tradable = market_state[0, :(self.book.get_market_size() + 1)]

        prior_book_value = self.book_value
        self.book_value = self.book.book_value(market_state, time)
        chg_book_value = self.book_value - prior_book_value

        prior_hedge_value = self.hedge_value
        self.hedge_value = tradable @ self.hedge
        chg_hedge_value = self.hedge_value - prior_hedge_value

        # liquidate hedge portfolio if episode is over
        if self.time_idx == self.num_hedges:
            transaction_cost += \
                self.transaction_cost_scale * tradable @ abs(self.hedge)

        reward = float(chg_book_value + chg_hedge_value - transaction_cost)

        self.observation = np.hstack(
            [time, self.path[:, self.time_idx], self.hedge])

        if self.time_idx == self.num_hedges:
            self._episode_ended = True
            return ts.termination(self.observation, reward)
        else:
            return ts.transition(self.observation, reward)

    def fill_batch(self):
        paths = self.book.sample_paths(
            self.init_state, self.num_paths_in_batch, self.num_hedges, False)
        self.batch = deque(paths)