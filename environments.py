# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import time_step as ts

from constants import NP_FLOAT_DTYPE
from derivative_books import DerivativeBook

class DerivativeBookHedgeEnv(PyEnvironment):
    def __init__(self, book, init_state, num_hedges, cost_scale):
        """Initialise DerivativeBookHedgeEnv
        Args:
            book: DerivativeBook
            init_state: np.ndarray (state_dimension, )
            num_hedges: int
            cost_scale: float
        Returns:
            None
        """
        assert issubclass(type(book), DerivativeBook)

        self.book = book
        self.init_state = np.array(init_state, NP_FLOAT_DTYPE, ndmin=1)
        self.num_hedges = int(num_hedges)
        self.cost_scale = float(cost_scale)

        self.num_paths_in_batch = 100
        self.time_step_size = self.book.maturity / self.num_hedges

        self.batch = deque()


    def observation_spec(self):
        d = self.book.get_state_dimension() + self.book.get_market_size()
        return ArraySpec((d, ), NP_FLOAT_DTYPE)


    def action_spec(self):
        return ArraySpec(self.book.get_market_size(), NP_FLOAT_DTYPE)

    def get_time(self):
        return self.time_step_size * self.time_idx

    def get_full_state(self):
        return self.path[:, self.time_idx]

    def get_market_state(self):
        return self.path[:(self.book.get_market_size() + 1), self.time_idx]

    def _reset(self):
        self._episode_ended = False
        self.time_idx = 0

        if not self.batch:
            self.fill_batch()
        self.path = self.batch.pop()

        self.book_value = self.book.book_value(
            self.get_full_state(), self.get_time())
        empty_hedge = np.zeros(self.book.get_market_size())
        observation = np.hstack(
            [self.get_time(), self.get_full_state(), empty_hedge])

        return ts.restart(observation)

    def step_book_value(self):
        prior = self.book_value
        self.book_value = self.book.book_value(
            self.get_full_state(), self.get_time())

        return self.book_value - prior


    def step_hedge_value(self):
        prior = self.hedge_value
        self.hedge_value = self.book.hedge_value(self.get_full_state())

        return self.hedge_value - prior

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # action chosen at time t
        if self.time_idx == 0:
            chg_hedge = self.book.setup_hedge(
                self.get_full_state(), action, -self.book_value)
            self.hedge_value = self.book.hedge_value(self.get_full_state())
        else:
            chg_hedge = self.book.rebalance_hedge(self.get_full_state(), action)

        # calculate costs
        transaction_cost = self.cost_scale \
            * self.get_market_state() @ abs(chg_hedge)

        # move to time t + 1
        self.time_idx += 1
        chg_book_value = self.step_book_value()
        chg_hedge_value = self.step_hedge_value()

        # liquidate hedge portfolio if episode is over
        if self.time_idx == self.num_hedges:
            transaction_cost += \
                self.cost_scale * self.get_market_state() @ abs(self.book.hedge)

        reward = float(chg_book_value + chg_hedge_value - transaction_cost)

        observation = np.hstack(
            [self.get_time(), self.get_full_state(), self.book.hedge])

        if self.time_idx == self.num_hedges:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward)

    def fill_batch(self):
        paths = self.book.sample_paths(
            self.init_state, self.num_paths_in_batch, self.num_hedges, False)
        self.batch = deque(paths)