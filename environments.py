# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import abc

import timestep as ts
from timestep import TimeStep, ActionStep

from collections import deque
from constants import NP_FLOAT_DTYPE
from derivative_books import DerivativeBook

# ==============================================================================
# === Environments
class Environment(abc.ABC):
    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        """Size of batch"""

    @property
    @abc.abstractmethod
    def action_dimension(self) -> int:
        """Dimensionality of action."""

    @property
    @abc.abstractmethod
    def observation_dimension(self) -> int:
        """Dimensionality of observation."""


    def reset(self) -> TimeStep:
        """Wrapper for _reset."""
        return self._reset()

    @abc.abstractmethod
    def _reset(self) -> TimeStep:
        """Resets the environment.
        Args:
            None
        Returns:
            time_step: the initial time step.
        """

    def step(self, action_step: ActionStep) -> TimeStep:
        """Wrapper for _step."""
        return self._step(action_step)

    @abc.abstractmethod
    def _step(self, action_step: ActionStep) -> TimeStep:
        """Performs one step in the environment.
        Args:
            action: ...
        Returns:
            time_step: the subsequent time step.
        """


class DerivativeBookHedgeEnv(Environment):
    def __init__(self, book, init_state, num_hedges, cost_scale, batch_size):
        """Initialise DerivativeBookHedgeEnv
        Args:
            book: DerivativeBook
            init_state: np.ndarray (state_dimension, )
            num_hedges: int
            cost_scale: float
            batch_size: int
        Returns:
            None
        """
        assert issubclass(type(book), DerivativeBook)

        self.book = book
        self.init_state = np.array(init_state, NP_FLOAT_DTYPE, ndmin=1)
        self.num_hedges = int(num_hedges)
        self.cost_scale = float(cost_scale)
        self._batch_size = int(batch_size)

        self.time_step_size = self.book.maturity / self.num_hedges

        self.max_paths_in_memory = 10000
        if self.batch_size > self.max_paths_in_memory:
            self.max_paths_in_memory = self.batch_size
        self.size_batch_of_paths = math.floor(
            self.max_paths_in_memory / self.batch_size)
        self.batch_of_paths = deque()

    @property
    def batch_size(self) -> int:
        return self._batch_size


    @property
    def action_dimension(self) -> int:
        return self.book.market_size


    @property
    def observation_dimension(self) -> int:
        return 1 + self.book.state_dimension + self.book.market_size


    def get_time(self):
        return self.time_step_size * self.time_idx


    def get_full_state(self):
        return self.paths[:, :, self.time_idx]


    def get_market_state(self):
        return self.paths[:, :(self.book.market_size + 1), self.time_idx]


    def _reset(self):
        self._episode_ended = False
        self.time_idx = 0

        if not self.batch_of_paths:
            self.fill_batch_of_paths()
        self.paths = self.batch_of_paths.pop()

        self.book_value = self.book.book_value(
            self.get_full_state(), self.get_time())

        empty_hedge = np.zeros((self.batch_size, self.book.market_size))
        time = np.tile(self.get_time(), (self.batch_size, 1))
        observation = np.hstack([time, self.get_full_state(), empty_hedge])

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


    def _step(self, action_step: ActionStep) -> TimeStep:
        if self._episode_ended:
            raise ValueError("episode has ended.")

        # action chosen at time t
        if self.time_idx == 0:
            chg_hedge = self.book.setup_hedge(
                self.get_full_state(), action_step.action, -self.book_value)
            self.hedge_value = self.book.hedge_value(self.get_full_state())
        else:
            chg_hedge = self.book.rebalance_hedge(
                self.get_full_state(), action_step.action)

        # calculate costs
        if self.cost_scale > 0:
            transaction_cost = self.cost_scale \
                * np.sum(self.get_market_state() * abs(chg_hedge), axis=1)
        else:
            transaction_cost = 0

        # move to time t + 1
        self.time_idx += 1
        chg_book_value = self.step_book_value()
        chg_hedge_value = self.step_hedge_value()

        # liquidate hedge portfolio if episode is over
        if self.time_idx == self.num_hedges and self.cost_scale > 0:
            transaction_cost += self.cost_scale \
                * np.sum(self.get_market_state() * abs(self.book.hedge), axis=1)

        reward = chg_book_value + chg_hedge_value - transaction_cost

        time = np.tile(self.get_time(), (self.batch_size, 1))
        observation = tf.concat(
            [time, self.get_full_state(), self.book.hedge[:, :-1]], 1)

        if self.time_idx == self.num_hedges:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward)


    def fill_batch_of_paths(self):
        num_paths = self.size_batch_of_paths * self.batch_size
        paths = self.book.sample_paths(
            self.init_state, num_paths, self.num_hedges, False)

        self.batch_of_paths = deque(np.split(paths, self.size_batch_of_paths))