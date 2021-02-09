# -*- coding: utf-8 -*-
import abc

import timestep as ts
from timestep import ActionStep
from derivative_books import BlackScholesPutCallBook

# ==============================================================================
# === Policy
class Policy(abc.ABC):
    @abc.abstractmethod
    def _action(self, time_step: ts.TimeStep) -> ActionStep:
        """Returns action following the time_step."""

    def action(self, time_step: ts.TimeStep) -> ActionStep:
        return self._action(time_step)

class BlackScholesDeltaPolicy(Policy):
    def __init__(self, book):
        assert isinstance(book, BlackScholesPutCallBook)
        self.book = book

    def _action(self, time_step):
        market_size = self.book.market_size
        time = time_step.observation[0, 0]
        state = time_step.observation[:, 1:(market_size + 1)]
        action = -self.book.book_delta(state, time)
        return ActionStep(action)
