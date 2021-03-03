# -*- coding: utf-8 -*-
import tensorflow as tf
import abc

from utils import norm_cdf, norm_pdf
from constants import FLOAT_DTYPE, INT_DTYPE

class Derivative(abc.ABC):
    @abc.abstractmethod
    def payoff(self, instrument: tf.Tensor, numeraire: tf.Tensor):
        """Computes payoff of derivative in terms of numeraire for each sample
        in batch
        Args:
            instrument: (batch_size, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            payoff: (batch_size, )
        """


    @abc.abstractmethod
    def value(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes price of derivative in terms of numeraire for each sample
        in batch.
        Args:
            time: (time_steps + 1, )
            instrument: (batch_size, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            value: (batch_size, time_steps + 1)
        """


    @abc.abstractmethod
    def delta(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes the delta of the derivative in terms of numeraire for each
        sample in batch.
        Args:
            time: (time_steps + 1, )
            instrument: (batch_size, time_steps + 1)
            numeraire: (time_steps + 1, )
        Returns:
            delta: (batch_size, time_steps + 1)
        """


class PutCall(Derivative):
    def __init__(self, maturity: float, strike: float, rate: float,
                 volatility: float, theta: float):
        self.maturity = maturity
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.rate = float(rate)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)
        self.theta = tf.convert_to_tensor(theta, FLOAT_DTYPE)


    def payoff(self, instrument: tf.Tensor, numeraire: tf.Tensor):
        diff = self.theta * (instrument[..., -1] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)

        return (diff * itm) / numeraire[-1]


    def value(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        ttm = self.maturity - time
        raw_price = black_price(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_price / numeraire


    def delta(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        ttm = self.maturity - time
        raw_delta = black_delta(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_delta / numeraire



class BinaryCall(Derivative):
    def __init__(self, maturity: float, strike: float, volatility: float):
        self.maturity = float(maturity)
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)


    def payoff(self, instrument, numeraire):
        itm = tf.cast(instrument[..., -1] > self.strike, FLOAT_DTYPE)
        return itm / numeraire[-1]


    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time
        forward = instrument * numeraire[-1] / numeraire
        m = tf.math.log(forward / self.strike)
        v = self.volatility * tf.math.sqrt(ttm)

        return norm_cdf(m / v - v / 2.) / numeraire[-1]


    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time
        forward = instrument * numeraire[-1] / numeraire
        m = tf.math.log(forward / self.strike)
        v = self.volatility * tf.math.sqrt(ttm)
        scale = instrument * v
        raw_delta = norm_pdf(m / v - v / 2.) / scale / numeraire[-1]

        return tf.where(tf.equal(v, 0.), 0., raw_delta)


class Barrier(Derivative, abc.ABC):
    def __init__(self,
                 barrier: float,
                 rate: float,
                 volatility: float,
                 outin: float,
                 updown: float):
        self.barrier = tf.convert_to_tensor(barrier, FLOAT_DTYPE)
        self.rate = float(rate)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)

        assert outin == 1 or outin == -1
        assert updown == 1 or updown == -1

        self.outin = float(outin)
        self.updown = float(updown)

        self.p = 2. * self.rate / self.volatility**2 - 1


    @abc.abstractmethod
    def raw_payoff(self, instrument: tf.Tensor, numeraire: tf.Tensor):
        """Returns payoff of European type payoff in terms of numeriare."""


    @abc.abstractmethod
    def up_value(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        """Returns value of truncated up-contract in terms of numeraire."""


    @abc.abstractmethod
    def down_value(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        """Returns value of truncated up-contract in terms of numeraire."""


    @abc.abstractmethod
    def up_delta(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        """Returns value of truncated up-contract in terms of numeraire."""


    @abc.abstractmethod
    def down_delta(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        """Returns value of truncated up-contract in terms of numeraire."""


    def payoff(self, instrument: tf.Tensor, numeraire: tf.Tensor):
        raw_payoff = self.raw_payoff(instrument, numeraire)
        has_crossed = self.crossed(instrument)[..., -1]

        if self.outin == 1:
            return tf.where(has_crossed, 0., raw_payoff)
        else:
            return tf.where(has_crossed, raw_payoff, 0.)


    def value(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        if tf.sign(self.outin * self.updown) == 1:
            standard = self.up_value(time, instrument, numeraire)
        else:
            standard = self.down_value(time, instrument, numeraire)

        if self.updown == 1:
            reflected = self.up_value(
                time, self.barrier**2 / instrument, numeraire)
        else:
            reflected = self.down_value(
                time, self.barrier**2 / instrument, numeraire)

        scale = (self.barrier / instrument)**self.p
        crossed = self.crossed(instrument)

        non_crossed_value = standard - self.outin * scale * reflected

        if self.outin == 1:
            crossed_value = tf.zeros_like(non_crossed_value, FLOAT_DTYPE)
        else:
            crossed_value = self.up_value(time, instrument, numeraire) \
                + self.down_value(time, instrument, numeraire)

        return tf.where(crossed, crossed_value, non_crossed_value)


    def delta(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        if tf.sign(self.outin * self.updown) == 1:
            standard_grad = self.up_delta(time, instrument, numeraire)
        else:
            standard_grad = self.down_delta(time, instrument, numeraire)

        if self.updown == 1:
            reflected = self.up_value(
                time, self.barrier**2 / instrument, numeraire)
            reflected_grad = self.up_delta(
                time, self.barrier**2 / instrument, numeraire)
        else:
            reflected = self.down_value(
                time, self.barrier**2 / instrument, numeraire)
            reflected_grad = self.down_delta(
                time, self.barrier**2 / instrument, numeraire)

        scale = (self.barrier / instrument)**self.p
        crossed = self.crossed(instrument)

        non_crossed_delta = standard_grad + self.outin * scale * (
            self.p / instrument * reflected \
                + (self.barrier / instrument)**2 * reflected_grad)

        if self.outin == 1:
            crossed_delta = tf.zeros_like(non_crossed_delta, FLOAT_DTYPE)
        else:
            crossed_delta = self.up_delta(time, instrument, numeraire) \
                + self.down_delta(time, instrument, numeraire)

        return tf.where(crossed, crossed_delta, non_crossed_delta)


    def crossed(self, instrument: tf.Tensor) -> tf.Tensor:
        """Returns whether the option is crossed."""
        crossed = self.updown * (instrument - self.barrier) > 0

        return tf.cumsum(tf.cast(crossed, INT_DTYPE), 1) > 0


class BarrierCall(Barrier):
    def __init__(self,
                 maturity: float,
                 strike: float,
                 barrier: float,
                 rate: float,
                 volatility: float,
                 outin: float,
                 updown: float):
        self.strike = float(strike)
        super().__init__(barrier, rate, volatility, outin, updown)

        self.barrier_call = PutCall(
            maturity, self.barrier, self.rate, self.volatility, 1)
        self.strike_call = PutCall(
            maturity, self.strike, self.rate, self.volatility, 1)
        self.barrier_binary = BinaryCall(
            maturity, self.barrier, self.volatility)


    def raw_payoff(self, instrument: tf.Tensor, numeraire: tf.Tensor):
        return self.strike_call.payoff(instrument, numeraire)


    def down_value(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        if self.barrier < self.strike:
            return self.strike_call.value(time, instrument, numeraire)
        else:
            return self.barrier_call.value(time, instrument, numeraire) \
                + (self.barrier - self.strike) \
                    * self.barrier_binary.value(time, instrument, numeraire)


    def up_value(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        return self.strike_call.value(time, instrument, numeraire) \
            - self.down_value(time, instrument, numeraire)


    def down_delta(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        if self.barrier < self.strike:
            return self.strike_call.delta(time, instrument, numeraire)
        else:
            return self.barrier_call.delta(time, instrument, numeraire) \
                + (self.barrier - self.strike) \
                    * self.barrier_binary.delta(time, instrument, numeraire)


    def up_delta(self, time: tf.Tensor, instrument: tf.Tensor,
                 numeraire: tf.Tensor):
        return self.strike_call.delta(time, instrument, numeraire) \
            - self.down_delta(time, instrument, numeraire)

# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns price in Black's model.
    Args:
        time_to_maturity: (time_steps + 1, )
        spot: (batch_size, time_steps + 1)
        strike: float
        rate: float
        volatility: float
        theta: float
    Returns:
        price: (batch_size, time_steps + 1)
    """
    deflator = tf.math.exp(-rate * time_to_maturity)
    forward = spot / deflator
    m = tf.math.log(forward / strike)
    v = volatility * tf.math.sqrt(time_to_maturity)
    m_over_v = m / v
    v_over_2 = v / 2.

    value = deflator * theta \
            * (forward * norm_cdf(theta * (m_over_v + v_over_2)) \
               - strike * norm_cdf(theta * (m_over_v - v_over_2)))

    return value


def black_delta(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns delta in Black's model.
    Args:
        see black_price
    Returns:
        delta: (batch_size, time_steps + 1)
    """
    deflator = tf.math.exp(-rate * time_to_maturity)
    forward = spot / deflator
    m = tf.math.log(forward / strike)
    v = volatility * tf.math.sqrt(time_to_maturity)

    raw_delta = theta * norm_cdf(theta * (m / v + v / 2.))
    payoff_delta = theta * tf.cast(theta * (spot - strike) > 0, FLOAT_DTYPE)
    delta = tf.where(tf.equal(v, 0.), payoff_delta, raw_delta)

    return delta

