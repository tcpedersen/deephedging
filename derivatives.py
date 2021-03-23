# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc

from utils import norm_cdf, norm_pdf
from constants import FLOAT_DTYPE, INT_DTYPE

class Derivative(abc.ABC):
    @abc.abstractmethod
    def payoff(self, time: tf.Tensor, instrument: tf.Tensor, numeraire: tf.Tensor):
        """Computes payoff of derivative in terms of numeraire for each sample
        in batch
        Args:
            time: (timesteps + 1)
            instrument: (batch_size, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            payoff: (batch_size, )
        """


    @abc.abstractmethod
    def value(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes price of derivative in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instrument: (batch_size, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, timesteps + 1)
        """


    @abc.abstractmethod
    def delta(self, time: tf.Tensor, instrument: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes the delta of the derivative in terms of numeraire for each
        sample in batch.
        Args:
            time: (timesteps + 1, )
            instrument: (batch_size, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            delta: (batch_size, timesteps + 1)
        """


class PutCall(Derivative):
    def __init__(self, maturity: float, strike: float, rate: float,
                 volatility: float, theta: float):
        self.maturity = maturity
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.rate = float(rate)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)
        self.theta = tf.convert_to_tensor(theta, FLOAT_DTYPE)


    def payoff(self, time, instrument, numeraire):
        diff = self.theta * (instrument[..., -1] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)

        return (diff * itm) / numeraire[-1]


    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time
        raw_price = black_price(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_price / numeraire


    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time
        raw_delta = black_delta(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_delta / numeraire


class BinaryCall(Derivative):
    def __init__(self, maturity: float, strike: float, volatility: float):
        self.maturity = float(maturity)
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)


    def payoff(self, time, instrument, numeraire):
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
        self.barrier = float(barrier)
        self.rate = float(rate)
        self.volatility = float(volatility)

        assert outin == 1 or outin == -1
        assert updown == 1 or updown == -1

        self.outin = float(outin)
        self.updown = float(updown)

        self.p = 2. * self.rate / self.volatility**2 - 1


    @abc.abstractmethod
    def raw_payoff(self, time: tf.Tensor, instrument: tf.Tensor,
                   numeraire: tf.Tensor):
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


    def payoff(self, time, instrument, numeraire):
        raw_payoff = self.raw_payoff(time, instrument, numeraire)
        has_crossed = self.crossed(instrument)[..., -1]

        if self.outin == 1:
            return tf.where(has_crossed, 0., raw_payoff)
        else:
            return tf.where(has_crossed, raw_payoff, 0.)


    def value(self, time, instrument, numeraire):
        if self.outin * self.updown == 1:
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


    def delta(self, time, instrument, numeraire):
        if self.outin * self.updown == 1:
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

        if self.outin == 1 and self.updown == 1:
            assert self.barrier > self.strike, "derivative is worthless."

        self.barrier_call = PutCall(
            maturity, self.barrier, self.rate, self.volatility, 1)
        self.strike_call = PutCall(
            maturity, self.strike, self.rate, self.volatility, 1)
        self.barrier_binary = BinaryCall(
            maturity, self.barrier, self.volatility)


    def raw_payoff(self, time: tf.Tensor, instrument: tf.Tensor,
                   numeraire: tf.Tensor):
        return self.strike_call.payoff(time, instrument, numeraire)


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


class DiscreteGeometricAverage(Derivative):
    def __init__(self, maturity, rate, volatility, mtime):
        self.maturity = float(maturity)
        self.rate = float(rate)
        self.volatility = float(volatility)
        self.mtime = tf.convert_to_tensor(mtime, FLOAT_DTYPE)
        self.dt = tf.pad(self.mtime[1:] - self.mtime[:-1], [[1, 0]],
                         constant_values=self.mtime[0])

        padded_mtime = tf.pad(self.mtime, [[1, 0]])
        mu = (self.rate - self.volatility**2 / 2.) * tf.reduce_sum(
            tf.linalg.band_part(
                (mtime[..., tf.newaxis] - padded_mtime[:-1]) \
                    * self.dt[..., tf.newaxis], -1, 0), 0)
        vsq = self.volatility**2 * tf.cumsum(
            tf.square(self.maturity - padded_mtime[:-1]) * self.dt,
            reverse=True)
        self.expxi = tf.math.exp(mu + vsq / 2.)
        self.expxi = tf.pad(self.expxi, [[0, 1]], constant_values=1)


        assert tf.equal(self.maturity, self.mtime[-1]), \
            "maturity must be last monitoring date."
        assert not tf.equal(self.mtime[0], 0.), \
            "time 0 is not an allowed monitoring date."


    def get_time_mask(self, time, with_time_zero=False):
        mask = tf.convert_to_tensor(np.isin(time, self.mtime), tf.bool)
        num_true = tf.reduce_sum(tf.cast(mask, tf.int32))

        if not tf.equal(num_true, tf.size(self.mtime)):
            raise ValueError("time does not contain every element of mtime.")

        if with_time_zero:
            condition = tf.cast(tf.one_hot(0, tf.size(mask)), tf.bool)
            mask = tf.where(condition, True, mask)

        return mask


    def payoff(self, time, instrument, numeraire):
        mask = self.get_time_mask(time)
        v = tf.pow(tf.boolean_mask(instrument, mask, 1), self.dt)

        return tf.reduce_prod(v, -1) / numeraire[-1]


    def mvalue(self, time, instrument, numeraire):
        mask = self.get_time_mask(time, with_time_zero=True)
        masked = tf.boolean_mask(instrument, mask, 1)

        padded_dt = tf.pad(self.dt, [[1, 0]])
        measurable = tf.math.cumprod(tf.pow(masked, padded_dt), -1)
        ttm = tf.pad(self.maturity - self.mtime, [[1, 0]],
                     constant_values=self.maturity)

        return measurable * tf.pow(masked, ttm) / numeraire[-1] * self.expxi


    def _get_split_size(self, mask):
        indices = tf.squeeze(tf.where(mask))
        padded = tf.concat([indices, [tf.size(mask, indices.dtype)]], 0)
        size = padded[1:] - padded[:-1]

        return size

    def value(self, time, instrument, numeraire):
        mask = self.get_time_mask(time, with_time_zero=True)
        size = self._get_split_size(mask)

        time_split = tf.split(time, size)
        instrument_split = tf.split(instrument, size, 1)

        mvalue = self.mvalue(time, instrument, numeraire)
        masked = tf.boolean_mask(instrument, mask, 1)

        value = []

        for k, (t, spot) in enumerate(zip(time_split, instrument_split)):
            tk = self.mtime[k - 1] if k > 0 else 0. # lazy padding with zero
            ratio = tf.pow(spot / masked[..., k, tf.newaxis],
                           self.maturity - tk)

            p = self.rate - self.volatility**2 / 2.
            v = p + self.volatility**2 / 2. * (self.maturity - tk)
            push = tf.exp(-v * (self.maturity - tk) * (t - tk))

            value.append(ratio * mvalue[..., k, tf.newaxis] * push)

        return tf.concat(value, -1)


    def delta(self, time, instrument, numeraire):
        mask = self.get_time_mask(time, with_time_zero=True)
        size = self._get_split_size(mask)
        padded = tf.pad(self.mtime, [[2, 0]])

        time_at_k = tf.concat(
            [tf.ones(s) * padded[k] for k, s in enumerate(size)], 0)
        time_at_t = tf.concat(
            [tf.ones(s) * padded[k + 1] for k, s in enumerate(size)], 0)

        scale = self.maturity - tf.where(mask, time_at_k, time_at_t)

        return self.value(time, instrument, numeraire) * scale / instrument


# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, rate, volatility, theta):
    """Returns price in Black's model.
    Args:
        time_to_maturity: (timesteps + 1, )
        spot: (batch_size, timesteps + 1)
        strike: float
        rate: float
        volatility: float
        theta: float
    Returns:
        price: (batch_size, timesteps + 1)
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
        delta: (batch_size, timesteps + 1)
    """
    deflator = tf.math.exp(-rate * time_to_maturity)
    forward = spot / deflator
    m = tf.math.log(forward / strike)
    v = volatility * tf.math.sqrt(time_to_maturity)

    raw_delta = theta * norm_cdf(theta * (m / v + v / 2.))
    payoff_delta = theta * tf.cast(theta * (spot - strike) > 0, FLOAT_DTYPE)
    delta = tf.where(tf.equal(v, 0.), payoff_delta, raw_delta)

    return delta

