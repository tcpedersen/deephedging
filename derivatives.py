# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import abc
import math

import utils
from utils import norm_cdf, norm_pdf
from constants import FLOAT_DTYPE, INT_DTYPE

class Derivative(abc.ABC):
    @abc.abstractmethod
    def payoff(self, time: tf.Tensor, instrument: tf.Tensor,
               numeraire: tf.Tensor):
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
    def adjoint(self, time: tf.Tensor, instrument: tf.Tensor,
                numeraire: tf.Tensor):
        """Computes the derivative of the payoff wrt. to the instrument for each
        sample in batch.
        Args:
            time: (timesteps + 1)
            instrument: (batch_size, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            payoff: (batch_size, timesteps + 1)
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

        if tf.size(self.volatility) != 1:
            raise ValueError("volatility is not of dimension 1, "
                             f"but {tf.size(self.volatility)}.")


    def payoff(self, time, instrument, numeraire):
        diff = self.theta * (instrument[..., -1] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)

        return (diff * itm) / numeraire[-1]


    def adjoint(self, time, instrument, numeraire):
        diff = self.theta * (instrument[..., -1, tf.newaxis] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)
        chain = instrument[..., -1, tf.newaxis] / instrument

        return self.theta * itm * chain / numeraire[-1]


    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time
        raw_price = black_price(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_price / numeraire[-1]


    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time
        raw_delta = black_delta(ttm, instrument, self.strike,
                                self.rate, self.volatility, self.theta)

        return raw_delta / numeraire[-1]


class BinaryCall(Derivative):
    def __init__(self, maturity: float, strike: float, volatility: float):
        self.maturity = float(maturity)
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)


    def payoff(self, time, instrument, numeraire):
        itm = tf.cast(instrument[..., -1] > self.strike, FLOAT_DTYPE)
        return itm / numeraire[-1]


    def adjoint(self, time, instrument, numeraire):
        return tf.zeros_like(instrument)


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


        if isinstance(outin, str):
            if outin == "out":
                outin = 1
            elif outin == "in":
                outin = -1
        if isinstance(updown, str):
            if updown == "up":
                updown = 1
            elif updown == "down":
                updown = -1

        if outin not in [-1, 1]:
            raise ValueError(f"outin must be -1 or 1, not {outin}.")
        if updown not in [-1, 1]:
            raise ValueError(f"updown must be -1 or 1, not {updown}.")

        self.outin = float(outin)
        self.updown = float(updown)

        self.p = 2. * self.rate / self.volatility**2 - 1


    @abc.abstractmethod
    def raw_payoff(self, time: tf.Tensor, instrument: tf.Tensor,
                   numeraire: tf.Tensor):
        """Returns payoff of European type payoff in terms of numeriare."""


    def raw_adjoint(self, time: tf.Tensor, instrument: tf.Tensor,
                    numeraire: tf.Tensor):
        """Returns adjoint of European type payoff in terms of numeriare."""


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
        crossed = self.crossed(instrument)[..., -1]

        if self.outin == 1:
            return tf.where(crossed, 0., raw_payoff)
        else:
            return tf.where(crossed, raw_payoff, 0.)


    def adjoint(self, time, instrument, numeraire):
        raw_adjoint = self.raw_adjoint(time, instrument, numeraire)
        crossed = self.crossed(instrument)[..., -1, tf.newaxis]

        if self.outin == 1:
            return tf.where(crossed, 0., raw_adjoint)
        else:
            return tf.where(crossed, raw_adjoint, 0.)


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
        self.maturity = float(maturity)
        self.strike = float(strike)
        super().__init__(barrier, rate, volatility, outin, updown)

        if self.outin == 1:
            if self.updown == 1 and self.barrier < self.strike:
                raise ValueError("derivative is worthless.")
            elif self.updown == -1 and self.barrier > self.strike:
                raise ValueError("payoff is linear.")
        else:
            if self.updown == 1 and self.strike > self.barrier:
                raise ValueError("derivative is a call option.")

        self.barrier_call = PutCall(
            maturity, self.barrier, self.rate, self.volatility, 1)
        self.strike_call = PutCall(
            maturity, self.strike, self.rate, self.volatility, 1)
        self.barrier_binary = BinaryCall(
            maturity, self.barrier, self.volatility)


    def raw_payoff(self, time: tf.Tensor, instrument: tf.Tensor,
                   numeraire: tf.Tensor):
        return self.strike_call.payoff(time, instrument, numeraire)


    def raw_adjoint(self, time: tf.Tensor, instrument: tf.Tensor,
                   numeraire: tf.Tensor):
        return self.strike_call.adjoint(time, instrument, numeraire)


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


class DiscreteGeometricPutCall(Derivative):
    def __init__(self, maturity: float, strike: float, rate: float,
                 volatility: float, theta: float):
        self.maturity = maturity
        self.strike = tf.convert_to_tensor(strike, FLOAT_DTYPE)
        self.rate = float(rate)
        self.volatility = tf.convert_to_tensor(volatility, FLOAT_DTYPE)
        self.theta = tf.convert_to_tensor(theta, FLOAT_DTYPE)

        self.putcall = PutCall(
            maturity=self.maturity,
            strike=self.strike,
            rate=self.rate,
            volatility=self.volatility,
            theta=self.theta
            )


    def _increments(self, time, pad):
        return tf.pad(time[1:] - time[:-1], [[pad, 0]])


    def _mean(self, time):
        dt = self._increments(time, 0)
        scale = self.rate - self.volatility**2 / 2.
        mat = (time[1:, tf.newaxis] - time[:-1]) * dt[..., tf.newaxis]
        unpadded = scale * tf.reduce_sum(tf.linalg.band_part(mat, -1, 0), 0)

        return tf.pad(unpadded, [[0, 1]])


    def _variance(self, time):
        dt = self._increments(time, 0)
        sqdiff = tf.square(self.maturity - time[:-1])
        unpadded = self.volatility**2 * tf.cumsum(sqdiff * dt, reverse=True)

        return tf.pad(unpadded, [[0, 1]])


    def _derivatives(self, time):
        mean, variance = self._mean(time), self._variance(time)
        derivatives = []
        for k in tf.range(tf.size(time) - 1): # -1 as otherwise divide by zero
            ttm = self.maturity - time[k]
            option = PutCall(
                maturity=self.maturity,
                strike=self.strike,
                rate=(mean[k] + variance[k] / 2.) / ttm,
                volatility=tf.sqrt(variance[k] / ttm),
                theta=self.theta
            )

            derivatives.append(option)

        return derivatives


    def _dga(self, time, instrument):
        dt = self._increments(time, 1)

        return tf.math.cumprod(tf.pow(instrument, dt), -1)

    def _apply(self, time, instrument, numeraire, method):
        dga = self._dga(time, instrument)
        spot = dga * tf.pow(instrument, self.maturity - time)
        output = []

        for k, option in enumerate(self._derivatives(time) + [self.putcall]):
            indices = [0, k, tf.size(time) - 1]
            marginal = getattr(option, method)(
                time=tf.gather(time, indices),
                instrument=tf.gather(spot, indices, axis=-1),
                numeraire=tf.gather(numeraire, indices, axis=-1)
                )
            output.append(marginal[..., 1])

        return tf.stack(output, -1)


    def payoff(self, time, instrument, numeraire):
        return self.putcall.payoff(time, self._dga(time, instrument), numeraire)


    def adjoint(self, time, instrument, numeraire):
        terminal_dga = self._dga(time, instrument)[..., -1, tf.newaxis]
        itm = tf.cast(terminal_dga > self.strike, FLOAT_DTYPE)

        dt = self._increments(time, 1)
        scale = (self.maturity - time + dt) / instrument

        return itm * terminal_dga * scale / numeraire[-1]


    def value(self, time, instrument, numeraire):
        return self._apply(time, instrument, numeraire, "value")


    def delta(self, time, instrument, numeraire):
        unscaled = self._apply(time, instrument, numeraire, "delta")

        ttm = self.maturity - time
        dt = self._increments(time, 1)
        dga = self._dga(time, instrument)
        scale = (self.maturity - time + dt) / instrument

        return unscaled * dga * tf.pow(instrument, ttm) * scale


class JumpPutCall(Derivative):
    def __init__(self, maturity, strike, rate, volatility,
                 intensity, jumpsize, jumpvol, theta):
        if theta not in [-1, 1]:
            raise ValueError(f"theta = {theta} which is not in {-1, 1}.")

        self.maturity = maturity
        self.strike = strike
        self.rate = rate
        self.volatility = volatility
        self.jumpsize = jumpsize
        self.jumpvol = jumpvol
        self.theta = theta
        self.intensity = intensity

        self.kappa = tf.exp(self.jumpsize + self.jumpvol**2 / 2.0) - 1.0

        self.maxiter = 6


    def payoff(self, time, instrument, numeraire):
        option = PutCall(self.maturity, self.strike, self.rate, self.volatility,
                         self.theta)

        return option.payoff(time, instrument, numeraire)


    def adjoint(self, time, instrument, numeraire):
        option = PutCall(self.maturity, self.strike, self.rate, self.volatility,
                         self.theta)

        return option.adjoint(time, instrument, numeraire)


    def mertonsum(self, time, instrument, numeraire, func):
        # HACK set last value to one to avoid overflow in nrate and nvol.
        # zero ttm has no influence on price nor delta.
        ttm = self.maturity - time
        adjttm = tf.where(tf.equal(ttm, 0.), 1., ttm)

        value = tf.zeros_like(instrument)

        volsq = self.volatility * self.volatility
        jumpvolsq = self.jumpvol * self.jumpvol

        m = self.jumpsize + jumpvolsq / 2.0
        jumpcomp = self.intensity * (math.exp(m) - 1.)

        nfac = [math.factorial(n) for n in range(self.maxiter)]
        expterm = tf.math.exp(-self.intensity * ttm)

        for n in tf.range(self.maxiter):
            nfloat = tf.cast(n, FLOAT_DTYPE)
            nvol = tf.sqrt(volsq + nfloat * jumpvolsq / adjttm)
            nrate = self.rate + m * nfloat / adjttm - jumpcomp
            prob = expterm * tf.pow(self.intensity * ttm, nfloat) / nfac[n]
            raw_value = func(ttm, instrument, self.strike, nrate, nvol,
                             self.theta)
            value += prob * raw_value

        return value / numeraire[-1]


    def value(self, time, instrument, numeraire):
        return self.mertonsum(time, instrument, numeraire, black_price)


    def delta(self, time, instrument, numeraire):
        return self.mertonsum(time, instrument, numeraire, black_delta)


class BachelierCall(PutCall):
    def __init__(self, maturity, strike, volatility):
        super().__init__(maturity, strike, 0.0, volatility, 1.0)


    def adjoint(self, time, instrument, numeraire):
        diff = self.theta * (instrument[..., -1, tf.newaxis] - self.strike)
        itm = tf.cast(diff > 0, FLOAT_DTYPE)

        return self.theta * itm * tf.ones_like(instrument) / numeraire[-1]


    def value(self, time, instrument, numeraire):
        ttm = self.maturity - time

        return bachelier_price(ttm, instrument, self.strike, self.volatility)


    def delta(self, time, instrument, numeraire):
        ttm = self.maturity - time

        return bachelier_delta(ttm, instrument, self.strike, self.volatility)


class BachelierBasketCall(Derivative):
    def __init__(self, maturity, strike, diffusion, weights):
        if len(tf.shape(diffusion)) != 2:
            raise TypeError("diffusion of wrong dimension.")
        self.weights = weights / tf.reduce_sum(weights)

        self.basketsize = tf.shape(diffusion)[0]
        basketvol = tf.sqrt(self.weights[tf.newaxis, :] @ diffusion \
            @ tf.transpose(diffusion) @ self.weights[:, tf.newaxis])
        self.option = BachelierCall(maturity, strike, basketvol)


    def basketvalue(self, time, instrument, numeraire):
        if len(tf.shape(instrument)) != 3:
            raise ValueError("instrument is of wrong dimension.")
        weighted = self.weights[tf.newaxis, :, tf.newaxis] * instrument

        return tf.reduce_sum(weighted, 1)


    def payoff(self, time, instrument, numeraire):
        basketvalue = self.basketvalue(time, instrument, numeraire)

        return self.option.payoff(time, basketvalue, numeraire)


    def adjoint(self, time, instrument, numeraire):
        basketvalue = self.basketvalue(time, instrument, numeraire)
        chain = self.option.adjoint(time, basketvalue, numeraire)

        return chain[:, tf.newaxis, :] * self.weights[tf.newaxis, :, tf.newaxis]


    def value(self, time, instrument, numeraire):
        basketvalue = self.basketvalue(time, instrument, numeraire)

        return self.option.value(time, basketvalue, numeraire)


    def delta(self, time, instrument, numeraire):
        basketvalue = self.basketvalue(time, instrument, numeraire)
        chain = self.option.delta(time, basketvalue, numeraire)

        return chain[:, tf.newaxis, :] * self.weights[tf.newaxis, :, tf.newaxis]

# ==============================================================================
# === Black Scholes
def black_price(time_to_maturity, spot, strike, drift, volatility, theta):
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
    forward = spot * tf.math.exp(drift * time_to_maturity)
    m = tf.math.log(forward / strike)
    v = volatility * tf.math.sqrt(time_to_maturity)
    m_over_v = tf.where(tf.equal(m, 0.0), 0.0, m / v)
    v_over_2 = v / 2.0

    p1 = utils.norm_cdf(theta * (m_over_v + v_over_2), approx=True)
    p2 = utils.norm_cdf(theta * (m_over_v - v_over_2), approx=True)

    return theta * (forward * p1 - strike * p2)


def black_delta(time_to_maturity, spot, strike, drift, volatility, theta):
    """Returns delta in Black's model.
    Args:
        see black_price
    Returns:
        delta: (batch_size, timesteps + 1)
    """
    inflator = tf.math.exp(drift * time_to_maturity)
    forward = spot * inflator
    m = tf.math.log(forward / strike)
    v = volatility * tf.math.sqrt(time_to_maturity)

    p1 = utils.norm_cdf(theta * (m / v + v / 2.0), approx=True)

    raw_delta = theta * inflator * p1
    payoff_delta = theta * tf.cast(theta * (spot - strike) > 0, FLOAT_DTYPE)
    delta = tf.where(tf.equal(v, 0.0), payoff_delta, raw_delta)

    return delta


def bachelier_price(time_to_maturity, spot, strike, volatility):
    """Returns price in Bachelier's model.
    Args:
        see black_price
    Returns:
        price: (batch_size, timesteps + 1)
    """
    v = volatility * tf.math.sqrt(time_to_maturity)
    d = (spot - strike) / v
    price = v * (d * utils.norm_cdf(d, approx=True) + utils.norm_pdf(d))

    # due to no interest rate, v=0 implies S_T=S_t a.s.
    return tf.where(tf.equal(v, 0.0), tf.maximum(spot - strike, 0.0), price)


def bachelier_delta(time_to_maturity, spot, strike, volatility):
    """Returns delta in Bachelier's model.
    Args:
        see black_price
    Returns:
        delta: (batch_size, timesteps + 1)
    """
    v = volatility * tf.math.sqrt(time_to_maturity)
    d = (spot - strike) / v

    return utils.norm_cdf(d, approx=True)


# ==============================================================================
# === jumps
def jumpriskratios(tradebook, time, instruments, numeraire):
    # determine distribution of jumps
    jumpvol = tradebook.instrument_simulator.jumpvol
    jumpsize = tradebook.instrument_simulator.jumpsize
    sobol_dim = tradebook.instrument_dim - 1
    sobol = tf.math.sobol_sample(1, sobol_dim)
    grid = tf.sort(tf.squeeze(sobol)) if sobol_dim > 1 \
        else tf.squeeze(sobol)[tf.newaxis]
    jumps = tf.math.exp(utils.norm_qdf(grid) * jumpvol + jumpsize)

    # values before jumps
    hedgevalue = tradebook.value(
        time, instruments, numeraire)[:, tf.newaxis, tf.newaxis, :]
    tradevalue = instruments[:, 1:, :][:, tf.newaxis, ...]

    delta = tradebook.delta(time, instruments, numeraire)[:, tf.newaxis, ...]
    hedgedelta = delta[..., 0, tf.newaxis, :]
    tradedelta = delta[..., 1:, :]

    jumpinstruments = tf.concat(
        tf.split(
            instruments[:, 0, tf.newaxis, :] * jumps[tf.newaxis, :, tf.newaxis],
            len(jumps),
            axis=1),
        axis=0)
    jumphedgevalue = tf.stack(
        tf.split(
            tradebook.hedgebook.value(time, jumpinstruments, numeraire),
            len(jumps)
            ),
        axis=1)[..., tf.newaxis, :]
    jumptradevalue = tf.stack(
        tf.split(
            tradebook.link_apply(
                "value", time, jumpinstruments, numeraire), len(jumps)),
        axis=1)

    dhedgevalue = jumphedgevalue - hedgevalue
    dtradevalue = jumptradevalue - tradevalue
    dinstruments = tf.stack(tf.split(jumpinstruments, len(jumps)), 1) \
        - instruments[:, 0, tf.newaxis, :][..., tf.newaxis, :]

    hedgereturn = dhedgevalue / dinstruments - hedgedelta
    tradereturn = dtradevalue / dinstruments - tradedelta

    unstable = tf.abs(tradereturn) < 0.05
    tradereturn = tf.where(unstable, 0.0, tradereturn)

    traderatios = []
    instrumentratios = []

    for step in tf.range(len(time) - 1):
        inv = tf.linalg.pinv(tradereturn[..., step])
        ratio = inv @ hedgereturn[..., step]
        e = hedgedelta[..., step] - tradedelta[..., step] @ ratio
        traderatios.append(ratio)
        instrumentratios.append(e)

    traderatios = tf.concat(traderatios, -1)
    instrumentratios = tf.concat(instrumentratios, -1)

    return tf.concat([instrumentratios, traderatios], axis=1)
