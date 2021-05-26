# -*- coding: utf-8 -*-
import tensorflow as tf

import derivatives
import simulators

from constants import FLOAT_DTYPE, INT_DTYPE

# ==============================================================================
# === Base
class DerivativeBook(object):
    def __init__(self,
                 maturity: float,
                 instrument_simulator: simulators.Simulator,
                 numeraire_simulator: simulators.Simulator):
        self.maturity = float(maturity)
        for simulator in [instrument_simulator, numeraire_simulator]:
            if not issubclass(type(simulator), simulators.Simulator):
                raise TypeError("simulator must be of type Simulator.")

        self.instrument_simulator = instrument_simulator
        self.numeraire_simulator = numeraire_simulator

        self.derivatives = []


    @property
    def book_size(self) -> int:
        """Number of derivatives in the book."""
        return len(self.derivatives)


    @property
    def instrument_dim(self) -> int:
        """Number of underlying risky instruments processes."""
        return self.instrument_simulator.dimension


    def add_derivative(self,
                       derivative: derivatives.Derivative,
                       link: int,
                       exposure: float):
        assert issubclass(type(derivative), derivatives.Derivative)
        assert 0 <= int(link) < self.instrument_dim

        entry = {
            "derivative": derivative,
            "link": int(link),
            "exposure": float(exposure)
            }

        self.derivatives.append(entry)


    def link_apply(self, attr, time, instruments, numeraire):
        """Computes attr of derivative for each derivative according to link.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            marginals: (batch_size, book_size, ...)
        """
        marginals = []
        for entry in self.derivatives:
            linked = instruments[:, entry["link"], :]
            func = getattr(entry["derivative"], attr)
            marginal = func(time, linked, numeraire)
            marginals.append(marginal * entry["exposure"])

        return tf.stack(marginals, axis=1)


    def bucket(self, marginals):
        """Sums each marginal into a bucket according to link.
        Args:
            marginals: (batch_size, book_size, ...)
        Returns:
            bucket: (batch_size, instrument_dim, ...)
        """
        links = tf.constant([entry["link"] for entry in self.derivatives],
                            INT_DTYPE)

        v = []
        for k in range(self.instrument_dim):
            mask = tf.squeeze(tf.where(links == k), axis=1)
            v.append(tf.reduce_sum(
                tf.gather(marginals, mask, axis=1), axis=1, keepdims=True))

        return tf.concat(v, axis=1)


    def payoff(self, time: tf.Tensor, instruments: tf.Tensor,
               numeraire: tf.Tensor):
        """Computes payoff of book in terms of numeraire for each sample
        in batch
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            payoff: (batch_size, )
        """
        marginals = self.link_apply("payoff", time, instruments, numeraire)

        return tf.reduce_sum(marginals, axis=1)


    def adjoint(self, time: tf.Tensor, instruments: tf.Tensor,
                numeraire: tf.Tensor):
        """Computes adjoint of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, instrument_dim, timesteps + 1)
        """
        marginals = self.link_apply("adjoint", time, instruments, numeraire)
        bucket = self.bucket(marginals)

        return bucket


    def value(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes value of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instrument: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, timesteps + 1)
        """
        marginals = self.link_apply("value", time, instruments, numeraire)

        return tf.reduce_sum(marginals, axis=1)


    def delta(self, time: tf.Tensor, instruments: tf.Tensor,
              numeraire: tf.Tensor):
        """Computes delta of book in terms of numeraire for each sample
        in batch.
        Args:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        Returns:
            value: (batch_size, instrument_dim, timesteps + 1)
        """
        marginals = self.link_apply("delta", time, instruments, numeraire)
        bucket = self.bucket(marginals)

        return bucket


    def discretise_time(self, timesteps):
        return tf.cast(tf.linspace(0., self.maturity, timesteps + 1),
                       FLOAT_DTYPE)


    def sample_numeraire(self, time, init_numeraire, risk_neutral):
        numeraire = self.numeraire_simulator.simulate(
            time, init_numeraire, 1, risk_neutral)

        return tf.squeeze(numeraire)


    def sample_paths(
            self,
            init_instruments: tf.Tensor,
            init_numeraire: tf.Tensor,
            batch_size: int,
            timesteps: int,
            risk_neutral: bool,
            use_sobol: bool=False,
            skip: int=0,
            exploring_loc: float=None,
            exploring_scale: float=None
            ) -> tf.Tensor:
        """Simulate sample paths.
        Args:
            init_instruments: (state_dim, )
            init_numeraire: (1, )
            batch_size: int
            timesteps: int
            risk_neutral: bool
        Returns:
            time: (timesteps + 1, )
            instruments: (batch_size, instrument_dim, timesteps + 1)
            numeraire: (timesteps + 1, )
        """
        time = self.discretise_time(timesteps)

        if exploring_loc is not None and exploring_scale is not None:
            exploring = self.exploring_start(
                init_instruments,
                batch_size,
                exploring_loc,
                exploring_scale
                )
        else:
            exploring = init_instruments

        instruments = self.instrument_simulator.simulate(
            time=time,
            init_state=exploring,
            batch_size=batch_size,
            risk_neutral=risk_neutral,
            use_sobol=use_sobol,
            skip=skip)
        numeraire = self.sample_numeraire(time, init_numeraire, risk_neutral)

        return time, instruments, numeraire


    def exploring_start(self, state, batch_size, loc, scale):
        rvs = tf.random.truncated_normal(
            shape=(batch_size, tf.shape(state)[-1]),
            mean=tf.math.log(loc**2 / tf.sqrt(loc**2 + scale**2)),
            stddev=tf.sqrt(tf.math.log(scale**2 / loc**2 + 1)),
            dtype=FLOAT_DTYPE
            )

        return tf.exp(rvs)


class TradeBook(DerivativeBook):
    def __init__(self, hedgebook):
        if not isinstance(hedgebook, DerivativeBook):
            raise TypeError("hedgebook not of type DerivativeBook.")
        elif hedgebook.instrument_dim > 1:
            raise ValueError("multivariate books not supported.")

        super().__init__(
            hedgebook.maturity,
            hedgebook.instrument_simulator, # not used
            hedgebook.numeraire_simulator)  # not used
        self.hedgebook = hedgebook


    @property
    def book_size(self) -> int:
        """Number of derivatives in the book."""
        return len(self.hedgebook.derivatives)


    @property
    def instrument_dim(self) -> int:
        """Number of underlying risky instruments processes."""
        return self.hedgebook.instrument_simulator.dimension \
            + len(self.derivatives)


    def add_derivative(self, derivative: derivatives.Derivative):
        super().add_derivative(derivative, 0, 1)


    def value(self, time, instruments, numeraire):
        main = instruments[:, 0, tf.newaxis, :]
        return self.hedgebook.value(time, main, numeraire)


    def delta(self, time, instruments, numeraire):
        main = instruments[:, 0, tf.newaxis, :]
        maindelta = self.hedgebook.delta(time, main, numeraire)
        marginals = super().link_apply("delta", time, main, numeraire)

        return tf.concat([maindelta, marginals], axis=1)


    def payoff(self, time, instruments, numeraire):
        return self.hedgebook.payoff(time, instruments, numeraire)


    def adjoint(self, time, instruments, numeraire):
        raise NotImplementedError("adjoints not supported.")


    def sample_paths(self,
                     init_instruments: tf.Tensor,
                     init_numeraire: tf.Tensor,
                     batch_size: int,
                     timesteps: int,
                     risk_neutral: bool,
                     **kwargs):
        time, instruments, numeraire = super().sample_paths(
            init_instruments, init_numeraire, batch_size, timesteps,
            risk_neutral, **kwargs)
        tradevalues = super().link_apply("value", time, instruments, numeraire)

        return time, tf.concat([instruments, tradevalues], axis=1), numeraire


class BasketBook(DerivativeBook):
    def _apply(self, attr, time, instruments, numeraire):
        derivative = self.derivatives[0]["derivative"]

        return getattr(derivative, attr)(time, instruments, numeraire)


    def payoff(self, time, instruments, numeraire):
        return self._apply("payoff", time, instruments, numeraire)


    def adjoint(self, time, instruments, numeraire):
        return self._apply("adjoint", time, instruments, numeraire)


    def value(self, time, instruments, numeraire):
        return self._apply("value", time, instruments, numeraire)


    def delta(self, time, instruments, numeraire):
        return self._apply("delta", time, instruments, numeraire)


    def exploring_start(self, state, batch_size, loc, scale):
        rvs = tf.random.truncated_normal(
            shape=(batch_size, tf.shape(state)[-1]),
            mean=loc,
            stddev=scale,
            dtype=FLOAT_DTYPE
            )

        return rvs
