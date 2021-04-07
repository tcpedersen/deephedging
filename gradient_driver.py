# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from constants import FLOAT_DTYPE

class GradientDriver(object):
    def __init__(self,
                 timesteps,
                 frequency,
                 init_instruments,
                 init_numeraire,
                 book,
                 learning_rate_min=1e-4,
                 learning_rate_max=1e-2):
        self.timesteps = timesteps
        self.frequency = frequency
        self.init_instruments = init_instruments
        self.init_numeraire = init_numeraire
        self.book = book

        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max

        self.exploring_loc = None
        self.exploring_scale = None

        self.testcases = []

    def  add_testcase(self, name, model, normaliser):
        self.testcases.append({
            "name": name,
            "model": model,
            "normaliser": normaliser,
            "trained": False,
            "tested": False
            })


    def set_exploration(self, loc, scale):
        self.exploring_loc = tf.convert_to_tensor(loc, FLOAT_DTYPE)
        self.exploring_scale = tf.convert_to_tensor(scale, FLOAT_DTYPE)


    def sample(self, size, skip, exploring=True, metrics=None):
        time, instruments, numeraire = self.book.sample_paths(
            init_instruments=self.init_instruments,
            init_numeraire=self.init_numeraire,
            batch_size=size,
            timesteps=self.timesteps * 2**self.frequency,
            risk_neutral=True,
            use_sobol=True,
            skip=skip,
            exploring_loc=self.exploring_loc if exploring else None,
            exploring_scale=self.exploring_scale if exploring else None
            )

        value = self.book.value(time, instruments, numeraire)
        delta = self.book.delta(time, instruments, numeraire)
        payoff = self.book.payoff(time, instruments, numeraire)
        adjoint = self.book.adjoint(time, instruments, numeraire)

        if metrics is not None:
            metrics = [m(time, instruments, numeraire) for m in metrics]

        skip = 2**self.frequency
        raw_data = {
            "time": time[::skip],
            "instruments": instruments[..., ::skip],
            "numeraire": numeraire[::skip],
            "value": value[..., ::skip],
            "delta": delta[..., ::skip],
            "payoff": payoff,
            "adjoint": adjoint[..., ::skip],
            "metrics": metrics
            }

        if not tf.math.reduce_variance(raw_data["payoff"]) > 0:
            raise RuntimeError("payoff data is degenerate.")

        return raw_data


    def get_input(self, case, raw_data):
        keys = ["instruments", "payoff", "adjoint"]
        x, y, dydx = [raw_data[key] for key in keys]

        if not case["trained"]:
            return case["normaliser"].fit_transform(x, y, dydx)

        return case["normaliser"].transform(x, y, dydx)

    def train_case(self, case, batch_size, epochs, raw_data):
        x, y, dydx = self.get_input(case, raw_data)

        lr_schedule = utils.PeakSchedule(
            self.learning_rate_min, self.learning_rate_max, epochs)
        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule)]

        weight = 1 / (1 + tf.math.reduce_std(y) / tf.math.reduce_std(dydx))

        case["model"].compile(optimizer=tf.keras.optimizers.Adam(),
                              loss="mean_squared_error",
                              loss_weights=[weight, 1 - weight])

        case["history"] = case["model"].fit(
            x, [y, dydx], batch_size, epochs, callbacks=callbacks)

        case["trained"] = True

        return

    def train(self, sample_size, epochs, batch_size):
        raw_data = self.sample(sample_size, skip=0, exploring=True)

        for case in self.testcases:
            self.train_case(case, batch_size, epochs, raw_data)
            case["train_batch_size"] = sample_size

        return


    def test_skip(self):
        return max([case["train_batch_size"] for case in self.testcases])


    def test(self, sample_size):
        skip = self.test_skip()
        raw_data = self.sample(sample_size, skip, exploring=True)

        for case in self.testcases:
            input_data = self.get_input(case, raw_data)[0]
            value, delta = case["model"](input_data, training=False)

            case["test_value_mse"] = tf.reduce_mean(
                tf.square(value - raw_data["value"]), 0)
            case["test_delta_mse"] = tf.reduce_mean(
                tf.square(delta - raw_data["delta"]), 0)

        return


    def evaluate_case(self, case, raw_data):
        if not case["trained"]:
            raise ValueError("case must be trained before evaluation.")

        norm_x = self.get_input(case, raw_data)[0]
        norm_y, norm_dydx = case["model"](norm_x, training=False)

        return case["normaliser"].inverse_transform(norm_x, norm_y, norm_dydx)


def barrier_visualiser(driver, sample_size):
    derivative = driver.book.derivatives[0]["derivative"]

    def metric(time, instruments, numeraire):
        return derivative.crossed(instruments[:, 0, :])[..., ::2**driver.frequency]

    skip = driver.test_skip()
    raw_data = driver.sample(
        sample_size,
        skip,
        exploring=True,
        metrics=[metric]
        )
    crossed = raw_data["metrics"][0]

    for case in driver.testcases:
        x, y, dydx = driver.evaluate_case(case, raw_data)

        for step in tf.range(driver.timesteps + 1):

            names = ["crossed", "non-crossed"]
            masks = [crossed[..., step], ~crossed[..., step]]

            for name, mask in zip(names, masks):
                xaxis = tf.boolean_mask(x[..., step], mask)

                # value
                plt.figure()
                prediction = tf.boolean_mask(y[..., step], mask)
                target = tf.boolean_mask(raw_data["value"][..., step], mask)
                data = tf.boolean_mask(raw_data["payoff"], mask)

                plt.scatter(xaxis, data, color="grey", s=0.5)
                plt.scatter(xaxis, target, color="red", s=0.5)
                plt.scatter(xaxis, prediction, color="black", s=0.5)

                plt.title(f"value {step} {name}")
                plt.show()

                # delta
                plt.figure()
                prediction = tf.boolean_mask(dydx[..., step], mask)
                target = tf.boolean_mask(raw_data["delta"][..., step], mask)
                data = tf.boolean_mask(raw_data["adjoint"][..., step], mask)

                plt.scatter(xaxis, data, color="grey", s=0.5)
                plt.scatter(xaxis, target, color="red", s=0.5)
                plt.scatter(xaxis, prediction, color="black", s=0.5)

                plt.title(f"delta {step} {name}")
                plt.show()
