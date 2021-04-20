# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

from time import perf_counter

import utils
import preprocessing
from constants import FLOAT_DTYPE
import gradient_models

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

    def add_testcase(self, name, model, train_size):
        if isinstance(model, gradient_models.SequenceDeltaNetwork):
            normaliser = preprocessing.IOMeanVarianceNormaliser()

        elif isinstance(model, gradient_models.SequenceTwinNetwork) \
            or isinstance(model, gradient_models.SequenceValueNetwork):
            normaliser = preprocessing.DifferentialMeanVarianceNormaliser()

        else:
            raise TypeError("invalid model.")

        self.testcases.append({
            "name": name,
            "model": model,
            "normaliser": normaliser,
            "train_size": int(train_size),
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
            "time": time[::skip][..., :-1],
            "instruments": instruments[..., ::skip][..., :-1],
            "numeraire": numeraire[::skip][..., :-1],
            "value": value[..., ::skip][..., :-1],
            "delta": delta[..., ::skip][..., :-1],
            "payoff": payoff,
            "adjoint": adjoint[..., ::skip][..., :-1],
            "metrics": metrics
            }

        if not tf.math.reduce_variance(raw_data["payoff"]) > 0:
            raise RuntimeError("payoff data is degenerate.")

        return raw_data


    def transform(self, case, raw_data, only_inputs=False):
        if only_inputs and case["trained"]:
            return case["normaliser"].transform_x(
                raw_data["instruments"][..., :self.timesteps])
        elif only_inputs and not case["trained"]:
            raise ValueError("if only_inputs is True, then case must be ",
                             "'trained'.")

        keys = ["instruments", "payoff", "adjoint"]
        x, y, dydx = [raw_data[key] for key in keys]

        attr = "fit_transform" if not case["trained"] else "transform"

        if isinstance(case["model"], gradient_models.SequenceValueNetwork):
            norm_x, norm_y, norm_dydx = getattr(case["normaliser"], attr)(
                x, y, dydx)

            return norm_x, norm_y

        elif isinstance(case["model"], gradient_models.SequenceTwinNetwork):
            norm_x, norm_y, norm_dydx = getattr(case["normaliser"], attr)(
                x, y, dydx)

            return norm_x, [norm_y, norm_dydx]

        elif isinstance(case["model"], gradient_models.SequenceDeltaNetwork):
            norm_x, norm_dydx = getattr(case["normaliser"], attr)(x, dydx)

            return norm_x, norm_dydx


    def inverse_transform(self, case, norm_dydx):
        if isinstance(case["model"], gradient_models.SequenceValueNetwork) \
            or isinstance(case["model"], gradient_models.SequenceTwinNetwork):
            return case["normaliser"].inverse_transform_dydx(norm_dydx)

        elif isinstance(case["model"], gradient_models.SequenceDeltaNetwork):
            return case["normaliser"].inverse_transform_y(norm_dydx)


    def evaluate_case(self, case, raw_data):
        if not case["trained"]:
            raise ValueError("case must be trained before evaluation.")

        inputs = self.transform(case, raw_data, only_inputs=True)
        norm_dydx = case["model"].gradient(inputs)
        dydx = self.inverse_transform(case, norm_dydx)

        return dydx


    def _get_loss_instance(self):
        reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        loss = tf.keras.losses.MeanSquaredError(reduction)

        return loss

    def _compute_weight(self, case, inputs, output):
        y, dydx = case["model"](inputs)
        loss = self._get_loss_instance()
        x = loss(output[0][:, tf.newaxis], y) / loss(output[1], dydx)

        return 1 / (1 + x)


    def train_case(self, case, batch_size, epochs, raw_data):
        inputs, output = self.transform(case, raw_data)

        inputs = inputs[:case["train_size"], ...]

        if isinstance(output, list):
            output = [x[:case["train_size"], ...] for x in output]
        else:
            output = output[:case["train_size"], ...]

        lr_schedule = utils.PeakSchedule(
            self.learning_rate_min, self.learning_rate_max, epochs)
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_schedule),
            tf.keras.callbacks.EarlyStopping("loss", patience=10)]

        if isinstance(case["model"], gradient_models.SequenceTwinNetwork):
            weight = self._compute_weight(case, inputs, output)
            loss_weights = [weight, 1 - weight]
        else:
            loss_weights = [1.]

        case["model"].compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=self._get_loss_instance(),
            loss_weights=loss_weights)

        case["model"](inputs) # remove initial overhead time

        start = perf_counter()
        case["history"] = case["model"].fit(
            inputs, output,
            batch_size,
            epochs,
            callbacks=callbacks,
            verbose=0)
        end = perf_counter() - start

        case["train_time"] = end
        case["trainable_variables"] = tf.reduce_sum(
            [tf.size(w) for w in case["model"].trainable_variables])
        case["non_trainable_variables"] = tf.reduce_sum(
            [tf.size(w) for w in case["model"].non_trainable_variables])

        case["trained"] = True

        return


    def train(self, epochs, batch_size):
        sample_size = max([case["train_size"] for case in self.testcases])
        raw_data = self.sample(sample_size, skip=0, exploring=True)

        for case in self.testcases:
            self.train_case(case, batch_size, epochs, raw_data)

        return


    def test_skip(self):
        return max([case["train_size"] for case in self.testcases])


    def weighted_mape(self, actual, prediction):
        a = utils.cast_apply(tf.reduce_sum, tf.abs(actual - prediction), axis=0)
        b = utils.cast_apply(tf.reduce_sum, tf.abs(actual), axis=0)

        return a / b


    def mean_squared_error(self, actual, prediction):
        error = utils.cast_apply(
            tf.reduce_mean,
            tf.square(actual - prediction),
            axis=0)

        return error


    def mean_absolute_error(self, actual, prediction):
        error = utils.cast_apply(
            tf.reduce_mean,
            tf.abs(actual - prediction),
            axis=0)

        return error


    def test(self, sample_size):
        skip = self.test_skip()
        raw_data = self.sample(sample_size, skip, exploring=True)

        for case in self.testcases:
            dydx = self.evaluate_case(case, raw_data)

            for attr in ["weighted_mape",
                         "mean_squared_error",
                         "mean_absolute_error"]:
                error_measure = getattr(self, attr)
                case[f"test_delta_{error_measure.__name__}"] = error_measure(
                    raw_data["delta"], dydx)

            case["tested"] = True

        return


    def test_summary(self, file_name=None):
        columns = [
            {"title": "training time", "key": "train_time"},
            {"title": "trainable vars", "key": "trainable_variables"},
            {"title": "non-trainable vars", "key": "non_trainable_variables"}
            ]

        case_size = max([len(case["name"]) for case in self.testcases]) + 1
        block_size = max([len(col["title"]) for col in columns]) + 3

        header = "".rjust(case_size) + "".join(
            [col["title"].rjust(block_size) for col in columns])
        body = ""
        for case in self.testcases:
            body += case["name"].ljust(case_size)
            for val in [case[col["key"]] for col in columns]:
                body += f"{val:.6f}".rjust(block_size)
            body += "\n"

        summary = header + "\n" + body

        if file_name is not None:
            with open(file_name, "w") as file:
                file.write(summary)

        print(summary)


    def boxplot(self, sample_size):
        skip = self.test_skip()
        raw_data = self.sample(sample_size, skip, exploring=True)

        for case in self.testcases:
            x = raw_data["instruments"]
            y, dydx = self.evaluate_case(case, x)

            plt.figure()
            plt.boxplot(
                tf.math.log(y / raw_data["value"]).numpy()[..., :-1],
                sym="",
                whis=[2.5, 97.5])
            plt.show()

            plt.figure()
            plt.boxplot(
                tf.math.log(dydx / raw_data["delta"])[:, 0, :-1].numpy(),
                sym="",
                whis=[2.5, 97.5])
            plt.show()

        return


    def distance_to_line_plot(self, sample_size):
        skip = self.test_skip()
        raw_data = self.sample(sample_size, skip, exploring=True)

        dydx_lst = [self.evaluate_case(case, raw_data) \
                    for case in self.testcases]

        for dim in tf.range(self.book.instrument_dim):
            for step in tf.range(self.timesteps):
                plt.figure()

                xaxis = raw_data["delta"][..., dim, step]

                for dydx in dydx_lst:
                    yaxis = dydx[..., dim, step]
                    plt.scatter(xaxis, yaxis, s=0.5)
                plt.legend([case["name"] for case in self.testcases])

                xlim = plt.xlim()
                ylim = plt.ylim()

                val = min(plt.xlim()[0], plt.ylim()[0])
                plt.axline(
                    [val, val],
                    slope=1.,
                    color="black")

                plt.xlim(xlim)
                plt.ylim(ylim)

                plt.show()


    def make_feature_function(self, case):
        def gradient_function(raw_data):
            dydx = self.evaluate_case(case, raw_data)

            return tf.unstack(dydx * raw_data["numeraire"][:-1], axis=-1)

        return gradient_function


def barrier_visualiser(driver, sample_size):
    derivative = driver.book.derivatives[0]["derivative"]

    def metric(time, instruments, numeraire):
        crossed = derivative.crossed(instruments[:, 0, :])
        return crossed[..., ::2**driver.frequency]

    skip = driver.test_skip()
    raw_data = driver.sample(
        sample_size,
        skip,
        exploring=True,
        metrics=[metric]
        )
    crossed = raw_data["metrics"][0]

    for case in driver.testcases:
        x = raw_data["instruments"]
        y, dydx = driver.evaluate_case(case, x)

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

                plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
                plt.scatter(xaxis, target, color="red", s=0.5, alpha=0.5)
                plt.scatter(xaxis, prediction, color="black", s=0.5, alpha=0.5)

                plt.title(f"value {step} {name}")
                plt.show()

                # delta
                plt.figure()
                prediction = tf.boolean_mask(dydx[..., step], mask)
                target = tf.boolean_mask(raw_data["delta"][..., step], mask)
                data = tf.boolean_mask(raw_data["adjoint"][..., step], mask)

                plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
                plt.scatter(xaxis, target, color="red", s=0.5, alpha=0.5)
                plt.scatter(xaxis, prediction, color="black", s=0.5, alpha=0.5)

                plt.title(f"delta {step} {name}")
                plt.show()


def dga_putcall_visualiser(driver, sample_size):
    derivative = driver.book.derivatives[0]["derivative"]
    skip = driver.test_skip()
    raw_data = driver.sample(
        sample_size,
        skip,
        exploring=True
        )

    for case in driver.testcases:
        x = raw_data["instruments"]
        y, dydx = driver.evaluate_case(case, x)

        instrument = raw_data["instruments"][:, 0, :]
        dga = derivative._dga(raw_data["time"], instrument)
        spot = dga * tf.pow(instrument, derivative.maturity - raw_data["time"])

        dt = derivative._increments(raw_data["time"], 1)
        scale = (derivative.maturity - raw_data["time"] + dt) / instrument

        for step in tf.range(driver.timesteps + 1):
            xaxis = spot[..., step]

            # value
            plt.figure()
            prediction = y[..., step]
            target = raw_data["value"][..., step]
            data = raw_data["payoff"]

            plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
            plt.scatter(xaxis, target, color="red", s=0.5, alpha=0.5)
            plt.scatter(xaxis, prediction, color="blue", s=0.5, alpha=0.5)

            plt.title(f"value {step}")
            plt.show()

            # delta
            plt.figure()
            prediction = dydx[..., 0, step] / scale[..., step]
            target = raw_data["delta"][..., 0, step] / scale[..., step]
            data = raw_data["adjoint"][..., 0, step] / scale[..., step]

            plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
            plt.scatter(xaxis, target, color="red", s=0.5, alpha=0.5)
            plt.scatter(xaxis, prediction, color="blue", s=0.5, alpha=0.5)

            plt.title(f"delta {step}")
            plt.show()


def markovian_visualiser(driver, sample_size):
    skip = driver.test_skip()
    raw_data = driver.sample(
        sample_size,
        skip,
        exploring=True
        )

    for case in driver.testcases:
        dydx = driver.evaluate_case(case, raw_data)

        instrument = raw_data["instruments"][:, 0, :]

        for step in tf.range(driver.timesteps):
            key = tf.argsort(instrument[..., step])
            xaxis = tf.gather(instrument[..., step], key)

            # # value
            # plt.figure()
            # prediction = tf.gather(y[..., step], key).numpy()
            # target = tf.gather(raw_data["value"][..., step], key).numpy()
            # data = tf.gather(raw_data["payoff"], key).numpy()

            # plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
            # plt.plot(xaxis, target, "--", color="red")
            # plt.plot(xaxis, prediction, color="black")

            # plt.title(f"value {step}")
            # plt.show()

            # delta
            plt.figure()
            prediction = tf.gather(dydx[..., 0, step], key).numpy()
            target = tf.gather(raw_data["delta"][..., 0, step], key).numpy()
            data = tf.gather(raw_data["adjoint"][..., 0, step], key).numpy()

            plt.scatter(xaxis, data, color="grey", s=0.5, alpha=0.5)
            plt.plot(xaxis, target, "--", color="red")
            plt.plot(xaxis, prediction, color="black")

            plt.title(f"delta {step}")
            plt.show()
