# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from time import perf_counter
from tensorflow_probability.python.internal import special_math

from constants import FLOAT_DTYPE_EPS, FLOAT_DTYPE, DPI
import hedge_models

ONE_OVER_SQRT_TWO_PI = 1. / tf.sqrt(2. * np.pi)
SQRT_TWO = tf.sqrt(2.)

# ==============================================================================
# === Gaussian
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * tf.exp(-0.5 * x * x)


def abrahamowitz_stegun_cdf(x):
    """Approximation of norm_cdf."""
    p = 0.2316419
    c = 0.918938533204672

    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    a = tf.abs(x)
    t = 1 / (1 + a * p)
    s = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    y = s * tf.exp(-0.5 * x * x - c)

    return tf.where(x < 0, y, 1 - y)


def norm_cdf(x, approx=False):
    return abrahamowitz_stegun_cdf(x) if approx else special_math.ndtr(x)


def norm_qdf(x):
    return special_math.ndtri(x)


def near_positive_definite(A):
    dtype = tf.float64
    C = (A + tf.transpose(A)) / 2.
    eigval, eigvec = tf.linalg.eig(tf.cast(C, dtype))
    eigval = tf.where(tf.math.real(eigval) < 0, 0, eigval)
    psd = tf.math.real(eigvec @ tf.linalg.diag(eigval) @ tf.transpose(eigvec))
    eps = tf.sqrt(tf.cast(FLOAT_DTYPE_EPS, dtype))

    return tf.cast(psd + tf.eye(psd.shape[0], dtype=dtype) * eps, FLOAT_DTYPE)

# ==============================================================================
# === Training
class PeakSchedule:
    def __init__(self, a, b, n):
        self.a = a
        self.b = b

        self.n1, self.n2, self.n3 = 0, n // 4, n // 2

    def __call__(self, n, alpha):
        if n <= self.n2:
            return (self.a - self.b)/(self.n1 - self.n2) * n \
                - (self.a * self.n2 - self.b * self.n1) / (self.n1 - self.n2)
        elif self.n2 < n < self.n3:
            return -(self.a - self.b) / (self.n2 - self.n3) * n \
                + (self.a * self.n2 - self.b * self.n3) / (self.n2 - self.n3)
        else:
            return self.a


# ==============================================================================
# === other
def cast_apply(func, x, **kwargs):
    return tf.cast(func(tf.cast(x, tf.float64), **kwargs), x.dtype)

def precise_mean(x, **kwargs):
    return cast_apply(tf.reduce_mean, x, **kwargs)

def precise_confidence_interval(x, alpha, **kwargs):
    n = tf.cast(tf.shape(x)[0], FLOAT_DTYPE)
    z1, z2 = norm_qdf((1 - alpha) / 2), norm_qdf((1 + alpha) / 2)
    mean = precise_mean(x, **kwargs)
    var = cast_apply(tf.math.reduce_variance, x, **kwargs)

    return mean + z1 * tf.sqrt(var / n), mean + z2 * tf.sqrt(var / n)

# ==============================================================================
# === experiments
class HedgeDriver(object):
    def __init__(self,
                 timesteps,
                 frequency,
                 init_instruments,
                 init_numeraire,
                 book,
                 cost=None,
                 risk_neutral=False,
                 learning_rate=1e-2):
        self.timesteps = timesteps
        self.frequency = frequency
        self.init_instruments = init_instruments
        self.init_numeraire = init_numeraire
        self.book = book
        self.cost = cost
        self.risk_neutral = risk_neutral

        self.learning_rate = float(learning_rate)
        self.optimizer = tf.keras.optimizers.Adam

        self.verbose = 0

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            verbose=self.verbose,
            patience=2
        )

        self.callbacks = [early_stopping, reduce_lr]
        self.testcases = []
        self.liability_free = None


    def sample(self, size, metrics=None, **kwargs):
        time, instruments, numeraire = self.book.sample_paths(
            self.init_instruments,
            self.init_numeraire,
            size,
            self.timesteps * 2**self.frequency,
            self.risk_neutral,
            **kwargs)

        if metrics is not None:
            metrics = [m(time, instruments, numeraire) for m in metrics]

        delta = self.book.delta(time, instruments, numeraire)
        payoff = self.book.payoff(time, instruments, numeraire)

        skip = 2**self.frequency
        sample = {
            "time": time[::skip],
            "instruments": instruments[..., ::skip],
            "numeraire": numeraire[::skip],
            "delta": delta[..., ::skip],
            "payoff": payoff,
            "metrics": metrics
            }

        return sample


    def _translate_feature_function(self, feature_function_name):
        if not isinstance(feature_function_name, str):
            raise TypeError("feature_function_name must of type str, not ",
                            type(feature_function_name))

        if feature_function_name == "log_martingale":
            return self._log_martingale
        elif feature_function_name == "log_martingale_with_time":
            return self._log_martingale_with_time
        elif feature_function_name == "delta":
            return self._undiscounted_delta
        else:
            raise ValueError("feature_function_name: ",
                             f"'{feature_function_name}' ",
                             "is not valid.")

    def validate_price_type(self, price_type):
        valid_price_types = ["indifference", "arbitrage", "constant"]
        if not price_type in valid_price_types:
            raise ValueError(f"price_type '{price_type}' not in ",
                             f"{valid_price_types}.")


    def process_case(self,
                     name,
                     model,
                     risk_measure,
                     normaliser,
                     feature_function,
                     price_type):
        if not callable(feature_function):
            feature_function = self._translate_feature_function(
                feature_function)
        self.validate_price_type(price_type)

        if not issubclass(type(model), hedge_models.BaseHedge):
            raise TypeError("model must be subclass of ",
                            f"{str(hedge_models.Hedge)}.")

        if not issubclass(type(risk_measure), hedge_models.OCERiskMeasure):
            raise TypeError("risk_measure must be subclass of ",
                            f"{str(hedge_models.OCERiskMeasure)}.")

        if self.cost is not None:
            model.add_cost(self.cost)

        case = {"name": str(name),
                "model": model,
                "risk_measure": risk_measure,
                "normaliser": normaliser,
                "price_type": price_type,
                "feature_function": feature_function,
                "trained": False,
                "tested": False}

        return case

    def add_testcase(self,
                     name,
                     model,
                     risk_measure,
                     normaliser,
                     feature_function,
                     price_type):
        case = self.process_case(name, model, risk_measure, normaliser,
                                 feature_function, price_type)

        self.testcases.append(case)


    def add_liability_free(
            self,
            model,
            risk_measure,
            normaliser,
            feature_function):
        self.liability_free = self.process_case(
            name="liability free",
            model=model,
            risk_measure=risk_measure,
            normaliser=normaliser,
            feature_function=feature_function,
            price_type="constant")

        self.liability_free["price"] = 0.


    def normalise_features(self, case, features):
        if case["normaliser"] is not None:
             if not case["trained"]:
                 case["normaliser"].fit(features)
             return case["normaliser"].transform(features)
        return features


    def _split_by_timesteps(self, tensor):
        """Splits tensor along last dimension into timesteps pieces ignoring the
        last entry.
        Args:
            tensor: (batch, height, timesteps + 1)
        Returns:
            list of (batch, height) of len timesteps.
        """
        return tf.unstack(tensor[..., :-1], self.timesteps, -1)


    def _log_martingale(self, raw_data):
        tensor = tf.math.log(raw_data["instruments"] / raw_data["numeraire"])

        return self._split_by_timesteps(tensor)


    def _log_martingale_with_time(self, raw_data):
        time = raw_data["time"]
        lst = self._log_martingale(raw_data)
        pad = [[0, 0], [1, 0]]

        return [tf.pad(x, pad, constant_values=t) for t, x in zip(time, lst)]


    def _undiscounted_delta(self, raw_data):
        tensor = raw_data["delta"] * raw_data["numeraire"]
        return self._split_by_timesteps(tensor)


    def get_input(self, case, raw_data):
        raw_features = case["feature_function"](raw_data)
        features = self.normalise_features(case, raw_features)
        martingales = raw_data["instruments"] / raw_data["numeraire"]

        if len(features) != self.timesteps:
            raise RuntimeError(f"len(features): {len(features)} != timesteps.")
        if tf.shape(martingales)[-1] != self.timesteps + 1:
            raise RuntimeError("tf.shape(martingales)[-1]: ",
                               f"{tf.shape(martingales)[-1]} != timesteps.")

        return [features, martingales, raw_data["payoff"]]


    def evaluate_case(self, case, raw_data):
        input_data = self.get_input(case, raw_data)

        return case["model"](input_data[:-1])


    def get_risk(self, case, input_data):
        value, costs = case["model"](input_data[:-1])
        wealth = value - costs - input_data[-1]

        return value, costs, wealth, case["model"].risk_measure.evaluate(wealth)


    def assert_case_is_trained(self, case):
        assert case["trained"], f"case '{case['name']}' is not trained yet."


    def assert_case_is_tested(self, case):
        assert case["tested"], f"case '{case['name']}' is not tested yet."


    def assert_all_trained(self):
        for case in self.testcases:
            self.assert_case_is_trained(case)
        if self.liability_free is not None:
            self.assert_case_is_trained(self.liability_free)


    def assert_all_tested(self):
        for case in self.testcases:
            self.assert_case_is_tested(case)
        if self.liability_free is not None:
            self.assert_case_is_tested(self.liability_free)


    def train_case(self, case, input_data, **kwargs):
        optimizer = self.optimizer(self.learning_rate)
        case["model"].compile(
            risk_measure=case["risk_measure"],
            optimizer=optimizer)

        case["model"](input_data[:-1]) # skip overhead time in build

        start = perf_counter()
        case["history"] = case["model"].fit(
            input_data,
            callbacks=self.callbacks,
            verbose=self.verbose,
            **kwargs)
        end = perf_counter() - start

        case["trainable_variables"] = tf.reduce_sum(
            [tf.size(w) for w in case["model"].trainable_variables])
        case["non_trainable_variables"] = tf.reduce_sum(
            [tf.size(w) for w in case["model"].non_trainable_variables])

        case["train_time"] = end
        case["trained"] = True

        case["train_risk"] = self.get_risk(case, input_data)[-1]


    def train(self, sample_size, epochs, batch_size):
        raw_data = self.sample(sample_size)

        for case in self.testcases:
            input_data = self.get_input(case, raw_data)
            self.train_case(case, input_data,
                            batch_size=batch_size,
                            epochs=epochs)

        if self.liability_free is not None:
            input_data = self.get_input(self.liability_free, raw_data)
            input_data[-1] = tf.zeros_like(input_data[-1])
            self.train_case(self.liability_free, input_data,
                            batch_size=batch_size,
                            epochs=epochs)


    def test_case(self, case, input_data):
        value, costs, wealth, risk = self.get_risk(case, input_data)

        names = ["value", "costs", "wealth"]
        tensors = [value, costs, wealth]
        for name, tensor in zip(names, tensors):
            case[f"test_mean_{name}"] = precise_mean(tensor)
            case[f"test_mean_abs_{name}"] = precise_mean(tf.abs(tensor))
            case[f"test_variance_{name}"] = cast_apply(
                tf.math.reduce_variance, tensor)

        case["test_risk"] = risk
        case["price"] = self.get_price(case)

        case["test_wealth_with_price_abs_mean"] = precise_mean(
            tf.abs(wealth + case["price"]))
        case["test_wealth_with_price_variance"] = cast_apply(
            tf.math.reduce_variance, wealth + case["price"])

        case["tested"] = True


    def test(self, sample_size):
        self.assert_all_trained()
        raw_data = self.sample(sample_size)

        if self.liability_free is not None:
            input_data = self.get_input(self.liability_free, raw_data)
            input_data[-1] = tf.zeros_like(input_data[-1])
            self.test_case(self.liability_free, input_data)

        for case in self.testcases:
            input_data = self.get_input(case, raw_data)
            self.test_case(case, input_data)

        self.test_mean_payoff = precise_mean(raw_data["payoff"])


    def get_price(self, case):
        self.assert_case_is_trained(case)

        if case["price_type"] == "arbitrage":
            raw_data = self.sample(1)
            keys = ["time", "instruments", "numeraire"]
            input_data = [raw_data[key] for key in keys]

            value = self.book.value(*input_data)[0, 0]

        elif case["price_type"] == "indifference":
            if self.cost is None and self.risk_neutral:
                liability_free_risk = 0.
            else:
                liability_free_risk = self.liability_free["test_risk"]

            value = case["test_risk"] - liability_free_risk

        elif case["price_type"] == "constant":
            return case["price"]

        return value


    def test_summary(self, file_name=None):
        self.assert_all_tested()

        columns = [
            {"title": "name",
             "key": "name",
             "type": str},

            {"title": "mean costs",
             "key": "test_mean_costs",
             "type": float},

            {"title": "variance costs",
             "key": "test_variance_costs",
             "type": float},

            {"title": "mean wealth",
             "key": "test_mean_wealth",
             "type": float},

            {"title": "variance wealth",
             "key": "test_variance_wealth",
             "type": float},

            {"title": "mean abs wealth w. price",
             "key": "test_wealth_with_price_abs_mean",
             "type": float},

            {"title": "variance wealth w. price",
             "key": "test_wealth_with_price_variance",
             "type": float},

            {"title": "risk (train)",
             "key": "train_risk",
             "type": float},

            {"title": "risk (test)",
             "key": "test_risk",
             "type": float},

            {"title": "price",
             "key": "price",
             "type": float},

            {"title": "training time",
             "key": "train_time",
             "type": float},

            {"title": "trainable vars",
             "key": "trainable_variables",
             "type": int},

            {"title": "non-trainable vars",
             "key": "non_trainable_variables",
             "type": int}
            ]

        dictionary = {}
        for col in columns:
            dictionary[col["title"]] = [col["type"](case[col["key"]]) \
                                        for case in self.testcases]

        df = pd.DataFrame(dictionary)

        appendum = "\n\n" + f"payoff: {self.test_mean_payoff: .6f}"

        if file_name is not None:
            df.to_csv(file_name, index=False)

            with open(file_name, "a") as file:
                file.write(appendum)

        with pd.option_context('display.max_columns', None):
            print(df)


    def plot_distributions(self, file_name=None, legend_loc="right"):
        self.assert_all_tested()
        raw_data = self.sample(int(2**14))

        v, c, w = [], [], []
        for case in self.testcases:
            input_data = self.get_input(case, raw_data)
            value, costs, wealth, risk = self.get_risk(case, input_data)
            v.append(value)
            c.append(costs)
            w.append(wealth)

        plot_data = [v, c, w] if self.cost is not None else [v, w]
        plot_name = ["value", "costs", "wealth"] if self.cost is not None \
            else ["value", "wealth"]

        for lst, name in zip(plot_data, plot_name):
            plt.figure()
            for data in lst:
                sns.kdeplot(data.numpy(), shade=True)

            plt.legend([case["name"] for case in self.testcases],
                       loc=legend_loc)
            plt.xlabel(name)

            if file_name is not None:
                plt.savefig(fr"{file_name}-{name}.pdf", dpi=DPI)
            else:
                plt.show()


    def plot_scatter_payoff(self, file_name=None):
        self.assert_all_tested()
        raw_data = self.sample(int(2**14))

        for case in self.testcases:
            plt.figure()
            value, costs = self.evaluate_case(case, raw_data)

            x = raw_data["payoff"].numpy()
            y = (value - costs).numpy()
            # beta, alpha = np.polyfit(x, y, 1)
            alpha = y.mean() - x.mean()

            plt.scatter(x, y, s=0.5)
            plt.plot(x, alpha + x, "--", color="black")

            plt.show()


def driver_data_dumb(list_of_drivers, result_keys, file_name):
    for idx in range(len(list_of_drivers[0].testcases)):
        with open(file_name, "a") as file:
            name = list_of_drivers[0].testcases[idx]["name"]
            file.write("".ljust(80, "=") + "\n")
            file.write("=== " + name + "\n")

        dict_of_results = {key: [] for key in result_keys}

        for driver in list_of_drivers:
            case = driver.testcases[idx]

            for key in result_keys:
                dict_of_results[key].append(case[key])

        for key in result_keys:
            with open(file_name, "a") as file:
                file.write(key + "\n")
            pd.DataFrame(dict_of_results[key]).to_csv(
                file_name,
                header=False,
                index=False,
                mode="a"
            )

        with open(file_name, "a") as file:
            file.write("\n\n")


def plot_markovian_payoff(driver, size, price, file_name=None):
    raw_data = driver.sample(size)
    payoff = raw_data["payoff"]

    terminal_spot = raw_data["instruments"][:, 0, -1]
    key = tf.argsort(terminal_spot)

    for case in driver.testcases:
        value, _ = driver.evaluate_case(case, raw_data)

        plt.figure()
        plt.scatter(terminal_spot.numpy(), (value + price).numpy(), s=0.5)
        plt.plot(tf.gather(terminal_spot, key).numpy(),
                 tf.gather(payoff, key).numpy(), color="black")
        if file_name is not None:
            # must be png, as eps/pdf too heavy
            plt.savefig(fr"{file_name}-{case['name']}.png", dpi=DPI)
        else:
            plt.show()


def plot_geometric_payoff(driver, size, file_name=None):
    raw_data = driver.sample(size)
    payoff = raw_data["payoff"]

    derivative = driver.book.derivatives[0]["derivative"]
    terminal_spot = derivative._dga(
        raw_data["time"], raw_data["instruments"][:, 0, :])[..., -1]
    key = tf.argsort(terminal_spot)

    for case in driver.testcases:
        value, _ = driver.evaluate_case(case, raw_data)

        plt.figure()
        plt.scatter(terminal_spot.numpy(), (value + case["price"]).numpy(),
                    s=0.5)
        plt.plot(tf.gather(terminal_spot, key).numpy(),
                 tf.gather(payoff, key).numpy(), color="black")
        if file_name is not None:
            # must be png, as eps/pdf too heavy
            plt.savefig(fr"{file_name}-{case['name']}.png", dpi=DPI)
        else:
            plt.show()


def plot_univariate_hedge_ratios(driver, size, file_name=None):
    raw_data = driver.sample(size)

    for case in driver.testcases[0]:
        input_data = driver.get_input(case, raw_data)
        ratios = case["model"].hedge_ratios(input_data)[0]
        vmin, vmax = float(tf.reduce_min(ratios - raw_data[4][..., :-1])), \
            float(tf.reduce_max(ratios - raw_data[4][..., :-1]))

        for idx in range(15 - 1):
            x1 = raw_data[1][:, 0, idx]
            x2 = ratios[:, 0, idx - 1]
            y = ratios[:, 0, idx] - raw_data[4][:, 0, idx]

            plt.figure()
            plt.scatter(x1.numpy(), x2.numpy(), c=y.numpy(), s=0.5)
            plt.clim(vmin, vmax)
            plt.colorbar()
            plt.ioff()
            plt.savefig(fr"{file_name}-{case['name']}-{idx}.png", dpi=DPI)
            plt.close()


def plot_univariate_barrier_payoff(driver, size, price, file_name=None):
    def metric(time, instruments, numeraire):
        derivative = driver.book.derivatives[0]["derivative"]

        return derivative.crossed(instruments[:, 0, :])[..., -1]

    raw_data = driver.sample(size, metrics=[metric])
    crossed = tf.squeeze(raw_data["metrics"][0])

    terminal_spot = raw_data["instruments"][:, 0, -1]

    for case in driver.testcases:
        value, _ = driver.evaluate_case(case, raw_data)

        for idx, indices in enumerate([crossed, ~crossed]):
            m = tf.boolean_mask(terminal_spot, indices, 0)

            key = tf.argsort(m, 0)
            x = tf.gather(m, key)

            y1 = tf.gather(raw_data["payoff"][indices], key)
            y2 = tf.gather(tf.boolean_mask(value + price, indices), key)

            plt.figure()
            # plt.xlim(*xlim)
            plt.scatter(x, y2.numpy(), s=0.5)
            plt.plot(x, y1.numpy(), color="black")

            if file_name is not None:
                append = "crossed" if idx == 0 else "non-crossed"
                plt.savefig(fr"{file_name}-{case['name']}-{append}.png",
                            dpi=DPI)
            else:
                plt.show()
            plt.close()


def plot_correlation_matrix(driver):
    corr = driver.book.instrument_simulator.correlation
    plt.matshow(corr)
    plt.colorbar()
    plt.show()
