# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import erfinv
from tensorflow_probability.python.internal import special_math

from constants import FLOAT_DTYPE_EPS, FLOAT_DTYPE
import hedge_models

ONE_OVER_SQRT_TWO_PI = 1. / tf.sqrt(2. * np.pi)
SQRT_TWO = tf.sqrt(2.)

# ==============================================================================
# === Gaussian
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * tf.exp(-0.5 * x * x)

def norm_cdf(x):
    return special_math.ndtr(x)

def norm_qdf(x):
    return erfinv(2. * x - 1.) * SQRT_TWO

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
def precise_mean(x, **kwargs):
    return tf.cast(tf.reduce_mean(tf.cast(x, tf.float64)), x.dtype)


# ==============================================================================
# === experiments
class Driver(object):
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

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            verbose=1,
            patience=2
        )

        self.callbacks = [early_stopping, reduce_lr]
        self.testcases = []
        self.liability_free = None


    def sample(self, size):
        time, instruments, numeraire = self.book.sample_paths(
            self.init_instruments,
            self.init_numeraire,
            size,
            self.timesteps * (self.frequency + 1),
            self.risk_neutral)

        skip = self.frequency + 1
        value = self.book.value(time, instruments, numeraire)[...,::skip]
        delta = self.book.delta(time, instruments, numeraire)[...,::skip]
        payoff = self.book.payoff(time, instruments, numeraire)

        return time[::skip], instruments[..., ::skip], numeraire[::skip], \
            value, delta, payoff


    def validate_feature_type(self, feature_type):
        valid_feature_types = ["log_martingale", "delta"]
        assert feature_type in valid_feature_types, \
            f"feature_type '{feature_type}' not in {valid_feature_types}"


    def validate_price_type(self, price_type):
        valid_price_types = ["indifference", "arbitrage"]
        assert price_type in valid_price_types, \
            f"price_type '{price_type}' not in {valid_price_types}"


    def add_testcase(self,
                     name,
                     model,
                     risk_measure,
                     normaliser,
                     feature_type,
                     price_type):
        self.validate_feature_type(feature_type)
        self.validate_price_type(price_type)

        assert issubclass(type(model), hedge_models.Hedge)
        assert issubclass(type(risk_measure), hedge_models.OCERiskMeasure)
        assert model.timesteps == self.timesteps

        if self.cost is not None:
            model.add_cost_layers(self.cost)

        self.testcases.append(
            {"name": str(name),
             "model": model,
             "risk_measure": risk_measure,
             "normaliser": normaliser,
             "feature_type": feature_type,
             "price_type": price_type,
             "trained": False,
             "tested": False}
            )


    def add_liability_free(self, model, risk_measure, normaliser, feature_type):
        self.validate_feature_type(feature_type)
        assert issubclass(type(model), hedge_models.Hedge)
        assert issubclass(type(risk_measure), hedge_models.OCERiskMeasure)

        self.liability_free = {"name": "liability free",
                               "model": model,
                               "risk_measure": risk_measure,
                               "normaliser": normaliser,
                               "feature_type": feature_type,
                               "price": 0.,
                               "trained": False,
                               "tested": False}


    def normalise_features(self, case):
        return case["feature_type"] not in ["delta"]


    def get_input(self, case, raw_data):
        time, instruments, numeraire, value, delta, payoff = raw_data

        if case["feature_type"] == "log_martingale":
            features = tf.math.log(instruments / numeraire)
        elif case["feature_type"] == "delta":
            features = delta * numeraire
        else:
            message = f"feature_type '{case['feature_type']}' is not valid."
            raise NotImplementedError(message)

        if case["normaliser"] is not None:
             if not case["trained"]:
                 case["normaliser"].fit(features)
             features = case["normaliser"].transform(features)

        return [features, instruments / numeraire, payoff]


    def get_risk(self, case, input_data):
        value, costs = case["model"](input_data)
        wealth = value - costs - input_data[-1]

        return value, costs, wealth, case["model"].risk_measure.evaluate(wealth)


    def train_case(self, case, input_data, **kwargs):
        optimizer = self.optimizer(self.learning_rate)
        case["model"].compile(
            risk_measure=case["risk_measure"],
            optimizer=optimizer)

        case["history"] = case["model"].fit(
            input_data, callbacks=self.callbacks, **kwargs)
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
        case["test_value"] = precise_mean(value)
        case["test_costs"] = precise_mean(costs)
        case["test_wealth"] = precise_mean(wealth)
        case["test_risk"] = risk

        case["tested"] = True


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
            case["price"] = self.get_price(case)


    def get_price(self, case):
        self.assert_case_is_trained(case)
        self.assert_case_is_tested(case)

        if case["price_type"] == "arbitrage":
            _, _, _, value, _, _ = self.sample(1)
            return value[0, 0]
        elif case["price_type"] == "indifference":
            if self.cost is None and self.risk_neutral:
                liability_free_risk = 0.
            else:
                liability_free_risk = self.liability_free["test_risk"]

            return case["test_risk"] - liability_free_risk


    def test_summary(self, file_name=None):
        self.assert_all_tested()

        block_size = 17
        case_size = max([len(case["name"]) for case in self.testcases]) + 1

        header_titles = ["value", "costs", "wealth",
                         "risk (train)", "risk (test)", "price"]
        header = "".rjust(case_size) + "".join(
            [ht.rjust(block_size) for ht in header_titles])
        body = ""
        print_keys = ["test_value", "test_costs", "test_wealth",
                      "train_risk", "test_risk"]

        extra = [self.liability_free] if self.liability_free is not None else []
        for case in self.testcases + extra:
            body += case["name"].ljust(case_size)
            for val in [case[key] for key in print_keys]:
                body += f"{val:.4f}".rjust(block_size)

            price = case["price"]
            body += f"{price: .4f}".rjust(block_size)

            body += "\n"

        summary = header + "\n" + body

        if file_name is not None:
            with open(file_name, "w") as file:
                file.write(summary)

        print(summary)


    def plot_distributions(self, file_name=None, legend_loc="right"):
        self.assert_all_tested()
        raw_data = self.sample(int(2**18))

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
                plt.savefig(fr"{file_name}-{name}.pdf")
            else:
                plt.show()


def plot_markovian_payoff(driver, size, file_name=None):
    raw_data = driver.sample(size)
    payoff = raw_data[-1]

    terminal_spot = raw_data[1][:, 0, -1]
    key = tf.argsort(terminal_spot)

    for case in driver.testcases:
        input_data = driver.get_input(case, raw_data)
        value, _ = case["model"](input_data)

        plt.figure()
        plt.scatter(terminal_spot.numpy(), (value + case["price"]).numpy(), s=0.5)
        plt.plot(tf.gather(terminal_spot, key).numpy(),
                 tf.gather(payoff, key).numpy(), color="black")
        if file_name is not None:
            plt.savefig(fr"{file_name}-{case['name']}.png") # must be png, as eps/pdf too heavy
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
            plt.savefig(fr"{file_name}-{case['name']}-{idx}.png")
            plt.close()


def plot_barrier_payoff(
        model, norm_inputs, price, time, instruments, numeraire, book):
    derivative = book.derivatives[0]["derivative"]
    crossed = tf.squeeze(tf.reduce_any(derivative.crossed(instruments), 2))
    payoff = book.payoff(time, instruments, numeraire)
    xlim = (tf.reduce_min(instruments[:, 0, -1]),
            tf.reduce_max(instruments[:, 0, -1]))

    value, costs = model(norm_inputs)

    for indices in [crossed, ~crossed]:
        m = tf.boolean_mask(instruments[..., 0, -1], indices, 0)

        key = tf.argsort(m, 0)
        x = tf.gather(m, key)

        y1 = tf.gather(payoff[indices], key)
        y2 = tf.gather(tf.boolean_mask(price + value, indices), key)

        plt.figure()
        plt.xlim(*xlim)
        plt.scatter(x, y2, s=0.5)
        plt.plot(x, y1, color="black")
        plt.show()


def plot_correlation_matrix(corr):
    plt.matshow(corr)
    plt.colorbar()
    plt.show()