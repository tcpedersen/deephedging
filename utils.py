# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import erfinv
from tensorflow_probability.python.internal import special_math

from constants import FLOAT_DTYPE_EPS, FLOAT_DTYPE
import models

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
def expected_shortfall(wealth, alpha):
    """Emperical expected shortfall."""
    loss = -wealth
    var = np.quantile(loss, alpha)
    return tf.reduce_mean(loss[loss > var])


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
                 cost,
                 risk_neutral):
        self.timesteps = timesteps
        self.frequency = frequency
        self.init_instruments = init_instruments
        self.init_numeraire = init_numeraire
        self.book = book
        self.cost = cost
        self.risk_neutral = risk_neutral

        self.learning_rate = 1e-2
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
            self.timesteps * self.frequency,
            self.risk_neutral)

        return time, instruments, numeraire


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

        assert issubclass(type(model), models.Hedge)
        assert issubclass(type(risk_measure), models.OCERiskMeasure)

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
        assert issubclass(type(model), models.Hedge)
        assert issubclass(type(risk_measure), models.OCERiskMeasure)

        self.liability_free = {"model": model,
                               "risk_measure": risk_measure,
                               "normaliser": normaliser,
                               "feature_type": feature_type,
                               "trained": False,
                               "tested": False}


    def normalise_features(self, case):
        return case["feature_type"] not in ["delta"]


    def get_input(self, case, raw_data):
        time, instruments, numeraire = raw_data

        if case["feature_type"] == "log_martingale":
            features = tf.math.log(instruments / numeraire)
        elif case["feature_type"] == "delta":
            features = self.book.delta(time, instruments, numeraire) * numeraire
        else:
            message = f"feature_type '{case['feature_type']}' is not valid."
            raise NotImplementedError(message)

        if case["normaliser"] is not None:
             if not case["trained"]:
                 case["normaliser"].fit(features)
             features = case["normaliser"].transform(features)

        martingales = instruments / numeraire
        payoff = self.book.payoff(time, instruments, numeraire)

        return [features, martingales, payoff]


    @tf.function
    def get_risk(self, case, input_data):
        value, costs = case["model"](input_data)
        wealth = value - costs - input_data[-1]

        return value, costs, wealth, case["model"].risk_measure(wealth)


    def train_case(self, case, input_data, **kwargs):
        if self.cost is not None:
            case["model"].add_cost_layers(self.cost)

        optimizer = self.optimizer(self.learning_rate)
        case["model"].compile(case["risk_measure"],
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


    def assert_all_tested(self):
        for case in self.testcases:
            self.assert_case_is_tested(case)


    def test(self, sample_size):
        self.assert_all_trained()

        raw_data = self.sample(sample_size)
        for case in self.testcases:
            input_data = self.get_input(case, raw_data)
            self.test_case(case, input_data)

        if self.liability_free is not None:
            input_data = self.get_input(self.liability_free, raw_data)
            input_data[-1] = tf.zeros_like(input_data[-1])
            self.test_case(self.liability_free, input_data)


    def get_price(self, case):
        self.assert_case_is_trained(case)
        self.assert_case_is_tested(case)

        time, instruments, numeraire = self.sample(1)

        if case["price_type"] == "arbitrage":
            return self.book.value(time, instruments, numeraire)[0, 0]
        elif case["price_type"] == "indifference":
            if self.cost is None and self.risk_neutral:
                liability_free_risk = 0.
            else:
                liability_free_risk = self.liability_free["test_risk"]

            case_risk = case["test_risk"]
            return numeraire[0] * (case_risk - liability_free_risk)


    def test_summary(self):
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

        for case in self.testcases:
            body += case["name"].ljust(case_size)
            for val in [case[key] for key in print_keys]:
                body += f"{val:.4f}".rjust(block_size)

            price = self.get_price(case)
            body += f"{price: .4f}".rjust(block_size)

            body += "\n"

        print(header + "\n" + body)


    def plot_distributions(self):
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
        for lst in plot_data:
            plt.figure()
            for data in lst:
                sns.distplot(data.numpy(), hist=False,
                             kde_kws={'shade': True, 'linewidth': 3})
            plt.legend([case["name"] for case in self.testcases])
            plt.show()


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
    plt.figure()
    plt.matshow(corr)
    plt.colorbar()
    plt.show()