# -*- coding: utf-8 -*-
import tensorflow as tf

import hedge_models
import books
import utils
import preprocessing
import approximators

# ==============================================================================
folder_name = r"figures\continuous-multivariate\no-cost"
activation = tf.keras.activations.softplus

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 14
hedge_multiplier = 2**0
frequency = 4 - 0
lookback = 2 if frequency > 2 else 1
alpha = 0.95

shallow_layers, shallow_units = 2, 15
deep_layers, deep_units = 4, 5

# ==============================================================================
# === setup
init_instruments, init_numeraire, book = books.random_barrier_book(
    timesteps / 250, 25, 10, 10, 69)

# init_instruments, init_numeraire, book = books.simple_barrier_book(
#     timesteps / 250, 100, 105, 95, 0.02, 0.05, 0.2, "out", "down")

driver = utils.HedgeDriver(
    timesteps=timesteps * hedge_multiplier,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=True,
    learning_rate=1e-1
    )

driver.add_testcase(
    "shallow network",
    hedge_models.NeuralHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=0,
        num_layers=shallow_layers,
        num_units=shallow_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale",
    price_type="indifference")

driver.add_testcase(
    "deep network",
    hedge_models.NeuralHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=0,
        num_layers=deep_layers,
        num_units=deep_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale",
    price_type="indifference")

driver.add_testcase(
    "shallow memory network",
    hedge_models.NeuralHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=shallow_layers,
        num_units=shallow_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale",
    price_type="indifference")

driver.add_testcase(
    "deep memory network",
    hedge_models.NeuralHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=deep_layers,
        num_units=deep_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale",
    price_type="indifference")


driver.add_testcase(
    "lstm",
    hedge_models.RecurrentHedge(
        timesteps=timesteps * hedge_multiplier,
        rnn=tf.keras.layers.LSTM,
        instrument_dim=book.instrument_dim,
        cells=2,
        units=15),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale_with_time",
    price_type="indifference"
    )


driver.add_testcase(
    "gru",
    hedge_models.RecurrentHedge(
        timesteps=timesteps * hedge_multiplier,
        rnn=tf.keras.layers.GRU,
        instrument_dim=book.instrument_dim,
        cells=2,
        units=15),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale_with_time",
    price_type="indifference"
    )


driver.add_testcase(
    name="identity feature map",
    model=hedge_models.LinearFeatureHedge(
        timesteps * hedge_multiplier,
        book.instrument_dim,
        [approximators.IdentityFeatureMap] * (1 + int(driver.cost is not None))
    ),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="indifference")

driver.add_testcase(
    name="continuous-time",
    model=hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.LinearFeatureHedge(
            timesteps=timesteps * hedge_multiplier,
            instrument_dim=book.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] \
                * (1 + (driver.cost is not None))),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale")

driver.train(train_size, 1000, int(2**10))
driver.test(test_size)
driver.test_summary(fr"{folder_name}\test-summary.txt")

if not book.instrument_dim > 1:
    driver.plot_distributions(fr"{folder_name}\hist", "upper right")

    utils.plot_univariate_barrier_payoff(
        driver,
        test_size,
        driver.testcases[-1]["price"],
        file_name=fr"{folder_name}\payoff")
