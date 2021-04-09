# -*- coding: utf-8 -*-
import tensorflow as tf

import hedge_models
import utils
import approximators
import preprocessing
import books

# ==============================================================================
folder_name = r"figures\bin"

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 14
hedge_multiplier = 1
alpha = 0.95

shallow_layers = 2
shallow_units = 20
deep_layers = 4
deep_units = 5

activation = tf.keras.activations.softplus

# ==============================================================================
# ===
init_instruments, init_numeraire, book = books.random_put_call_book(
    timesteps / 250, 25, 10, 10, 69)
# init_instruments, init_numeraire, book = books.simple_put_call_book(
#     timesteps / 250, 100, 100, 0.02, 0.05, 0.2, 1)


driver = utils.HedgeDriver(
    timesteps=timesteps * hedge_multiplier,
    frequency=0, # no need for frequency for non-path dependent derivatives.
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1/100,
    risk_neutral=False,
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
    feature_type="log_martingale",
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
    feature_type="log_martingale",
    price_type="indifference")

driver.add_testcase(
    "identity feature map",
    hedge_models.LinearFeatureHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        mappings=[approximators.IdentityFeatureMap] \
            * (1 + (driver.cost is not None))),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="indifference")

driver.add_testcase(
    "continuous-time",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
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
        feature_type="log_martingale")

driver.train(train_size, 1000, int(2**10))
driver.test(test_size)
driver.test_summary(fr"{folder_name}\test-summary.txt")
driver.plot_distributions(fr"{folder_name}\hist", "upper right")

if book.book_size == 1:
    utils.plot_markovian_payoff(
        driver,
        test_size,
        driver.testcases[-1]["price"],
        fr"{folder_name}\payoff"
        )
