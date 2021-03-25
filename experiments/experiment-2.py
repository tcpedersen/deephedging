# -*- coding: utf-8 -*-
import tensorflow as tf

import hedge_models
import books
import utils
import preprocessing
import approximators

# ==============================================================================
folder_name = r"figures\discrete-univariate\no-cost-frequent"
activation = tf.keras.activations.softplus

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 12
hedge_multiplier = 5
alpha = 0.95
num_layers, num_units = 2, 15

# ==============================================================================
# === setup
init_instruments, init_numeraire, book = books.random_dga_putcall_book(
    timesteps / 12, 1, 1, 1, 69)

driver = utils.Driver(
    timesteps=timesteps * hedge_multiplier,
    frequency=0, # no need for frequency
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=True,
    learning_rate=1e-1
    )

driver.add_testcase("shallow network",
                    hedge_models.SimpleHedge(
                        timesteps * hedge_multiplier,
                        book.instrument_dim,
                        num_layers,
                        num_units,
                        activation=activation),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase("deep network",
                    hedge_models.SimpleHedge(
                        timesteps * hedge_multiplier,
                        book.instrument_dim,
                        num_layers * 2,
                        num_units,
                        activation=activation),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase(
    name="shallow memory network",
    model=hedge_models.MemoryHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=num_layers,
        num_units=num_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_type="log_martingale",
    price_type="indifference"
    )

driver.add_testcase(
    name="deep memory network",
    model=hedge_models.MemoryHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=num_layers * 2,
        num_units=num_units,
        activation=activation),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_type="log_martingale",
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
    feature_type="delta",
    price_type="indifference")

driver.add_testcase(
    name="continuous-time",
    model=hedge_models.DeltaHedge(
        timesteps * hedge_multiplier,
        book.instrument_dim),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.SimpleHedge(
            timesteps * hedge_multiplier,
            book.instrument_dim,
            num_layers=1,
            num_units=num_units),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="log_martingale")

driver.train(train_size, 100, int(2**10))
driver.test(test_size)
driver.test_summary(fr"{folder_name}\test-summary.txt")
driver.plot_distributions(fr"{folder_name}\hist", "upper right")
