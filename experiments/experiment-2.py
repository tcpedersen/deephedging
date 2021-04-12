# -*- coding: utf-8 -*-
import tensorflow as tf

import hedge_models
import books
import utils
import preprocessing
import approximators

# ==============================================================================
folder_name = r"figures\discrete-multivariate\cost"
activation = tf.keras.activations.softplus

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 12
hedge_multiplier = 1
alpha = 0.95

shallow_layers = 2
shallow_units = 15
deep_layers = 4
deep_units = 5

# ==============================================================================
# === setup
# init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
#     timesteps / 12, 100, 100**(timesteps / 12), 0.02, 0.05, 0.4, 1)
init_instruments, init_numeraire, book = books.random_dga_putcall_book(
    timesteps / 12, 25, 10, 10, 69)

driver = utils.HedgeDriver(
    timesteps=timesteps * hedge_multiplier,
    frequency=0, # no need for frequency
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1/100,
    risk_neutral=False,
    learning_rate=1e-1
    )

# driver.add_testcase(
#     "shallow network",
#     hedge_models.NeuralHedge(
#         timesteps=timesteps * hedge_multiplier,
#         instrument_dim=book.instrument_dim,
#         internal_dim=0,
#         num_layers=shallow_layers,
#         num_units=shallow_units,
#         activation=activation),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale",
#     price_type="indifference")

# driver.add_testcase(
#     "deep network",
#     hedge_models.NeuralHedge(
#         timesteps=timesteps * hedge_multiplier,
#         instrument_dim=book.instrument_dim,
#         internal_dim=0,
#         num_layers=deep_layers,
#         num_units=deep_units,
#         activation=activation),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale",
#     price_type="indifference")

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

# driver.add_testcase(
#     "deep memory network",
#     hedge_models.NeuralHedge(
#         timesteps=timesteps * hedge_multiplier,
#         instrument_dim=book.instrument_dim,
#         internal_dim=book.instrument_dim,
#         num_layers=deep_layers,
#         num_units=deep_units,
#         activation=activation),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale",
#     price_type="indifference")

# driver.add_testcase(
#     "lstm",
#     hedge_models.LSTMHedge(
#         timesteps=timesteps * hedge_multiplier,
#         instrument_dim=book.instrument_dim,
#         lstm_cells=2,
#         lstm_units=15),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale_with_time",
#     price_type="indifference"
#     )

# driver.add_testcase(
#     name="identity feature map",
#     model=hedge_models.LinearFeatureHedge(
#         timesteps * hedge_multiplier,
#         book.instrument_dim,
#         [approximators.IdentityFeatureMap] * (1 + int(driver.cost is not None))
#     ),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=None,
#     feature_function="delta",
#     price_type="indifference")

# driver.add_testcase(
#     name="continuous-time",
#     model=hedge_models.DeltaHedge(
#         timesteps * hedge_multiplier,
#         book.instrument_dim),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=None,
#     feature_function="delta",
#     price_type="arbitrage")

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
driver.test_summary(fr"{folder_name}\test-summary-extra.txt")
# driver.plot_distributions(fr"{folder_name}\hist", "upper right")
