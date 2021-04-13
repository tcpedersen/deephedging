# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

import hedge_models
import books
import utils
import preprocessing
import approximators


# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 12
hedge_multiplier = 1 if str(sys.argv[3]) == "normal" else 5
alpha = 0.95

if str(sys.argv[1]) == "univariate":
    units = 3
elif str(sys.argv[1]) == "multivariate":
    units = 10
else:
    raise ValueError

shallow_layers = 2
shallow_units = units
deep_layers = 4
deep_units = units

activation = tf.keras.activations.softplus

# ==============================================================================
# === setup
if str(sys.argv[1]) == "univariate":
    init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
        timesteps / 12, 100, 100**(timesteps / 12), 0.02, 0.05, 0.4, 1)
elif str(sys.argv[1]) == "multivariate":
    init_instruments, init_numeraire, book = books.random_dga_putcall_book(
        timesteps / 12, 25, 10, 10, 69)
else:
    raise ValueError

driver = utils.HedgeDriver(
    timesteps=timesteps * hedge_multiplier,
    frequency=0, # no need for frequency
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1/100 if str(sys.argv[2]) == "cost" else None,
    risk_neutral=False if str(sys.argv[2]) == "cost" else True,
    learning_rate=1e-1
    )

# === set folder name
outer_folder = f"discrete-{'multivariate' if book.instrument_dim > 1 else 'univariate'}"
if driver.cost is not None:
    inner_folder = "cost"
elif hedge_multiplier > 1:
    inner_folder = "no-cost-frequent"
else:
    inner_folder = "no-cost"

folder_name = fr"figures\{outer_folder}\{inner_folder}"

# =============================================================================
# === train
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
        num_units=shallow_units * 2,
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
#         num_units=deep_units * 2,
#         activation=activation),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale",
#     price_type="indifference")

# driver.add_testcase(
#     "lstm",
#     hedge_models.RecurrentHedge(
#         timesteps=timesteps * hedge_multiplier,
#         rnn=tf.keras.layers.LSTM,
#         instrument_dim=book.instrument_dim,
#         cells=2,
#         units=15),
#     risk_measure=hedge_models.ExpectedShortfall(alpha),
#     normaliser=preprocessing.MeanVarianceNormaliser(),
#     feature_function="log_martingale_with_time",
#     price_type="indifference"
#     )
#
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
#
# driver.add_testcase(
#     name="continuous-time",
#     model=hedge_models.FeatureHedge(),
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
driver.test_summary(fr"{folder_name}\test-summary-shallow-memory-extra.txt")
# driver.plot_distributions(fr"{folder_name}\hist", "upper right")
