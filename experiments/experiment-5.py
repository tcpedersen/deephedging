# -*- coding: utf-8 -*-
import tensorflow as tf

import gradient_models
import gradient_driver
import hedge_models
import books
import derivatives
import utils
import preprocessing
import approximators

train_size, test_size, timesteps = int(2**12), int(2**18), 14

init_instruments, init_numeraire, book = books.simple_empty_book(
    timesteps / 250, 100, 0.02, 0.05, 0.2)

spread = 10
itm = derivatives.PutCall(
    book.maturity,
    init_instruments - spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
atm = derivatives.PutCall(
    book.maturity,
    init_instruments,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
otm = derivatives.PutCall(
    book.maturity,
    init_instruments + spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)

book.add_derivative(itm, 0, 1)
book.add_derivative(atm, 0, -2)
book.add_derivative(otm, 0, 1)

multiplier = 1
start_time = 5.

# ==============================================================================
# === train gradient models
folder_name = r"figures\markovian-add\univariate-call-spread"

warmup_driver = gradient_driver.GradientDriver(
    timesteps=timesteps * multiplier,
    frequency=0,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-5,
    learning_rate_max=1e-2
    )

warmup_driver.set_exploration(
    init_instruments,
    book.derivatives[0]["derivative"].volatility * tf.sqrt(start_time))

for layers in [2, 3, 4, 5]:
    for units in [5]:
        warmup_driver.add_testcase(
            name=f"layers {layers} units {units}",
            model=gradient_models.SequenceTwinNetwork(
                layers=layers,
                units=units,
                activation=tf.keras.activations.softplus
                )
            )

warmup_driver.train(train_size, 100, 32)
warmup_driver.test(test_size)
warmup_driver.test_summary(fr"{folder_name}\test-summary-gradient.txt")

# gradient_driver.markovian_visualiser(warmup_driver, test_size)


# ==============================================================================
# === run hedge experiment
train_size, test_size = int(2**18), int(2**18)
alpha = 0.95

driver = utils.HedgeDriver(
    timesteps=timesteps * multiplier,
    frequency=0, # no need for frequency for non-path dependent derivatives.
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=False,
    learning_rate=1e-1
    )

driver.add_testcase(
    "deep network",
    hedge_models.NeuralHedge(
        timesteps=timesteps * multiplier,
        instrument_dim=book.instrument_dim,
        internal_dim=0,
        num_layers=4,
        num_units=5,
        activation=tf.keras.activations.softplus),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_function="log_martingale",
    price_type="indifference")

driver.add_testcase(
    "identity feature map",
    hedge_models.LinearFeatureHedge(
        timesteps=timesteps * multiplier,
        instrument_dim=book.instrument_dim,
        mappings=[approximators.IdentityFeatureMap] \
            * (1 + (driver.cost is not None))),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="indifference")

driver.add_testcase(
    "continuous-time",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="arbitrage")

for case in warmup_driver.testcases:
    driver.add_testcase(
        case["name"] + " identity feature map",
        hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function=warmup_driver.make_feature_function(case),
        price_type="indifference")

    driver.add_testcase(
        case["name"] + " linear feature map",
        hedge_models.LinearFeatureHedge(
            timesteps=timesteps * multiplier,
            instrument_dim=book.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] \
                * (1 + (driver.cost is not None))
            ),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function=warmup_driver.make_feature_function(case),
        price_type="indifference")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.LinearFeatureHedge(
            timesteps=timesteps * multiplier,
            instrument_dim=book.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] \
                * (1 + (driver.cost is not None))),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale")

driver.train(train_size, 1000, int(2**10))
driver.test(test_size)
driver.test_summary(fr"{folder_name}\test-summary-hedge.txt")
