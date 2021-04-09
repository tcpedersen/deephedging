# -*- coding: utf-8 -*-
import tensorflow as tf

import utils
import hedge_models
import gradient_models
import gradient_driver
import books
import preprocessing

train_size, test_size, timesteps = int(2**12), int(2**18), 14
init_instruments, init_numeraire, book = books.simple_put_call_book(
    timesteps / 12, 100, 100, 0.02, 0.05, 0.2, 1)
multiplier = 2**6

# ==============================================================================
# === train gradient models
layers = 4
units = 5
activation = tf.keras.activations.softplus

warmup_driver = gradient_driver.GradientDriver(
    timesteps=timesteps * multiplier,
    frequency=0,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-5,
    learning_rate_max=1e-2
    )

warmup_driver.set_exploration(100., 25.)

warmup_driver.add_testcase(
    name="twin network",
    model=gradient_models.SequenceTwinNetwork(
        layers=layers,
        units=units,
        activation=activation
        ),
    normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
    )

warmup_driver.train(train_size, 100, 32)
# warmup_driver.test(test_size)
# gradient_driver.markovian_visualiser(warmup_driver, test_size)

# =============================================================================
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
    "continuous-time",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="arbitrage")

for case in warmup_driver.testcases:
    driver.add_testcase(
        "delta regularisation " + case["name"],
        hedge_models.FeatureHedge(),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=None,
        feature_function=warmup_driver.make_feature_function(case),
        price_type="arbitrage")

driver.train(train_size, 1000, int(2**10))
driver.test(test_size)
driver.test_summary()

utils.plot_markovian_payoff(driver, int(2**14), driver.testcases[0]["price"])