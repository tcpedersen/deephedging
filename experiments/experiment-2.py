# -*- coding: utf-8 -*-
import models
import utils
import preprocessing
from books import random_put_call_book

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 30
frequency = 1
alpha = 0.95
num_layers, num_units = 2, 15

# ==============================================================================
# ===
init_instruments, init_numeraire, book = random_put_call_book(
    timesteps / 250, 25, 10, 10, 69)

driver = utils.Driver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=False
    )

# driver.add_testcase("simple network (ZCA)",
#                     models.SimpleHedge(
#                         timesteps, book.instrument_dim, num_layers, num_units),
#                     risk_measure=models.ExpectedShortfall(alpha),
#                     normaliser=preprocessing.ZeroComponentAnalysis(),
#                     feature_type="log_martingale",
#                     price_type="indifference")

driver.add_testcase("simple network (MV)",
                    models.SimpleHedge(
                        timesteps, book.instrument_dim, num_layers, num_units),
                    risk_measure=models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase("linear feature",
                    models.LinearFeatureHedge(
                        timesteps,
                        len(init_instruments),
                        [models.IdentityFeatureMap]),
                    risk_measure=models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="indifference")

driver.add_testcase("RBF feature",
                    models.LinearFeatureHedge(
                        timesteps,
                        len(init_instruments),
                        [models.GaussianFeatureMap]),
                    risk_measure=models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="indifference")

driver.add_testcase("delta",
                    models.DeltaHedge(timesteps, len(init_instruments)),
                    risk_measure=models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        models.SimpleHedge(timesteps, book.instrument_dim, num_layers, num_units),
        risk_measure=models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="delta")

driver.train(train_size, 100, 1024)
driver.test(test_size)
driver.test_summary()
driver.plot_distributions()
