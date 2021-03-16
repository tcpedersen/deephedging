# -*- coding: utf-8 -*-
import hedge_models
import utils
import approximators
import preprocessing
import books

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**20), 14
frequency = 1
alpha = 0.95
num_layers, num_units = 2, 15

# ==============================================================================
# ===
init_instruments, init_numeraire, book = books.simple_put_call_book(
    timesteps / 250, 100, 100, 0.02, 0.05, 0.2, 1)

driver = utils.Driver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=2.5 / 100,
    risk_neutral=False,
    learning_rate=1e-2
    )

driver.add_testcase("shallow network",
                    hedge_models.SimpleHedge(
                        timesteps * frequency,
                        book.instrument_dim,
                        num_layers,
                        num_units),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase("identity feature map",
                    hedge_models.LinearFeatureHedge(
                        timesteps * frequency,
                        book.instrument_dim,
                        [approximators.IdentityFeatureMap,
                         approximators.IdentityFeatureMap]),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="indifference")

driver.add_testcase("continuous-time",
                    hedge_models.DeltaHedge(
                        timesteps * frequency,
                        book.instrument_dim),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.SimpleHedge(
            timesteps * frequency,
            book.instrument_dim,
            num_layers,
            num_units),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="log_martingale")

driver.train(train_size, 100, int(2**12))
driver.test(test_size)
driver.test_summary()
driver.plot_distributions()
utils.plot_markovian_payoff(driver, test_size)

