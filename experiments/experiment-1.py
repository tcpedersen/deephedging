# -*- coding: utf-8 -*-
import hedge_models
import utils
import approximators
import preprocessing
import books

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**20), int(2**20), 14
alpha = 0.95
num_layers, num_units = 2, 15

# ==============================================================================
# ===
init_instruments, init_numeraire, book = books.random_put_call_book(
    timesteps / 250, 25, 10, 10, 69)

driver = utils.Driver(
    timesteps=timesteps,
    frequency=0, # no need for frequency for non-path dependent derivatives.
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=True,
    learning_rate=1e-2
    )

driver.add_testcase("shallow network",
                    hedge_models.SimpleHedge(
                        timesteps,
                        book.instrument_dim,
                        num_layers,
                        num_units),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase("deep network",
                    hedge_models.SimpleHedge(
                        timesteps,
                        book.instrument_dim,
                        num_layers * 2,
                        num_units),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=preprocessing.MeanVarianceNormaliser(),
                    feature_type="log_martingale",
                    price_type="indifference")

driver.add_testcase("identity feature map",
                    hedge_models.LinearFeatureHedge(
                        timesteps,
                        book.instrument_dim,
                        [approximators.IdentityFeatureMap] * (1 + int(driver.cost is not None))
                    ),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="indifference")

driver.add_testcase("continuous-time",
                    hedge_models.DeltaHedge(
                        timesteps,
                        book.instrument_dim),
                    risk_measure=hedge_models.ExpectedShortfall(alpha),
                    normaliser=None,
                    feature_type="delta",
                    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.SimpleHedge(
            timesteps,
            book.instrument_dim,
            num_layers,
            num_units),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="log_martingale")

driver.train(train_size, 1000, int(2**12))
driver.test(test_size)
driver.test_summary()
driver.plot_distributions()

if book.book_size == 1:
    utils.plot_markovian_payoff(driver, test_size)
