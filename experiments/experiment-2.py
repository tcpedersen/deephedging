# -*- coding: utf-8 -*-
import hedge_models
import books
import utils
import preprocessing
import approximators

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**18), 14
frequency = 10
alpha = 0.95

num_layers, num_units = 2, 15

# ==============================================================================
# === setup
init_instruments, init_numeraire, book = books.random_geometric_asian_book(
    timesteps / 250, 25, 10, 10, 72)

driver = utils.Driver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=1. / 100,
    risk_neutral=False,
    learning_rate=1e-2
    )

driver.add_testcase(
    name="shallow memory network",
    model=hedge_models.MemoryHedge(
        timesteps=timesteps,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=num_layers,
        num_units=num_units),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_type="log_martingale",
    price_type="indifference"
    )

driver.add_testcase(
    name="deep memory network",
    model=hedge_models.MemoryHedge(
        timesteps=timesteps,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=num_layers * 2,
        num_units=num_units),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_type="log_martingale",
    price_type="indifference"
    )

driver.add_testcase(
    name="identity feature map",
    model=hedge_models.LinearFeatureHedge(
        timesteps,
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
            num_layers=num_layers,
            num_units=num_units),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="log_martingale")

driver.train(train_size, 100, int(2**10))
driver.test(test_size)
driver.test_summary()
driver.plot_distributions()
