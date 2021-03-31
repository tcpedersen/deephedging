# -*- coding: utf-8 -*-
from books import simple_put_call_book
import utils
import hedge_models

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**20), int(2**20), 14
hedge_multiplier = 1
alpha = 0.95

folder_name = r"figures\markovian-univariate\delta"

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = simple_put_call_book(
    timesteps / 250, 100., 100., 0.02, 0.05, 0.2, 1.)

driver = utils.Driver(
    timesteps=timesteps * hedge_multiplier,
    frequency=0, # no need for frequency
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=False,
    learning_rate=1e-1
    )

driver.add_testcase(
    "delta",
    hedge_models.DeltaHedge(
        timesteps=timesteps * hedge_multiplier,
        instrument_dim=len(init_instruments)
    ),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="arbitrage")

driver.train(train_size, 1, 1024)
driver.test(test_size)
utils.plot_markovian_payoff(driver, test_size, fr"{folder_name}/payoff-{hedge_multiplier}")
