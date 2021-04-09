# -*- coding: utf-8 -*-
import books
import utils
import hedge_models

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**16), int(2**16), 14
frequency = 0 # only > 0 for continuous, else 0
hedge_multiplier = 2**7
alpha = 0.99

folder_name = r"figures\bin"

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = books.simple_put_call_book(
    timesteps / 250, 100., 100, 0.02, 0.05, 0.2, 1.)
# init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
#     timesteps / 250, 100., 100**(timesteps / 250), 0.02, 0.05, 0.2, 1.)
# init_instruments, init_numeraire, book = books.simple_barrier_book(
#     timesteps / 250, 100, 105, 95, 0.02, 0.05, 0.2, 1, -1)

driver = utils.HedgeDriver(
    timesteps=timesteps * hedge_multiplier,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=None,
    risk_neutral=True,
    learning_rate=1e-1
    )

driver.add_testcase(
    "delta",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="arbitrage")

driver.train(train_size, 100, 2**12)
driver.test(test_size)
driver.test_summary()

full_folder_name = fr"{folder_name}/payoff-{hedge_multiplier}-{frequency}"

utils.plot_markovian_payoff(driver, test_size, driver.testcases[0]["price"])
# utils.plot_geometric_payoff(driver, test_size)
# utils.plot_univariate_barrier_payoff(driver, test_size, full_folder_name)
