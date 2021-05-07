# -*- coding: utf-8 -*-
import random_books
import utils
import hedge_models

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**16), int(2**16), 14
hedge_multiplier = 2**0
frequency = 0 # only > 0 for continuous, else 0
alpha = 0.95

folder_name = r"figures\delta-pnl-plots"
case_name = "call"

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = random_books.random_empty_book(
    timesteps / 250, 1, 0.02, 0.05, 0.2, seed=69) # NOTE timesteps / 52 for DGA

random_books.add_calls(init_instruments, book)
# random_books.add_dga_calls(init_instruments, book)
# random_books.add_rko(init_instruments, book, 10.0)

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
    case_name,
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function="delta",
    price_type="arbitrage")

driver.train(train_size, 100, 2**12)
driver.test(test_size)
driver.test_summary()

full_folder_name = fr"{folder_name}/payoff-{hedge_multiplier}-{frequency}"

price = driver.testcases[0]["price"]
utils.plot_markovian_payoff(driver, test_size, price, full_folder_name)
# utils.plot_geometric_payoff(driver, test_size, price, full_folder_name)
# utils.plot_univariate_barrier_payoff(driver, test_size, price, full_folder_name)
