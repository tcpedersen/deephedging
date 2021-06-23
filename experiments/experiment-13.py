# -*- coding: utf-8 -*-
import random_books
import utils
import hedge_models
import sys

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**2), int(2**14), 7
hedge_multiplier = 5 * int(sys.argv[1])
frequency = 0 # only > 0 for continuous, else 0
alpha = 0.95

folder_name = r"figures\delta-pnl-plots"
case_name = "call"

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = random_books.random_empty_book(
    timesteps / 250, 1, 0.0, 0.05, 0.2, seed=69) # NOTE timesteps / 52 for DGA

# random_books.add_calls(init_instruments, book)
random_books.add_butterfly(init_instruments, book, 10)
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

driver.train(sample_size=train_size, epochs=0, batch_size=2**2)
driver.test(test_size)

print(f"{float(driver.testcases[0]['test_risk']):.10f}    {float(driver.testcases[0]['price']):.10f}")
