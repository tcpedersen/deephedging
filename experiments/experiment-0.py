# -*- coding: utf-8 -*-
import books
import utils
import hedge_models

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**10), int(2**18), 14
hedge_multiplier = 1
alpha = 0.95

folder_name = r"figures\discrete-univariate\delta"

# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
    timesteps / 250, 100., 100**(timesteps / 250), 0.01, 0.05, 0.4, 1.)

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

driver.train(train_size, 1, train_size)
driver.test(test_size)

full_folder_name = fr"{folder_name}/payoff-{hedge_multiplier}"
utils.plot_geometric_payoff(driver, test_size, file_name=full_folder_name)




# time, instruments, numeraire = book.sample_paths(
#     init_instruments, init_numeraire, int(2**24), timesteps, True, True, 0)
# payoff = book.payoff(time, instruments, numeraire)

# tf.reduce_mean(payoff)
# #book.value(time, instruments, numeraire)[0, 0]
