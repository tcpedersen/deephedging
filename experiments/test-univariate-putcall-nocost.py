# -*- coding: utf-8 -*-
import tensorflow as tf

from books import simple_put_call_book
import utils
import hedge_models

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**20), int(2**20), 30
alpha = 0.95


# ==============================================================================
# === sample data
init_instruments, init_numeraire, book = simple_put_call_book(
    1., 100., 105., 0.05, 0.1, 0.2, 1.)

driver = utils.HedgeDriver(
    timesteps=timesteps,
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
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="arbitrage")

driver.train(train_size, 1, int(2**10))
driver.test(test_size)

# ==============================================================================
# ====
# https://www.dropbox.com/s/g1elm90a6i6x79n/hedge_scatter.R?dl=0

raw_data = driver.sample(test_size)
input_data = driver.get_input(driver.testcases[0], raw_data)
numeraire = raw_data["numeraire"]

# hedge_ratios = driver.testcases[0]["model"].hedge_ratios(input_data)[0]
# initial_hedge_ratio = hedge_ratios[0, 0, 0]
# print(f"initial hedge ratio: {initial_hedge_ratio:.4f}, should be {0.5422283:.4f}.")

price = driver.testcases[0]["price"]
print(f"initial investment: {price * numeraire[0]:.4f}, should be {8.0214:.4f}.")

payoff = tf.reduce_mean(raw_data["payoff"] * numeraire[0])
print(f"average discounted option payoff: {payoff:.4f}, should be {11.0641:.4f}.")

hedge_wealth = (price + driver.testcases[0]["test_value"]) * numeraire[0]
print(f"average discounted portfolio value: {hedge_wealth:.4f}, should be {11.0559:.4f} ({11.02643:.4f}, {11.08537:.4f}).")
