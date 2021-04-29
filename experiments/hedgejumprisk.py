# -*- coding: utf-8 -*-
import tensorflow as tf

import books
import simulators
import derivatives
import utils
import hedge_models

from constants import FLOAT_DTYPE

# ==============================================================================
# === parameters
cost = False
train_size, test_size, timesteps = int(2**10), int(2**10), 14 # TODO change
maturity = timesteps / 250
rate, drift, diffusion = 0.02, [0.05], [[0.15]]
intensity, jumpsize, jumpvol = 1.0, -0.2, 0.15

time = tf.cast(tf.linspace(0., maturity, timesteps + 1), FLOAT_DTYPE)
init_instruments = tf.constant([100.], FLOAT_DTYPE)
init_numeraire = tf.constant([1.], FLOAT_DTYPE)
alpha = 0.95

# ==============================================================================
# === instantiate simulator
numeraire_simulator = simulators.ConstantBankAccount(rate)
instrument_simulator = simulators.JumpGBM(
    rate, drift, diffusion, intensity, jumpsize, jumpvol)

# ==============================================================================
# === hedgebook
hedgebook = books.DerivativeBook(
    maturity, instrument_simulator, numeraire_simulator)

spread = 10
itm = derivatives.PutCall(
    hedgebook.maturity,
    init_instruments - spread / 2,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    1)
atm = derivatives.PutCall(
    hedgebook.maturity,
    init_instruments,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    1)
otm = derivatives.PutCall(
    hedgebook.maturity,
    init_instruments + spread / 2,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    1)

hedgebook.add_derivative(itm, 0, 1)
hedgebook.add_derivative(atm, 0, -2)
hedgebook.add_derivative(otm, 0, 1)

# ==============================================================================
# === tradable book
tradebook = books.TradeBook(hedgebook)
num_tradables = 4
minm, maxm = 0.05, 0.25

for logmoneyness in tf.linspace(maxm, minm, num_tradables):
    putstrike = init_instruments * tf.exp(-logmoneyness)
    put = derivatives.JumpPutCall(
        maturity, putstrike, rate, instrument_simulator.volatility,
        intensity, jumpsize, jumpvol, -1)
    tradebook.add_derivative(put)

for logmoneyness in tf.linspace(minm, maxm, num_tradables):
    callstrike = init_instruments * tf.exp(logmoneyness)
    call = derivatives.JumpPutCall(
        maturity, callstrike, rate, instrument_simulator.volatility,
        intensity, jumpsize, jumpvol, -1)
    tradebook.add_derivative(call)

# ==============================================================================
# === compute hedge ratios
def lazy_feature_function(raw_data):
    main = raw_data["delta"][:, 0, tf.newaxis, :-1] * raw_data["numeraire"][:-1]
    pad = [[0, 0], [0, tf.shape(raw_data["delta"])[1] - 1], [0, 0]]

    return tf.unstack(tf.pad(main, pad), axis=-1)

def jump_feature_function(raw_data):
    ratios = derivatives.jumpriskratios(
        tradebook,
        raw_data["time"],
        raw_data["instruments"],
        raw_data["numeraire"])

    return tf.unstack(ratios * raw_data["numeraire"][:-1], axis=-1)


driver = utils.HedgeDriver(
    timesteps=timesteps,
    frequency=0, # no need for frequency for non-path dependent derivatives.
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=tradebook,
    cost=1/100 if cost else None,
    risk_neutral=not cost,
    learning_rate=1e-1
    )

driver.add_testcase(
    "lazy continuous-time",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function=lazy_feature_function,
    price_type="arbitrage")

driver.add_testcase(
    "jump continuous-time",
    hedge_models.FeatureHedge(),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_function=jump_feature_function,
    price_type="arbitrage")

driver.train(train_size, 1000, int(2**10))
driver.test(test_size)

utils.plot_markovian_payoff(driver, int(2**14), driver.testcases[0]["price"])
