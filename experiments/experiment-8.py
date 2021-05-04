# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

from functools import partial
from time import perf_counter

import books
import simulators
import derivatives
import utils
import hedge_models
import preprocessing
import approximators

from constants import FLOAT_DTYPE

tf.get_logger().setLevel('ERROR')

# ==============================================================================
# === parameters
cost = True if str(sys.argv[1]) == "cost" else False
train_size, test_size, timesteps = int(2**18), int(2**18), 14
maturity = timesteps / 250
rate, drift, diffusion = 0.02, [0.05], [[0.15]]
intensity, jumpsize, jumpvol = 0.25, -0.2, 0.15

time = tf.cast(tf.linspace(0., maturity, timesteps + 1), FLOAT_DTYPE)
init_instruments = tf.constant([100.], FLOAT_DTYPE)
init_numeraire = tf.constant([1.], FLOAT_DTYPE)

alpha = 0.95
risk_measure = partial(hedge_models.ExpectedShortfall, alpha=alpha)

units = 15
layers = 4
activation = tf.keras.activations.softplus

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
itm = derivatives.JumpPutCall(
    hedgebook.maturity,
    init_instruments - spread / 2,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    intensity, jumpsize, jumpvol, 1)
atm = derivatives.JumpPutCall(
    hedgebook.maturity,
    init_instruments,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    intensity, jumpsize, jumpvol, 1)
otm = derivatives.JumpPutCall(
    hedgebook.maturity,
    init_instruments + spread / 2,
    hedgebook.instrument_simulator.rate,
    hedgebook.instrument_simulator.volatility,
    intensity, jumpsize, jumpvol, 1)

hedgebook.add_derivative(itm, 0, 1)
hedgebook.add_derivative(atm, 0, -2)
hedgebook.add_derivative(otm, 0, 1)

# ==============================================================================
# === tradable book
tradebook = books.TradeBook(hedgebook)
num_tradables = int(sys.argv[2])

# determine distribution of strikes
sobol = tf.math.sobol_sample(1, num_tradables)
grid = tf.sort(tf.squeeze(sobol)) if num_tradables > 1 else tf.squeeze(sobol)
strikes = init_instruments * tf.math.exp(
    utils.norm_qdf(grid) * jumpvol + jumpsize)

for k in strikes:
    theta = int(tf.sign(k - init_instruments))
    assert theta != 0
    option = derivatives.JumpPutCall(
        maturity, k, rate, instrument_simulator.volatility,
        intensity, jumpsize, jumpvol, theta)
    tradebook.add_derivative(option)

# ==============================================================================
# === feature functions
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

# ==============================================================================
# === test
folder_name = r"results\experiment-8\cost" if cost \
    else r"results\experiment-8\no-cost"
num_trials = 2**3
lst_of_drivers =  []

for num in range(num_trials):
    print(f"number of tradables {num_tradables} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

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
        risk_measure=risk_measure(),
        normaliser=None,
        feature_function=lazy_feature_function,
        price_type="arbitrage")

    driver.add_testcase(
        "jump continuous-time",
        hedge_models.FeatureHedge(),
        risk_measure=risk_measure(),
        normaliser=None,
        feature_function=jump_feature_function,
        price_type="arbitrage")

    driver.add_testcase(
        "deep network",
        hedge_models.NeuralHedge(
            timesteps=timesteps,
            instrument_dim=tradebook.instrument_dim,
            internal_dim=0,
            num_layers=layers,
            num_units=units,
            activation=activation),
        risk_measure=risk_measure(),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_function="log_martingale",
        price_type="arbitrage")

    driver.add_testcase(
        "identity feature map",
        hedge_models.LinearFeatureHedge(
            timesteps=timesteps,
            instrument_dim=tradebook.instrument_dim,
            mappings=[approximators.IdentityFeatureMap] \
                * (1 + (driver.cost is not None))),
        risk_measure=risk_measure(),
        normaliser=None,
        feature_function=jump_feature_function, # use delta?
        price_type="arbitrage")

    if driver.cost is not None or not driver.risk_neutral:
        driver.add_liability_free(
            hedge_models.LinearFeatureHedge(
                timesteps=timesteps,
                instrument_dim=tradebook.instrument_dim,
                mappings=[approximators.IdentityFeatureMap] \
                    * (1 + (driver.cost is not None))),
            risk_measure=risk_measure(),
            normaliser=preprocessing.MeanVarianceNormaliser(),
            feature_function="log_martingale")
    driver.train(train_size, 1000, int(2**10))
    driver.test(test_size)

    lst_of_drivers.append(driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")

file_name = os.path.join(folder_name, fr"tradables-{num_tradables}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

utils.driver_data_dumb(
    lst_of_drivers,
    ["train_risk", "test_risk",
     "test_mean_value", "test_mean_abs_value", "test_variance_value",
     "test_mean_costs", "test_mean_abs_costs", "test_variance_costs",
     "test_mean_wealth", "test_mean_abs_wealth", "test_variance_wealth",
     "test_wealth_with_price_abs_mean", "test_wealth_with_price_variance",
     "price", "train_time"],
    file_name
    )
