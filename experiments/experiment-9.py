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

from constants import FLOAT_DTYPE

tf.get_logger().setLevel('ERROR')

# ==============================================================================
# === parameters
cost = True # TODO must fix drift first
train_size, test_size, timesteps = int(2**18), int(2**18), 14
maturity = timesteps / 250
rate, drift, diffusion = 0.0, [0.0], [[0.15]]
intensity, jumpsize, jumpvol = 0.25, -0.2, 0.15

time = tf.cast(tf.linspace(0.0, maturity, timesteps + 1), FLOAT_DTYPE)
init_instruments = tf.constant([100.], FLOAT_DTYPE)
init_numeraire = tf.constant([1.0], FLOAT_DTYPE)

risk_measure = partial(hedge_models.ExpectedShortfall, alpha=0.95)
# risk_measure = partial(hedge_models.MeanVariance, aversion=100.0)

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
approxhedgebook = books.DerivativeBook(
    maturity, instrument_simulator, numeraire_simulator)
hedgebook = books.DerivativeBook(
    maturity, instrument_simulator, numeraire_simulator)

spread = 20 # TODO how wide?
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

approxhedgebook.add_derivative(
    derivatives.BachelierJumpCall.from_jumpputcall(
        itm, init_instruments), 0, 1)
approxhedgebook.add_derivative(
    derivatives.BachelierJumpCall.from_jumpputcall(
        atm, init_instruments), 0, -2)
approxhedgebook.add_derivative(
    derivatives.BachelierJumpCall.from_jumpputcall(
        otm, init_instruments), 0, 1)


# ==============================================================================
# === tradable book
approxtradebook = books.TradeBook(approxhedgebook)
tradebook = books.TradeBook(hedgebook)
num_tradables = 2 # int(sys.argv[2]) TODO

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
    approxtradebook.add_derivative(
        derivatives.BachelierJumpCall.from_jumpputcall(
            option, init_instruments))

# ==============================================================================
# === feature functions
def lazy_feature_function(raw_data, **kwargs):
    main = raw_data["delta"][:, 0, tf.newaxis, :-1] * raw_data["numeraire"][:-1]
    pad = [[0, 0], [0, tf.shape(raw_data["delta"])[1] - 1], [0, 0]]

    return tf.unstack(tf.pad(main, pad), axis=-1)


# ==============================================================================
# === test
folder_name = r"results\experiment-9\cost" if cost \
    else r"results\experiment-9\no-cost"
num_trials = 2**3
lst_of_drivers =  []

for num in range(num_trials):
    print(f"number of tradables {num_tradables} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    # === true driver
    truedriver = utils.HedgeDriver(
        timesteps=timesteps,
        frequency=0, # no need for frequency for non-path dependent derivatives.
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=tradebook,
        cost=1/100 if cost else None,
        risk_neutral=True, # TODO not cost,
        learning_rate=1e-1
        )

    truedriver.add_testcase(
        "lazy continuous-time",
        hedge_models.FeatureHedge(),
        risk_measure=risk_measure(),
        normaliser=None,
        feature_function=lazy_feature_function,
        price_type="arbitrage")

    truedriver.add_testcase(
        "deep network w. true value",
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

    truedriver.train(train_size, 1000, int(2**10))
    truedriver.test(test_size)
    lst_of_drivers.append(truedriver)

    # === approx driver
    approxdriver = utils.HedgeDriver(
        timesteps=timesteps,
        frequency=0, # no need for frequency for non-path dependent derivatives.
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=approxtradebook, # important
        cost=1/100 if cost else None,
        risk_neutral=True, # TODO not cost,
        learning_rate=1e-1
        )
    approxdriver.add_testbook(tradebook)

    approxdriver.add_testcase(
        "deep network w. approximate value",
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

    approxdriver.train(train_size, 1000, int(2**10))
    approxdriver.test(test_size)

    lst_of_drivers.append(approxdriver)

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



raw_data = approxdriver.sample(int(2**10))

truecase = truedriver.testcases[1]
true_input_data = truedriver.get_input(truecase, raw_data)
truestrategy = truecase["model"].strategy(true_input_data[0], training=False)

approxcase = approxdriver.testcases[0]
approx_input_data = approxdriver.get_input(approxcase, raw_data)
approxstrategy = approxcase["model"].strategy(approx_input_data[0], training=False)


for step in tf.range(timesteps):
    print(np.round(truestrategy[..., step], 2))
    print(np.round(approxstrategy[..., step], 2))
