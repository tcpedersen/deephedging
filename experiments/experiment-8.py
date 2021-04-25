# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import simulators
import derivatives
import preprocessing
import gradient_models
import utils
import hedge_models
import approximators

from constants import FLOAT_DTYPE


def get_input_data(time, instrument, numeraire, hedge, tradables):
    tradable_value = [d.value(time, instrument, numeraire) for d in tradables]
    tradable_delta = [d.delta(time, instrument, numeraire) for d in tradables]

    hedge_delta = [hedge.delta(time, instrument, numeraire)]
    payoff = hedge.payoff(time, instrument, numeraire)

    features = tf.unstack(tf.stack(hedge_delta + tradable_delta, 1), axis=-1)

    instruments = tf.stack([instrument] + tradable_value, axis=1)
    martingales = instruments / numeraire

    return features, martingales, payoff

# ==============================================================================
# === parameters
timesteps = 14
maturity = timesteps / 250
rate, drift, diffusion = 0.02, [0.05], [[0.18]]
intensity, jumpsize, jumpvol = 1.0, -0.2, 0.15

time = tf.cast(tf.linspace(0., maturity, timesteps + 1), FLOAT_DTYPE)
init_instruments = tf.constant([100.], FLOAT_DTYPE)
init_numeraire = tf.constant([1.], FLOAT_DTYPE)

# ==============================================================================
# === instantiate simulator
numeraire_simulator = simulators.ConstantBankAccount(rate)
instrument_simulator = simulators.JumpGBM(
    rate, drift, diffusion, intensity, jumpsize, jumpvol)

# ==============================================================================
# === tradable derivatives
num_tradables = 10
tradable_strikes = tf.linspace(90.0, 110.0, num_tradables)
tradable_derivatives = []
for k in tradable_strikes:
    option = derivatives.JumpPutCall(
        maturity, k, rate, instrument_simulator.volatility,
        intensity, jumpsize, jumpvol, 1)
    tradable_derivatives.append(option)

hedge_derivative = derivatives.JumpPutCall(
        maturity, 100.0, rate, instrument_simulator.volatility,
        intensity, jumpsize, jumpvol, 1)

# ==============================================================================
# === construct training data
hedge_train_data = int(2**18)
train_main_instrument = instrument_simulator.simulate(
    time=time,
    init_state=init_instruments,
    batch_size=hedge_train_data,
    risk_neutral=True,
    use_sobol=False,
    skip=0
    )
numeraire = numeraire_simulator.simulate(time, init_numeraire, 1, True)[0, 0, :]

features, martingales, payoff = get_input_data(
    time,
    train_main_instrument[:, 0, :],
    numeraire,
    hedge_derivative,
    tradable_derivatives)

# ==============================================================================
# === train hedge model
model = hedge_models.LinearFeatureHedge(
    timesteps,
    len(tradable_derivatives) + 1,
    [approximators.IdentityFeatureMap]
    )

model.compile(
    risk_measure=hedge_models.ExpectedShortfall(0.95),
    optimizer=tf.keras.optimizers.Adam(1e-1))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    patience=2
)

callbacks = [early_stopping, reduce_lr]

history = model.fit(
    [features, martingales, payoff],
    batch_size=1024,
    epochs=1000,
    callbacks=callbacks,
    verbose=2)

train_value, train_costs = model([features, martingales])
train_risk = model.risk_measure(train_value - train_costs - payoff)
print(f"train risk: {train_risk}")

# ==============================================================================
# === train benchmark model
benchmark = hedge_models.LinearFeatureHedge(
    timesteps,
    1,
    [approximators.IdentityFeatureMap]
    )

benchmark.compile(
    risk_measure=hedge_models.ExpectedShortfall(0.95),
    optimizer=tf.keras.optimizers.Adam(1e-1))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    patience=2
)

callbacks = [early_stopping, reduce_lr]

history = benchmark.fit(
    [[f[:, 0, tf.newaxis] for f in features],
     martingales[:, 0, tf.newaxis, :],
     payoff],
    batch_size=1024,
    epochs=1000,
    callbacks=callbacks,
    verbose=2)

bm_train_value, bm_train_costs = benchmark([
    [f[:, 0, tf.newaxis] for f in features], martingales[:, 0, tf.newaxis, :]])
bm_train_risk = model.risk_measure(bm_train_value - bm_train_costs - payoff)
print(f"benchmark train risk: {bm_train_risk}")

# ==============================================================================
# === test hedge model
hedge_test_data = int(2**18)
test_main_instrument = instrument_simulator.simulate(
    time=time,
    init_state=init_instruments,
    batch_size=hedge_test_data,
    risk_neutral=True,
    use_sobol=False,
    skip=0
    )

features, martingales, payoff = get_input_data(
    time,
    test_main_instrument[:, 0, :],
    numeraire,
    hedge_derivative,
    tradable_derivatives)

test_value, test_costs = model([features, martingales])
test_risk = model.risk_measure.evaluate(test_value - test_costs - payoff)

print(f"test risk: {test_risk}")


# ==============================================================================
# === test benchmark model
benchmark_test_value, benchmark_test_costs = benchmark(
    [[f[:, 0, tf.newaxis] for f in features], martingales[:, 0, tf.newaxis, :]])
benchmark_test_risk = model.risk_measure.evaluate(
    benchmark_test_value - benchmark_test_costs - payoff)

print(f"benchmark test risk: {benchmark_test_risk}")
