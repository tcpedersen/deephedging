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
num_tradables = 1
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
# === generate network train data
network_train_size = int(2**13)
numeraire = numeraire_simulator.simulate(time, init_numeraire, 1, True)[0, 0, :]
loc, scale = 100., 15.
exploring = tf.exp(tf.random.truncated_normal(
    shape=(network_train_size, tf.shape(init_instruments)[-1]),
    mean=tf.math.log(loc**2 / tf.sqrt(loc**2 + scale**2)),
    stddev=tf.sqrt(tf.math.log(scale**2 / loc**2 + 1)),
    dtype=FLOAT_DTYPE
    ))
x = instrument_simulator.simulate(
    time=time,
    init_state=exploring,
    batch_size=network_train_size,
    risk_neutral=True,
    use_sobol=True,
    skip=0
    )

y = tf.stack([hedge_derivative.payoff(time, x[:, 0, :], numeraire)] +
    [d.payoff(time, x[:, 0, :], numeraire) \
     for d in tradable_derivatives], 1)
dydx = tf.stack([hedge_derivative.adjoint(time, x[:, 0, :], numeraire)] +
    [d.adjoint(time, x[:, 0, :], numeraire) \
     for d in tradable_derivatives], 1)[:, :, tf.newaxis, :-1]
x = x[..., :-1]

# ==============================================================================
# === train network
normaliser = preprocessing.DifferentialMeanVarianceNormaliser()
normaliser.fit(x, y)
normaliser.ymean = tf.zeros_like(normaliser.ymean)
normaliser.yvar = tf.ones_like(normaliser.yvar)

norm_x, norm_y, norm_dydx = normaliser.transform(x, y, dydx)

layers = 4
units = 20
activation = tf.keras.activations.softplus
output_dim = y.shape[1]

network = gradient_models.SequenceFullTwinNetwork(
    timesteps, layers, units, activation, output_dim)

network.compile(tf.keras.optimizers.Adam(1e-1), "mean_squared_error",
                loss_weights=0.5)

# TODO fix
for step, net in enumerate(network.networks):
    net(norm_x[..., step])
network(norm_x)

epochs = 100
lr_schedule = utils.PeakSchedule(1e-5, 1e-1, epochs)
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    tf.keras.callbacks.EarlyStopping("loss", patience=10)]

network.fit(norm_x, [norm_y[..., tf.newaxis], norm_dydx], 32, 100,
            callbacks=callbacks)

# ==============================================================================
# === construct second round of training data
hedge_train_data = int(2**18)
train_main_instrument = instrument_simulator.simulate(
    time=time,
    init_state=init_instruments,
    batch_size=hedge_train_data,
    risk_neutral=True,
    use_sobol=False,
    skip=0
    )

norm_main_instrument = normaliser.transform_x(train_main_instrument[..., :-1])
norm_y_pred, norm_dydx_pred = network(norm_main_instrument)

pred_value = normaliser.inverse_transform_y(norm_y_pred)
tradable_payoff = tf.stack(
    [d.payoff(time, train_main_instrument[:, 0, :], numeraire) \
     for d in tradable_derivatives],
    axis=1)

payoff = hedge_derivative.payoff(
    time, train_main_instrument[:, 0, :], numeraire)

instruments = tf.concat([
    train_main_instrument,
    tf.concat([pred_value[:, 1:, :], tradable_payoff[..., tf.newaxis]], -1)
    ], axis=1)

# TODO step = 0 should have true value
features = tf.unstack(
    normaliser.inverse_transform_dydx(norm_dydx_pred)[:, :, 0, :],
    axis=-1)
martingales = instruments / numeraire

# ==============================================================================
# === visualise prices
true_value = [
    hedge_derivative.value(time, train_main_instrument[:, 0, :], numeraire)] + \
    [d.value(time, train_main_instrument[:, 0, :], numeraire) \
     for d in tradable_derivatives]

for d in tf.range(tf.shape(pred_value)[1]):
    for step in tf.range(timesteps):
        spot = train_main_instrument[:, 0, step]
        key = tf.argsort(spot)

        xaxis = tf.gather(spot, key).numpy()
        prediction = tf.gather(pred_value[:, d, step], key).numpy()
        target = tf.gather(true_value[d][:, step], key).numpy()

        plt.figure()
        plt.plot(xaxis, prediction, color="black")
        plt.plot(xaxis, target, "--", color="red")
        plt.show()

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

norm_main_instrument = normaliser.transform_x(test_main_instrument[..., :-1])
norm_y_pred, norm_dydx_pred = network(norm_main_instrument)

true_value = [d.value(time, test_main_instrument[:, 0, :], numeraire) \
              for d in tradable_derivatives]

instruments = tf.concat([
    test_main_instrument,
    tf.stack(true_value, 1)
    ], axis=1)

features = tf.unstack(
    normaliser.inverse_transform_dydx(norm_dydx_pred)[:, :, 0, :],
    axis=-1)
martingales = instruments / numeraire
payoff = hedge_derivative.payoff(time, test_main_instrument[:, 0, :], numeraire)

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
