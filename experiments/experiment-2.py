# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from models import SimpleHedge, DeltaHedge, EntropicRisk, RecurrentHedge, MeanSquareRisk
from books import random_simple_put_call_book
from constants import FLOAT_DTYPE
from utils import PeakSchedule, MeanVarianceNormaliser

def split_sample(sample):
    numeraire = sample[:, -1, tf.newaxis, :]
    information = instruments = sample[:, :-1, :] / numeraire
    payoff = book.payoff(sample) / numeraire[:, 0, -1]

    return [information, instruments, payoff]

experiment = 2

if experiment == 1:
    risk_measure = MeanSquareRisk()
    min_delta = 1e-6
elif experiment == 2:
    risk_measure = EntropicRisk(1.)
    min_delta = 1e-4

# ==============================================================================
# === sample train data
num_paths, num_steps = int(10**6), 30
init_state, book = random_simple_put_call_book(num_steps / 250)
time, train_samples = book.sample_paths(init_state, num_paths, num_steps, False)
train = split_sample(train_samples)

# === normalise data
normaliser = MeanVarianceNormaliser()
train[0] = normaliser.fit_transform(train[0])

# === compile model
model = SimpleHedge(num_steps, book.instrument_dim, 2, 15)
optimizer = tf.keras.optimizers.Adam()
model.compile(risk_measure, optimizer=optimizer)

# === define callbacks
batch_size, epochs = 2**10, 50

early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=min_delta)
reduce_lr = ReduceLROnPlateau(monitor="loss", verbose=1, patience=2)
lr_schedule = LearningRateScheduler(PeakSchedule(1e-4, 1e-1, epochs), verbose=1)

callbacks = [early_stopping, reduce_lr]

# === train model
history = model.fit(train, batch_size=batch_size, epochs=epochs,
                    callbacks=callbacks)

# ==============================================================================
# === find indifference price
no_liability_model = SimpleHedge(num_steps, book.instrument_dim, 2, 15)
no_liability_model.compile(risk_measure, optimizer=optimizer)
no_liability = [train[0], train[1], tf.zeros_like(train[2])]
history = no_liability_model.fit(no_liability, batch_size=batch_size,
                                 epochs=epochs, callbacks=callbacks)

liability_risk = model.risk_measure(model(train))
no_liability_risk = model.risk_measure(no_liability_model(no_liability))

indiff_price = liability_risk - no_liability_risk

# ==============================================================================
# === visualise train
hedge_payoff = indiff_price + model(train)

plt.figure()
plt.scatter(train[1][..., -1], hedge_payoff + train[2], s=0.5)

x = tf.cast(tf.linspace(*plt.xlim(), 1000), FLOAT_DTYPE)
y = book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

plt.show()

# ==============================================================================
# === sample test data
time, test_samples = book.sample_paths(init_state, 10**6, num_steps, False)
test = split_sample(test_samples)
test[0] = normaliser.transform(test[0])

# === visualise
hedge_payoff = indiff_price + model(test)

plt.figure()
plt.scatter(test[1][..., -1], hedge_payoff + test[2], s=0.5)

x = tf.cast(tf.linspace(*plt.xlim(), 1000), FLOAT_DTYPE)
y = book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

plt.show()

# ==============================================================================
# === benchmark on test set
benchmark_model = DeltaHedge(num_steps, book)
benchmark = [test_samples[:, :-1, :], test_samples[:, :-1, :], book.payoff(test_samples)]
benchmark_model(benchmark)

price = book.book_value(init_state, time)[0, 0]
hedge_payoff = benchmark_model(benchmark)

plt.figure()
plt.scatter(test[1][..., -1], price + hedge_payoff + test[2], s=0.5) # TODO why not add price?

x = tf.cast(tf.linspace(*plt.xlim(), 1000), FLOAT_DTYPE)
y = book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

plt.show()

print(f"model risk (train): {model.risk_measure(model(train)):5f}")
print(f"model risk (test): {model.risk_measure(model(test)):5f}")
print(f"benchmark risk (test): {model.risk_measure(benchmark_model(benchmark)):5f}")

print(f"indifference price: {indiff_price:5f}")
print(f"no-rigidity price: {book.book_value(init_state, time)[..., 0].numpy()[0]:5f}")

plt.figure()
for data in [model(test), benchmark_model(benchmark)]:
    plot_data = np.random.choice(data, 100000)
    plt.hist(plot_data, bins=100, density=True, alpha=0.5)
plt.show()