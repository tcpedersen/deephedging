# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

from models import SimpleHedge, EntropicRisk
from derivative_books import random_simple_put_call_book
from constants import FLOAT_DTYPE

def split_sample(sample):
    information = tf.math.log(sample[:, :-1, :] / sample[:, :-1, 0][..., tf.newaxis])
    trade = sample[:, :-1, :]
    payoff = book.payoff(sample)

    return [information, trade, payoff]

num_paths, num_steps = int(10**6), 30
init_state, book = random_simple_put_call_book(num_steps / 250)
time, train_samples = book.sample_paths(init_state, num_paths, num_steps, False)

train = split_sample(train_samples)

model = SimpleHedge(num_steps, book.market_size, 2, 15)
optimizer = tf.keras.optimizers.Adam(1e-3)
risk_measure = EntropicRisk(1)
model.compile(risk_measure, optimizer=optimizer)

batch_size, epochs = 256, 50
history = model.fit(train, batch_size=batch_size, epochs=epochs, verbose=2)

price = model.loss_fn(model(train, False))
hedge_payoff = price + model(train, False)

plt.figure()
plt.scatter(train[1][..., -1], price + model(train, False) + train[2], s=0.5)

x = tf.cast(tf.linspace(*plt.xlim(), 100), FLOAT_DTYPE)
y = book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

plt.show()


# ==============================================================================
# === test set
time, test_samples = book.sample_paths(init_state, num_paths, num_steps, False)
test = split_sample(test_samples)

price = model.loss_fn(model(test, False))
hedge_payoff = price + model(test, False)

plt.figure()
plt.scatter(test[1][..., -1], price + model(test, False) + test[2], s=0.5)

x = tf.cast(tf.linspace(*plt.xlim(), 100), FLOAT_DTYPE)
y = book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

plt.show()
