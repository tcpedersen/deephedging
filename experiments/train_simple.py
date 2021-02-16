# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

from models import SimpleHedge, EntropicRisk
from derivative_books import random_simple_put_call_book

num_paths, num_steps = int(10**5), 30
init_state, book = random_simple_put_call_book(num_steps / 250)
time, samples = book.sample_paths(init_state, num_paths, num_steps, False)

information = tf.math.log(samples[:, :-1, :] / samples[:, :-1, 0][..., tf.newaxis])
trade = samples[:, :-1, :]
payoff = book.payoff(samples)

target = tf.zeros_like(payoff)
inputs = [information, trade, payoff]

model = SimpleHedge(num_steps, book.market_size, 2, 15)
optimizer = tf.keras.optimizers.Adam(1e-2)
risk_measure = EntropicRisk(1)
model.compile(optimizer, risk_measure)

batch_size, epochs = 256, 50
history = model.fit(inputs, target, batch_size, epochs, verbose=2)


plt.figure()

min_, max_ = tf.reduce_min(trade[:, -1]), tf.reduce_max(trade[:, -1])
x = tf.linspace(min_, max_, 1000)
y = -book.payoff(x[..., tf.newaxis, tf.newaxis])
plt.plot(x, y, color="black")

x = trade[..., -1]
y = model(inputs, True)
plt.scatter(x, y, s=0.5)

plt.show()



# with tf.GradientTape() as tape:
#     y = model(inputs)
#     loss = model.compiled_loss(target, y)
# dldw = tape.gradient(y, model.trainable_variables)

# dt = book.maturity / num_steps
# value = -payoff

# for step, strategy in enumerate(range(num_steps)):
#     info = trade[..., step]
#     holdings = -book.book_delta(info[..., tf.newaxis], step * dt)[..., -1]
#     increment = tf.math.subtract(trade[..., step + 1], info)
#     value += tf.reduce_sum(tf.math.multiply(holdings, increment), -1)

# plt.figure()
# x = samples[:, 0, -1]
# plt.scatter(x, value + payoff, s=0.5)
# plt.show()


model_simple = Deep_Hedging_Model(
    N = num_steps,
    d = 1,
    m = 15,
    risk_free = 0.,
    dt = book.maturity / num_steps,
    initial_wealth = 0.0,
    epsilon = 0.0,
    final_period_cost = False,
    strategy_type = "simple",
    use_batch_norm = True,
    kernel_initializer = "he_uniform",
    activation_dense = "relu",
    activation_output = "linear",
    delta_constraint = (0.0, 1.0),
    share_stretegy_across_time = False,
    cost_structure = "proportional")

loss = Entropy(model_simple.output, None, 1)
model_simple.add_loss(loss)

model_simple.compile(optimizer=optimizer)


fake_trade = tf.transpose(trade[:, 0, :])
fake_information = tf.transpose(information[:, 0, :])

x_all = []
for i in range(num_steps + 1):
  x_all += [fake_trade[i, :, None]]
  if i != num_steps:
    x_all += [fake_information[i, :, None]]
x_all += [payoff[:, None]]

model_simple.fit(x_all, batch_size=256, epochs=50, verbose=2)

plt.figure()
y = model_simple(x_all).numpy().ravel() - payoff
plt.scatter(fake_trade[-1, :], y, s=0.5)
plt.show()


for idx, var in enumerate(model_simple.trainable_variables):
    model.trainable_variables[idx].assign(var)
