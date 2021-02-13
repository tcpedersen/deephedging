# -*- coding: utf-8 -*-
import tensorflow as tf

from models import SimpleHedge, EntropicRisk
from derivative_books import random_black_scholes_put_call_book

num_paths, num_steps = 1000, 52
init_state, book = random_black_scholes_put_call_book(2, 2, 2, 69)
time, samples = book.sample_paths(init_state, num_paths, num_steps, False)
payoff = book.payoff(samples)
target = tf.zeros_like(payoff)
inputs = [samples[:, :-1, :], payoff]

model = SimpleHedge(num_steps, book.market_size, 2, 15)
optimizer = tf.keras.optimizers.Adam()
loss = EntropicRisk(1)
model.compile(optimizer, loss)

batch_size, epochs = 64, 100

print(loss(target, model(inputs)))
model.fit(inputs, target, batch_size, epochs)




@tf.function
def get_gradient(model, inputs):
    with tf.GradientTape() as tape:
        y = model(inputs)
        loss = loss_fn(y)
    return tape.gradient(loss, model.trainable_weights)

@tf.function
def train_step(model, optimizer, inputs):
    grads = get_gradient(model, inputs)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

