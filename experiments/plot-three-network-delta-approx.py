# -*- coding: utf-8 -*-
import tensorflow as tf

import utils
import derivatives
import books
import gradient_models
import gradient_driver

init_instruments, init_numeraire, book = books.simple_empty_book(
    1 / 52, 100, 0., 0.05, 0.2)
init_numeraire = tf.ones_like(init_numeraire)
layers = 4
units = 20
train_size, test_size = int(2**13), int(2**18)

spread = 10
itm = derivatives.PutCall(
    book.maturity,
    init_instruments - spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
atm = derivatives.PutCall(
    book.maturity,
    init_instruments,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)
otm = derivatives.PutCall(
    book.maturity,
    init_instruments + spread / 2,
    book.instrument_simulator.rate,
    book.instrument_simulator.volatility,
    1)

book.add_derivative(itm, 0, 1)
book.add_derivative(atm, 0, -2)
book.add_derivative(otm, 0, 1)

warmup_driver = gradient_driver.GradientDriver(
    timesteps=1,
    frequency=0,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-7,
    learning_rate_max=1e-2
)

warmup_driver.set_exploration(100., 15.)

warmup_driver.add_testcase(
    name="value network",
    model=gradient_models.SequenceValueNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.softplus
        ),
    train_size=train_size
    )

warmup_driver.add_testcase(
    name="twin network",
    model=gradient_models.SequenceTwinNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.softplus
        ),
    train_size=train_size
    )

warmup_driver.add_testcase(
    name="delta network",
    model=gradient_models.SequenceDeltaNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.sigmoid
        ),
    train_size=train_size
    )

warmup_driver.train(100, 64)
# warmup_driver.test(test_size)

file_name = r"figures\markovian-add\univariate-call-spread\delta"
gradient_driver.markovian_visualiser(warmup_driver, train_size, file_name)

