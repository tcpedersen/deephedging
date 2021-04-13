# -*- coding: utf-8 -*-
import tensorflow as tf

import gradient_models
import gradient_driver
import books
import derivatives

train_size, test_size, timesteps = int(2**12), int(2**18), 14

init_instruments, init_numeraire, book = books.simple_empty_book(
    timesteps / 250, 100, 0.02, 0.05, 0.2)

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

multiplier = 1

# # ==============================================================================
# # === train gradient models
layers = 4
units = 5

warmup_driver = gradient_driver.GradientDriver(
    timesteps=timesteps * multiplier,
    frequency=0,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-5,
    learning_rate_max=1e-2
    )

warmup_driver.set_exploration(100., 25.)

warmup_driver.add_testcase(
    name="value network",
    model=gradient_models.SequenceValueNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.softplus
        )
    )

warmup_driver.add_testcase(
    name="twin network",
    model=gradient_models.SequenceTwinNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.softplus
        )
    )

warmup_driver.add_testcase(
    name="delta network",
    model=gradient_models.SequenceDeltaNetwork(
        layers=layers,
        units=units,
        activation=tf.keras.activations.sigmoid
        )
    )

warmup_driver.train(train_size, 100, 32)
warmup_driver.test(test_size)
warmup_driver.test_summary()
gradient_driver.markovian_visualiser(warmup_driver, test_size)
