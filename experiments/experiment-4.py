# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys

from time import perf_counter

import utils
import gradient_models
import gradient_driver
import random_books

warmup_train_size_twin = int(2**13)
warmup_train_size_value = int(2**13)
test_size, timesteps = int(2**17), 1

layers = 4
units = 20

dimension = int(sys.argv[1])

folder_name = r"results\experiment-4"

# ==============================================================================
# === train gradient models
rate = 0.02
drift = 0.05

number_of_tests = 2**7
test_drivers = []

for num in range(number_of_tests):
    print(f"dimension {dimension} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    volatility = tf.random.uniform((dimension, ), 0.2, 0.3)
    init_instruments, init_numeraire, book = random_books.random_empty_book(
        13 / 52, dimension, rate, drift, volatility, num)
    # random_books.add_butterfly(init_instruments, book, spread=20)
    random_books.add_calls(init_instruments, book)

    warmup_driver = gradient_driver.GradientDriver(
        timesteps=timesteps,
        frequency=0,
        init_instruments=init_instruments,
        init_numeraire=init_numeraire,
        book=book,
        learning_rate_min=1e-7,
        learning_rate_max=1e-2
        )

    warmup_driver.set_exploration(100.0, 15.0)

    warmup_driver.add_testcase(
        name="value network",
        model=gradient_models.SequenceValueNetwork(
            layers=layers,
            units=units,
            activation=tf.keras.activations.softplus
            ),
        train_size=warmup_train_size_value
        )

    warmup_driver.add_testcase(
        name="twin network",
        model=gradient_models.SequenceTwinNetwork(
            layers=layers,
            units=units,
            activation=tf.keras.activations.softplus
            ),
        train_size=warmup_train_size_twin
        )

    warmup_driver.add_testcase(
        name="delta network",
        model=gradient_models.SequenceDeltaNetwork(
            layers=layers,
            units=units,
            activation=tf.keras.activations.sigmoid
            ),
        train_size=warmup_train_size_twin
        )

    warmup_driver.train(100, 64)
    warmup_driver.test(test_size)

    test_drivers.append(warmup_driver)

    end = perf_counter() - start
    print(f" {end:.3f}s")

file_name = os.path.join(folder_name, fr"dimension-{dimension}-call.txt")
if os.path.exists(file_name):
    os.remove(file_name)

utils.driver_data_dumb(
    test_drivers,
    ["train_time",
     "test_delta_mean_squared_error",
     "test_delta_mean_absolute_error"],
    file_name
    )
