# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import pandas as pd
import sys

from time import perf_counter

import gradient_models
import gradient_driver
import books

warmup_train_size_twin = int(2**13)
warmup_train_size_value = int(2**13)
test_size, timesteps = int(2**17), 1

layers = 4
units = 20

dimension = int(sys.argv[1])

folder_name = r"figures\markovian-add\experiment-4"

# ==============================================================================
# === train gradient models

number_of_tests = 2**3
test_drivers = []

for num in range(number_of_tests):
    print(f"dimension {dimension} at test {num + 1} ".ljust(80, "="), end="")
    start = perf_counter()

    init_instruments, init_numeraire, book = books.random_mean_putcall_book(
        13 / 52, dimension, num)

    warmup_driver = gradient_driver.GradientDriver(
        timesteps=timesteps,
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

file_name = os.path.join(folder_name, fr"dimension-{dimension}.txt")
if os.path.exists(file_name):
    os.remove(file_name)

for idx in range(len(warmup_driver.testcases)):
    with open(file_name, "a") as file:
        name = warmup_driver.testcases[idx]["name"]
        file.write("".ljust(80, "=") + "\n")
        file.write("=== " + name + "\n")

    train_time = []
    mse_error = []
    mae_error = []

    for test in test_drivers:
        train_time.append(test.testcases[idx]["train_time"])
        mse_error.append(
            tf.squeeze(test.testcases[idx]["test_delta_mean_squared_error"]))
        mae_error.append(
            tf.squeeze(test.testcases[idx]["test_delta_mean_absolute_error"]))

    with open(file_name, "a") as file:
        file.write("train time\n")
    pd.DataFrame(train_time).T.to_csv(
        file_name,
        header=False,
        index=False,
        mode="a"
        )

    with open(file_name, "a") as file:
        file.write("mean squared error\n")
    pd.DataFrame(tf.stack(mse_error).numpy()).to_csv(
        file_name,
        header=False,
        index=False,
        mode="a"
        )

    with open(file_name, "a") as file:
        file.write("mean absolute error\n")
    pd.DataFrame(tf.stack(mae_error).numpy()).to_csv(
        file_name,
        header=False,
        index=False,
        mode="a"
        )

    with open(file_name, "a") as file:
        file.write("\n\n")
