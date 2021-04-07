# -*- coding: utf-8 -*-
import books
import preprocessing
import gradient_driver
import gradient_models

train_size, test_size, timesteps = int(2**12), int(2**16), 14
init_instruments, init_numeraire, book = books.simple_barrier_book(
    timesteps / 250, 100, 100, 95, 0.02, 0.05, 0.4, -1, -1)
frequency = 0

driver = gradient_driver.GradientDriver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-4,
    learning_rate_max=1e-2)

driver.set_exploration(95., 20.)

driver.add_testcase(
    name="deep memory network",
    model=gradient_models.SemiRecurrentTwinNetwork(
        timesteps=timesteps,
        layers=4,
        units=10,
        internal_dim=book.instrument_dim,
        use_batchnorm=True
        ),
    normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
    )

driver.train(train_size, 100, int(2**5))
driver.test(test_size)
gradient_driver.barrier_visualiser(driver, train_size)
