# -*- coding: utf-8 -*-
import books
import preprocessing
import gradient_driver
import gradient_models

train_size, test_size, timesteps = int(2**14), int(2**18), 12
# init_instruments, init_numeraire, book = books.simple_barrier_book(
#     timesteps / 250, 100, 100, 105, 0.02, 0.05, 0.4, "in", "up")
init_instruments, init_numeraire, book = books.simple_dga_putcall_book(
    timesteps / 12, 100, 100**(timesteps / 12), 0.02, 0.05, 0.4, 1)
frequency = 0

driver = gradient_driver.GradientDriver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    learning_rate_min=1e-5,
    learning_rate_max=1e-2)

driver.set_exploration(100., 10.)

# driver.add_testcase(
#     name="deep memory network",
#     model=gradient_models.SemiRecurrentTwinNetwork(
#         timesteps=timesteps,
#         layers=4,
#         units=10,
#         internal_dim=0,
#         use_batchnorm=False
#         ),
#     normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
#     )

# driver.add_testcase(
#     name="better memory network (please)",
#     model=gradient_models.MemoryNetwork(
#         memory_dim=book.instrument_dim,
#         network_layers=4,
#         network_units=10
#         ),
#     normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
#     )

driver.add_testcase(
    name="twin lstm network",
    model=gradient_models.TwinLSTMNetwork(
        lstm_cells=3,
        lstm_units=5,
        network_layers=6,
        network_units=5
        ),
    normaliser=preprocessing.DifferentialMeanVarianceNormaliser()
    )

driver.train(train_size, 100, int(2**10))
driver.test(test_size)
# gradient_driver.barrier_visualiser(driver, train_size)
# gradient_driver.dga_putcall_visualiser(driver, test_size)


# raw_data = driver.sample(train_size, skip=0, exploring=True)
# x, y, dydx = driver.get_input(driver.testcases[0], raw_data)
# driver.testcases[0]["model"].warmup(x, y, batch_size=int(2**6), epochs=100)
