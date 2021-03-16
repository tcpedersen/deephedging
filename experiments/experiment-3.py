# -*- coding: utf-8 -*-
import hedge_models
import books
import utils
import preprocessing
import approximators

# ==============================================================================
# === hyperparameters
train_size, test_size, timesteps = int(2**18), int(2**20), 14
frequency = 1
alpha = 0.95

num_layers, num_units = 2, 15

# ==============================================================================
# === setup
init_instruments, init_numeraire, book = books.random_barrier_book(
    timesteps / 250, 25, 10, 10, 72)

driver = utils.Driver(
    timesteps=timesteps,
    frequency=frequency,
    init_instruments=init_instruments,
    init_numeraire=init_numeraire,
    book=book,
    cost=2.5 / 100,
    risk_neutral=False,
    learning_rate=1e-2
    )

driver.add_testcase(
    name="memory network",
    model=hedge_models.MemoryHedge(
        timesteps=timesteps * frequency,
        instrument_dim=book.instrument_dim,
        internal_dim=book.instrument_dim,
        num_layers=num_layers,
        num_units=num_units),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=preprocessing.MeanVarianceNormaliser(),
    feature_type="log_martingale",
    price_type="indifference"
    )

driver.add_testcase(
    name="identity feature map",
    model=hedge_models.LinearFeatureHedge(
        timesteps * frequency,
        book.instrument_dim,
        [approximators.IdentityFeatureMap, approximators.IdentityFeatureMap]),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="indifference")

driver.add_testcase(
    name="continuous-time",
    model=hedge_models.DeltaHedge(
        timesteps * frequency,
        book.instrument_dim),
    risk_measure=hedge_models.ExpectedShortfall(alpha),
    normaliser=None,
    feature_type="delta",
    price_type="arbitrage")

if driver.cost is not None or not driver.risk_neutral:
    driver.add_liability_free(
        hedge_models.SimpleHedge(
            timesteps * frequency,
            book.instrument_dim,
            num_layers=num_layers,
            num_units=num_units),
        risk_measure=hedge_models.ExpectedShortfall(alpha),
        normaliser=preprocessing.MeanVarianceNormaliser(),
        feature_type="log_martingale")

driver.train(train_size, 100, int(2**10))
driver.test(test_size)
driver.test_summary()
driver.plot_distributions()


# # ==============================================================================
# # === setup model
# model = models.DeltaHedge(book, time, numeraire)
# model.compile(models.ExpectedShortfall(alpha))

# train = utils.benchmark_input(time, instruments, numeraire, book)


# # ==============================================================================
# # ====

# # improves speed
# @tf.function
# def run(x):
#     return model(x)

# value, _ = model(train)

# price = book.value(time, instruments, numeraire)[0, 0]
# print(f"initial investment: {price * numeraire[0]:.4f}.")

# payoff = utils.precise_mean(book.payoff(time, instruments, numeraire))
# print(f"average discounted option payoff: {payoff:.4f}.")

# hedge_wealth = price + utils.precise_mean(value)
# print(f"average discounted portfolio value: {hedge_wealth:.4f}.")

# # =============================================================================
# # === visualize
# utils.plot_distributions([model], [train], [price])

# utils.plot_barrier_payoff(
#         model,
#         train,
#         price,
#         time,
#         instruments,
#         numeraire,
#         book
#     )