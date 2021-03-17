# -*- coding: utf-8 -*-
import models
import utils
import books

# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, timesteps = int(4 * 2**20), int(2**20), 30
num_hedges_per_day = 1
alpha = 0.95
cost = None
risk_neutral = True
num_layers, num_units = 2, 15


# ==============================================================================
# === sample train data
init_instruments, init_numeraire, book = books.random_barrier_book(
    timesteps / 250, 1, 1, 1, 69)
time, instruments, numeraire = book.sample_paths(
    init_instruments,
    init_numeraire,
    num_train_paths,
    timesteps * num_hedges_per_day,
    risk_neutral)

train = utils.hedge_model_input(time, instruments, numeraire, book)
benchmark = utils.benchmark_input(time, instruments, numeraire, book)
no_liability = utils.no_liability_input(time, instruments, numeraire, book)


# ==============================================================================
# === train benchmark
benchmark_model = models.DeltaHedge(book, time, numeraire)
if cost is not None:
    benchmark_model.add_cost_layers(cost)

_, _, _ = utils.train_model(benchmark_model, benchmark, alpha, False)


# ==============================================================================
# === train simple model
hedge_model = models.MemoryHedge(
        timesteps,
        book.instrument_dim,
        book.instrument_dim,
        num_layers,
        num_units
    )

if cost is not None:
    hedge_model.add_cost_layers(cost)

history, norm_train, normaliser = utils.train_model(
    hedge_model, train, alpha, True)


# ==============================================================================
# === train no liability
no_liability_model = models.MemoryHedge(
    timesteps, book.instrument_dim, book.instrument_dim, num_layers, num_units)

if cost is not None:
    no_liability_model.add_cost_layers(cost)

_, _, _ = utils.train_model(no_liability_model, no_liability, alpha)


# ==============================================================================
# === calculate risk
_, hedge_risk = utils.test_model(hedge_model, train, normaliser)
_, benchmark_risk = utils.test_model(benchmark_model, benchmark, None)
_, no_liability_risk = utils.test_model(
    no_liability_model, no_liability, normaliser)

print("=== train data")
print(f"hedge model risk: {hedge_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")

del train, benchmark, no_liability


# ==============================================================================
# === sample test data
time, instruments, numeraire = book.sample_paths(
        init_instruments,
        init_numeraire,
        num_test_paths,
        timesteps * num_hedges_per_day,
        risk_neutral
    )

test = utils.hedge_model_input(time, instruments, numeraire, book)
benchmark = utils.benchmark_input(time, instruments, numeraire, book)
no_liability = utils.no_liability_input(time, instruments, numeraire, book)


# ==============================================================================
# === calculate risk
norm_test, hedge_risk = utils.test_model(hedge_model, test, normaliser)
norm_benchmark, benchmark_risk = utils.test_model(
    benchmark_model, benchmark, None)
_, no_liability_risk = utils.test_model(
    no_liability_model, no_liability, normaliser)

print("=== test data")
print(f"hedge model risk: {hedge_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")


# ==============================================================================
# === calculate prices
hedge_price = hedge_risk - no_liability_risk
benchmark_price = book.value(time, instruments, numeraire)[0, 0]
no_liability_price = 0. # TODO is this true?

print(f"hedge model price: {hedge_price:5f}")
print(f"benchmark price: {benchmark_price:5f}")


# ==============================================================================
# === calculate total risk
print(f"hedge model total risk: {hedge_risk - hedge_price:5f}")
print(f"benchmark total risk: {benchmark_risk - benchmark_price:5f}")
print(f"no liability total risk: {no_liability_risk - no_liability_price:5f}")


# ==============================================================================
# === visualise distribution
utils.plot_distributions([hedge_model, benchmark_model],
                         [norm_test, norm_benchmark],
                         [hedge_price, benchmark_price])


# =============================================================================
# === visualize
utils.plot_barrier_payoff(
        hedge_model,
        norm_test,
        hedge_price,
        time,
        instruments,
        numeraire,
        book
    )

utils.plot_barrier_payoff(
        benchmark_model,
        norm_benchmark,
        benchmark_price,
        time,
        instruments,
        numeraire,
        book
    )