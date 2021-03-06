# -*- coding: utf-8 -*-
import models
import utils
from books import random_barrier_book

# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, timesteps = int(2**16), int(2**16), 7
num_hedges_per_day = 5
alpha = 0.95
cost = None
risk_neutral = True
num_layers, num_units = 2, 15


# ==============================================================================
# === sample train data
init_instruments, init_numeraire, book = random_barrier_book(
    timesteps / 250, 1, 1, 1, 73)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_train_paths,
    timesteps * num_hedges_per_day, risk_neutral)

train = utils.hedge_model_input(instruments, numeraire, book)
benchmark = utils.benchmark_input(instruments, numeraire, book)
no_liability = utils.no_liability_input(instruments, numeraire, book)

# ==============================================================================
# === train simple model
hedge_model = models.MemoryHedge(
    timesteps, book.instrument_dim, book.instrument_dim, num_layers, num_units)

if cost is not None:
    hedge_model.add_cost_layers(cost)

history, norm_train, normaliser = utils.train_model(hedge_model, train, alpha, False)

# ==============================================================================
# === train benchmark
benchmark_model = models.DeltaHedge(book, time, numeraire)
if cost is not None:
    benchmark_model.add_cost_layers(cost)

_, _, _ = utils.train_model(benchmark_model, benchmark, alpha, False)


value, costs = hedge_model(norm_train)
hedge_price = hedge_model.risk_measure(value - costs - norm_train[-1])
utils.plot_barrier_payoff(hedge_model, norm_train, hedge_price, instruments,
                          numeraire, book)
hedge_risk = hedge_model.risk_measure(value - costs - norm_train[-1])


value, costs = benchmark_model(benchmark)
benchmark_price = benchmark_model.risk_measure(value - costs - norm_train[-1])
utils.plot_barrier_payoff(benchmark_model, benchmark, benchmark_price, instruments,
                          numeraire, book)
benchmark_risk = benchmark_model.risk_measure(value - costs - benchmark[-1])


print(f"hedge: {hedge_risk: .4f}")
print(f"hedge: {benchmark_risk: .4f}")




















# ==============================================================================
# === train no liability
no_liability_model = models.MemoryHedge(
    timesteps, book.instrument_dim, book.instrument_dim, num_layers, num_units)

if cost is not None:
    no_liability_model.add_cost_layers(cost)

_, _, _ = utils.train_model(no_liability_model, no_liability, alpha)

# ==============================================================================
# === calculate risk
_, simple_risk = utils.test_model(hedge_model, train, normaliser)
_, benchmark_risk = utils.test_model(benchmark_model, benchmark, None)
_, no_liability_risk = utils.test_model(
    no_liability_model, no_liability, normaliser)

print("=== train data")
print(f"simple model risk: {simple_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")

del train, benchmark, no_liability

# ==============================================================================
# === sample test data
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_test_paths,
    timesteps * num_hedges_per_day, risk_neutral)

test = utils.hedge_model_input(instruments, numeraire, book)
benchmark = utils.benchmark_input(instruments, numeraire, book)
no_liability = utils.no_liability_input(instruments, numeraire, book)


# ==============================================================================
# === calculate risk
norm_test, hedge_risk = utils.test_model(hedge_model, test, normaliser)
norm_benchmark, benchmark_risk = utils.test_model(
    benchmark_model, benchmark, None)
_, no_liability_risk = utils.test_model(
    no_liability_model, no_liability, normaliser)

print("=== test data")
print(f"simple model risk: {hedge_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")


# ==============================================================================
# === calculate prices
hedge_price = hedge_risk - no_liability_risk
benchmark_price = book.value(time, instruments, numeraire)[0, 0]
no_liability_price = 0. # TODO is this true?

print(f"hedge_model price: {hedge_price:5f}")
print(f"benchmark price: {benchmark_price:5f}")


# ==============================================================================
# === calculate total risk
print(f"simple model total risk: {hedge_risk - hedge_price:5f}")
print(f"benchmark total risk: {benchmark_risk - benchmark_price:5f}")
print(f"no liability total risk: {no_liability_risk - no_liability_price:5f}")


# ==============================================================================
# === visualise distribution
utils.plot_distributions([hedge_model, benchmark_model],
                         [norm_test, norm_benchmark],
                         [hedge_price, benchmark_price])


# =============================================================================
# === visualize
utils.plot_barrier_payoff(hedge_model, norm_test, hedge_price, instruments,
                          numeraire, book)

utils.plot_barrier_payoff(benchmark_model, norm_benchmark, benchmark_price, instruments,
                          numeraire, book)