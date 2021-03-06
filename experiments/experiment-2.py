# -*- coding: utf-8 -*-
import models
import utils
from books import random_put_call_book

# ==============================================================================
# === hyperparameters
num_train_paths, num_test_paths, timesteps = int(2**18), int(2**18), 7
num_hedges_per_day = 1
alpha = 0.95
cost = None
risk_neutral = True
num_layers, num_units = 2, 15


# ==============================================================================
# === sample train data
init_instruments, init_numeraire, book = random_put_call_book(
    timesteps / 250, 25, 10, 10, 69)
time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, num_train_paths,
    timesteps * num_hedges_per_day, risk_neutral)

train = utils.hedge_model_input(time, instruments, numeraire, book)
benchmark = utils.benchmark_input(time, instruments, numeraire, book)
no_liability = utils.no_liability_input(time, instruments, numeraire, book)


# ==============================================================================
# === train simple model
hedge_model = models.SimpleHedge(
    timesteps, book.instrument_dim, num_layers, num_units)

if cost is not None:
    hedge_model.add_cost_layers(cost)

history, norm_train, normaliser = utils.train_model(hedge_model, train, alpha)


# ==============================================================================
# === train benchmark
benchmark_model = models.DeltaHedge(book, time, numeraire)
if cost is not None:
    benchmark_model.add_cost_layers(cost)

_, _, _ = utils.train_model(benchmark_model, benchmark, alpha, False)


# ==============================================================================
# === train no liability
no_liability_model = models.SimpleHedge(
    timesteps, book.instrument_dim, num_layers, num_units)

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

test = utils.hedge_model_input(time, instruments, numeraire, book)
benchmark = utils.benchmark_input(time, instruments, numeraire, book)
no_liability = utils.no_liability_input(time, instruments, numeraire, book)


# ==============================================================================
# === calculate risk
norm_test, simple_risk = utils.test_model(hedge_model, test, normaliser)
norm_benchmark, benchmark_risk = utils.test_model(
    benchmark_model, benchmark, None)
_, no_liability_risk = utils.test_model(
    no_liability_model, no_liability, normaliser)

print("=== test data")
print(f"simple model risk: {simple_risk:5f}")
print(f"benchmark risk: {benchmark_risk:5f}")
print(f"no liability risk: {no_liability_risk:5f}")


# ==============================================================================
# === calculate prices
hedge_model_price = simple_risk - no_liability_risk
benchmark_price = book.value(time, instruments, numeraire)[0, 0]
no_liability_price = 0. # TODO is this true?

print(f"hedge_model price: {hedge_model_price:5f}")
print(f"benchmark price: {benchmark_price:5f}")


# ==============================================================================
# === calculate total risk
print(f"simple model total risk: {simple_risk - hedge_model_price:5f}")
print(f"benchmark total risk: {benchmark_risk - benchmark_price:5f}")
print(f"no liability total risk: {no_liability_risk - no_liability_price:5f}")


# ==============================================================================
# === visualise distribution
utils.plot_distributions([hedge_model, benchmark_model],
                         [norm_test, norm_benchmark],
                         [hedge_model_price, benchmark_price])
