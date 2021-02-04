# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

from market_models import BlackScholes, ConstantRateBankAccount
from constants import FLOAT_DTYPE

maturity, asset_spot, strike = 1.25, 100, 100
bank_spot = 1
drift, rate, vol = 0.1, 0.01, 0.3

asset_model = BlackScholes(drift, rate, vol)
bank_model = ConstantRateBankAccount(rate)

num_timesteps = 2**12
num_paths = 10000
dt = maturity / num_timesteps

asset_paths = asset_model.sample_path(maturity, asset_spot, num_paths, num_timesteps, "p")
bank_path = bank_model.sample_path(maturity, bank_spot, num_paths, num_timesteps)

# initial holdings
V = asset_model.call_price(maturity, asset_spot, strike)
a = tf.ones(num_paths, FLOAT_DTYPE) * asset_model.call_delta(maturity, asset_spot, strike)
b = (V - asset_spot * a) / bank_spot

for idx in range(1, num_timesteps + 1):
    asset = asset_paths[:, idx]
    bank = bank_path[idx]
    V = a * asset + b * bank

    a = asset_model.call_delta(maturity - dt * idx, asset, strike)
    b = (V - asset * a) / bank

# === stats
payoff = (asset - strike) * tf.cast(asset > strike, FLOAT_DTYPE)
print(f"mean hedge error: {tf.reduce_mean(V - payoff):.4}.")
print(f"mean absolute hedge error: {tf.reduce_mean(tf.abs(V - payoff)):.4}.")
print(f"mean squared hedge error: {tf.reduce_mean(tf.square(V - payoff)):.4}.")

# === plot
xmin, xmax = 80., 110.
x = tf.linspace(xmin, xmax, 1000)
y = (x - strike) * tf.cast(x > strike, FLOAT_DTYPE)

inplot = (xmin < asset) & (asset < xmax)

plt.figure()
plt.plot(x, y, color="black")
plt.scatter(tf.boolean_mask(asset, inplot), tf.boolean_mask(V, inplot),
            5, alpha=0.5, color="cyan")
plt.show()
