# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softplus, linear

import books
import gradient_models
from constants import FLOAT_DTYPE

class DifferentialNeuralNetwork(tf.keras.Model):
    def __init__(self, num_layers, num_units, output_dim):
        super().__init__()

        self.num_layers = num_layers
        self.num_units = num_units
        self.output_dim = output_dim

        self.hidden_layers = \
            [Dense(self.num_units,
                   use_bias=True,
                   activation=softplus)
             for _ in range(self.num_layers - 1)]

        self.output_layer = Dense(self.output_dim, use_bias=True,
                                  activation=linear)

    def call(self, inputs, training=False):
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(inputs)
            x = inputs
            for layer in self.hidden_layers:
                x = layer(x)

            y = self.output_layer(x)
        dydx = t.batch_jacobian(y, inputs)

        return [y, dydx]

# =============================================================================
# === differential normaliser
class DifferentialLinearNormaliser:
    '''
    m is number of samples
    d is number of features
    q is number of labels
    '''

    def __init__(self):
        pass

    def fit(self, A, b, C, d):
        '''
        A: d x d
        b: d x 1
        C: q x q
        d: q x 1

        f: R^d -> R^q
        h(x) = C^-1 @ (f(A @ x + b) - d)
        '''
        self.A = tf.convert_to_tensor(A, FLOAT_DTYPE)
        self.b = tf.convert_to_tensor(b, FLOAT_DTYPE)
        self.C = tf.convert_to_tensor(C, FLOAT_DTYPE)
        self.d = tf.convert_to_tensor(d, FLOAT_DTYPE)

    def transform_x(self, x):
        return tf.matmul(x, self.A, transpose_b=True) + tf.transpose(self.b)

    def transform_y(self, y):
        return tf.matmul(y, self.C, transpose_b=True) + tf.transpose(self.d)

    def transform_dydx(self, dydx):
        '''
        dydx: m x q x d
        '''
        return self.C @ dydx @ tf.linalg.inv(self.A)

    def transform(self, x, y, dydx):
        return self.transform_x(x), self.transform_y(y), \
            self.transform_dydx(dydx)

    def fit_transform(self, x, y, dydx):
        self.fit(x, y)
        return self.transform(x, y, dydx)

    def inverse_transform_x(self, x):
        return tf.matmul(x - tf.transpose(self.b), tf.linalg.inv(self.A),
                         transpose_b=True)

    def inverse_transform_y(self, y):
        return tf.matmul(y - tf.transpose(self.d), tf.linalg.inv(self.C),
                         transpose_b=True)

    def inverse_transform_dydx(self, dydx):
        return tf.linalg.inv(self.C) @ dydx @ self.A

    def inverse_transform(self, x, y, dydx):
        return self.inverse_transform_x(x), self.inverse_transform_y(y), \
            self.inverse_transform_dydx(dydx)


class DifferentialMeanVarianceNormaliser(DifferentialLinearNormaliser):
    def __init__(self):
        pass

    def fit(self, x, y):
        self.x_mean, self.x_std = \
            tf.reduce_mean(x, axis=0), tf.math.reduce_std(x, axis=0)

        self.y_mean, self.y_std = \
            tf.reduce_mean(y, axis=0), tf.math.reduce_std(y, axis=0)

        super().fit(
            tf.linalg.diag(1 / self.x_std),
            tf.reshape(-self.x_mean / self.x_std, (-1, 1)),
            tf.linalg.diag(1 / self.y_std),
            tf.reshape(-self.y_mean / self.y_std, (-1, 1))
            )

train_size, test_size, timesteps = int(2**13), int(2**12), 1
init_instruments, init_numeraire, book = books.simple_put_call_book(
    timesteps / 12, 100, 100, 0.02, 0.05, 0.2, 1)

time, instruments, numeraire = book.sample_paths(
    init_instruments, init_numeraire, train_size, timesteps,
    True, True, 0, 100., 10.)

x = instruments[:, 0, 0, tf.newaxis]
y = book.payoff(time, instruments, numeraire)[..., tf.newaxis]
dydx = book.adjoint(time, instruments, numeraire)[..., 0, tf.newaxis]

normaliser = DifferentialMeanVarianceNormaliser()

norm_x, norm_y, norm_dydx = normaliser.fit_transform(x, y, dydx)

model = DifferentialNeuralNetwork(6, 5, 1)
model.compile("Adam", "mean_squared_error",
              loss_weights=[0.5, 0.5])
model.fit(norm_x, [norm_y, norm_dydx], epochs=100, batch_size=32)

norm_y_pred, norm_dydx_pred = model(norm_x)
_, y_pred, dydx_pred = normaliser.inverse_transform(
    norm_x, norm_y_pred, norm_dydx_pred)


# === Price
plt.figure()

plt.scatter(x.numpy(), tf.squeeze(y).numpy(), color="lightgrey", s=0.5)
plt.scatter(x.numpy(), book.value(time, instruments, numeraire)[:, 0].numpy(),
            color="red", s=0.5)
plt.scatter(x.numpy(), tf.squeeze(y_pred).numpy(), color="black", s=0.5)

plt.show()

# === Delta
plt.figure()

plt.scatter(x.numpy(), tf.squeeze(dydx).numpy(), color="lightgrey", s=0.5)
plt.scatter(x.numpy(), book.delta(time, instruments, numeraire)[:, 0, 0].numpy(),
            color="red", s=0.5)
plt.scatter(x.numpy(), tf.squeeze(dydx_pred).numpy(), color="black", s=0.5)

plt.show()
