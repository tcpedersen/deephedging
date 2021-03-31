# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

from tensorflow.debugging import assert_near

import derivatives
from constants import FLOAT_DTYPE

class test_putcall(unittest.TestCase):
    def test_call_value_delta(self):
        maturity, strike, rate, volatility, theta = 0.25, 90, 0.05, 0.2, 1
        putcall = derivatives.PutCall(maturity, strike, rate, volatility, theta)

        time = tf.constant([0, 0.1, maturity], FLOAT_DTYPE)
        instrument = tf.constant([[100, 110,  91],
                                  [100., 121, 85]], FLOAT_DTYPE)
        numeraire = tf.math.exp(rate * time)

        price_expected = tf.constant([
            [11.670086691861101, 20.680949336572269, 1.],
            [11.670086691861101, 31.672557545579522, 0.]
            ]) / numeraire

        delta_expected = tf.constant([
            [0.89039005940552085, 0.99679661077351212, 1.],
            [0.89039005940552085, 0.99996199608869929, 0.]
            ]) / numeraire

        payoff_expected = price_expected[..., -1]

        price_result = putcall.value(time, instrument, numeraire)
        delta_result = putcall.delta(time, instrument, numeraire)
        payoff_result = putcall.payoff(time, instrument, numeraire)

        assert_near(price_result, price_expected)
        assert_near(delta_result, delta_expected)
        assert_near(payoff_result, payoff_expected)


    @unittest.skip("tensorflow too imprecise.")
    def test_put_value_delta(self):
        maturity, strike, rate, volatility, theta = 1.3, 110, 0.02, 0.05, -1
        putcall = derivatives.PutCall(maturity, strike, rate, volatility, theta)

        time = tf.constant([0, 0.41, maturity], FLOAT_DTYPE)
        instrument = tf.constant([[100, 110,  120],
                                  [110., 121, 85]], FLOAT_DTYPE)
        numeraire = tf.math.exp(rate * time)

        price_expected = tf.constant([
            [7.4973075500600146, 1.2255281792756278, 0.],
            [1.3101071219942781, 0.014761684729265312, 25.]
            ]) / numeraire

        delta_expected = tf.constant([
            [-0.88244073147661295, -0.3442306041250397, 0.],
            [-0.31398908314745166, -0.0077279609756593093, -1]
            ]) / numeraire

        payoff_expected = price_expected[..., -1]

        price_result = putcall.value(time, instrument, numeraire)
        delta_result = putcall.delta(time, instrument, numeraire)
        payoff_result = putcall.payoff(time, instrument, numeraire)

        assert_near(price_result, price_expected)
        assert_near(delta_result, delta_expected)
        assert_near(payoff_result, payoff_expected)


class test_binary(unittest.TestCase):
    def test_value_delta(self):
        # https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html

        maturity, strike, rate, volatility = 1, 100, 0.05, 0.2
        binary = derivatives.BinaryCall(maturity, strike, volatility)

        time = tf.constant([0, 1/2, maturity], FLOAT_DTYPE)
        instrument = tf.constant([[100, 101,  91],
                                  [100., 121, 122]], FLOAT_DTYPE)
        numeraire = tf.math.exp(rate * time)

        price_expected = tf.constant([
            [0.532324834261914, 0.555946305588727, 0.],
            [0.532324834261914, 0.904132417939148, 1.]
            ]) / numeraire

        delta_expected = tf.constant([
            [0.018762017345847, 0.026819916282967, 0.],
            [0.018762017345847, 0.007901435056954, 0.]
            ]) / numeraire

        payoff_expected = price_expected[..., -1]

        price_result = binary.value(time, instrument, numeraire)
        delta_result = binary.delta(time, instrument, numeraire)
        payoff_result = binary.payoff(time, instrument, numeraire)

        assert_near(price_result, price_expected)
        assert_near(delta_result, delta_expected)
        assert_near(payoff_result, payoff_expected)


class test_barrier(unittest.TestCase):
    # https://www.coggit.com/tools/barrier_option_pricing.html

    def setUp(self):
        self.maturity, self.strike, self.rate, self.volatility \
            = 1, 100, 0.05, 0.2
        self.time = tf.constant([0, 1/2, self.maturity], FLOAT_DTYPE)
        self.numeraire = tf.math.exp(self.rate * self.time)


    def run_value_delta(self, instrument, binary, price_expected):
        with tf.GradientTape() as tape:
            tape.watch(instrument)
            price_result = binary.value(
                self.time, instrument, self.numeraire)

        delta_expected = tape.gradient(price_result, instrument)
        delta_result = binary.delta(
            self.time, instrument, self.numeraire)

        payoff_expected = price_expected[..., -1]
        payoff_result = binary.payoff(self.time, instrument, self.numeraire)

        assert_near(price_result, price_expected, atol=1e-4)
        assert_near(delta_result[..., :-1], delta_expected[..., :-1])
        assert_near(payoff_result, payoff_expected)

    def test_down_and_out_call_barrier_below_strike(self):
        instrument = tf.constant([[105, 93,  100],
                                  [100., 121, 122]], FLOAT_DTYPE)

        barrier = 95
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=1,
            updown=-1)

        price_expected = tf.constant([
            [10.8202386348, 0., 0.],
            [5.63625810907, 23.8003384294, 22]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_down_and_out_call_barrier_above_strike(self):
        instrument = tf.constant([[105, 93,  100],
                                  [110., 121, 122]], FLOAT_DTYPE)
        barrier = 103
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=1,
            updown=-1)

        price_expected = tf.constant([
            [3.05322108927, 0., 0.],
            [10.1365951857, 22.5643920906, 22]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_up_and_out_call_barrier_below_strike(self):
        barrier = 95

        with self.assertRaises(AssertionError):
            derivatives.BarrierCall(
                self.maturity,
                self.strike,
                barrier,
                self.rate,
                self.volatility,
                outin=1,
                updown=1)


    def test_up_and_out_call_barrier_above_strike(self):
        instrument = tf.constant([[105, 93,  103],
                                  [107., 121, 122]], FLOAT_DTYPE)
        barrier = 115
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=1,
            updown=1)

        price_expected = tf.constant([
            [0.345576012513, 0.950642284198, 3.],
            [0.28015077923, 0., 0.]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_down_and_in_call_barrier_below_strike(self):
        instrument = tf.constant([[105, 93,  105],
                                  [100., 121, 122]], FLOAT_DTYPE)

        barrier = 95
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=-1,
            updown=-1)

        price_expected = tf.constant([
            [3.03766763227, 3.40522515245, 5],
            [4.81432546312, 0.093422693371, 0]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_down_and_in_call_barrier_above_strike(self):
        instrument = tf.constant([[105, 93,  105],
                                  [110., 121, 122]], FLOAT_DTYPE)

        barrier = 102
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=-1,
            updown=-1)

        price_expected = tf.constant([
            [9.45567585598, 3.40522515245, 5],
            [6.5068057398, 1.0191439468, 0]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_up_and_in_barrier_below_strike(self):
        instrument = tf.constant([[95, 93,  96],
                                  [90., 121, 95]], FLOAT_DTYPE)

        barrier = 97
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=-1,
            updown=1)

        price_expected = tf.constant([
            [7.51087217835, 3.40522515245, 0],
            [5.09122207882, 23.8937611227, 0]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


    def test_up_and_in_barrier_above_strike(self):
        instrument = tf.constant([[95, 102,  101],
                                  [90, 121, 102],
                                  [110, 100, 110]], FLOAT_DTYPE)

        barrier = 105
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=-1,
            updown=1)

        price_expected = tf.constant([
            [7.49382803156, 8.12283426454, 0],
            [5.06817662477, 23.8937611227, 2],
            [17.662952423095703, 6.888732433319092, 10]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


class test_DiscreteGeometricPutCall(unittest.TestCase):
    def setUp(self):
        self.maturity, self.rate, self.volatility = 1.3, 0.02, 0.3
        self.strike = 100

        self.time = tf.constant([0, 0.1, 0.6, self.maturity], FLOAT_DTYPE)
        self.option = derivatives.DiscreteGeometricPutCall(
            maturity=self.maturity,
            strike=self.strike,
            rate=self.rate,
            volatility=self.volatility,
            theta=1
        )

        self.instrument = tf.constant([[100, 80, 120, 110],
                                       [110, 121, 120, 85]], FLOAT_DTYPE)
        self.numeraire = 1.1 * tf.math.exp(self.rate * self.time)


    def test_dga(self):
        result = self.option._dga(self.time, self.instrument)
        expected = tf.constant([
            [1.0000000000, 1.5499189875, 16.9785118357, 455.9051533831],
            [1.0000000000, 1.6153942662, 17.6957575773, 396.7001896122]],
            FLOAT_DTYPE)

        tf.debugging.assert_near(result, expected)


    def test_mean(self):
        result = self.option._mean(self.time)
        expected = tf.constant([-0.0305000000, -0.0272500000, -0.0122500000, \
                                0.0000000000], FLOAT_DTYPE)

        tf.debugging.assert_near(result, expected)


    def test_variance(self):
        result = self.option._variance(self.time)
        expected = tf.constant([0.1108800000, 0.0956700000, 0.0308700000,
                                0.0000000000], FLOAT_DTYPE)

        tf.debugging.assert_near(result, expected)


    def test_value_delta(self):
        drift = tf.constant([0.0191846154, 0.0171541667, 0.0045500000],
                            FLOAT_DTYPE)
        volatility = tf.constant([0.2920484681, 0.2823561581, 0.2100000000],
                                 FLOAT_DTYPE)

        gs = tf.constant([
            [398.1071705535, 297.8632906618, 484.5364955981, 455.9051533831],
            [450.6200285030, 510.0596651041, 505.0054119235, 396.7001896122]],
            FLOAT_DTYPE)

        scale = tf.constant([
            [5.1753932172, 4.8402784733, 4.8453649560, 2.9012146124],
            [5.3255094278, 5.4799798730, 5.0500541192, 3.2669427380]
            ], FLOAT_DTYPE)

        result_value = self.option.value(
            self.time, self.instrument, self.numeraire)
        result_delta = self.option.delta(
            self.time, self.instrument, self.numeraire)

        for k, (d, v) in enumerate(zip(drift, volatility)):
            benchmark = derivatives.PutCall(
                self.maturity, self.strike, d, v, self.option.theta)

            expected_value = benchmark.value(self.time, gs, self.numeraire)
            expected_delta = benchmark.delta(self.time, gs, self.numeraire) \
                * scale

            tf.debugging.assert_near(result_value[..., k],
                                     expected_value[..., k])

            tf.debugging.assert_near(result_delta[..., k],
                                     expected_delta[..., k])
