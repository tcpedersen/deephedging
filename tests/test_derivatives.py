# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

from tensorflow.debugging import assert_near

import derivatives
from books import random_simple_put_call_book
from constants import FLOAT_DTYPE

class test_binary(unittest.TestCase):
    def test_value_delta(self):
        # https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html

        maturity, strike, rate, volatility = 1, 100, 0.05, 0.2
        binary = derivatives.BinaryCall(maturity, strike, volatility)

        time = tf.constant([0, 1/2, 1], FLOAT_DTYPE)
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
        payoff_result = binary.payoff(instrument, numeraire)

        assert_near(price_result, price_expected)
        assert_near(delta_result, delta_expected)
        assert_near(payoff_result, payoff_expected)


class test_barrier(unittest.TestCase):
    def setUp(self):
        self.maturity, self.strike, self.rate, self.volatility \
            = 1, 100, 0.05, 0.2
        self.time = tf.constant([0, 1/2, 1], FLOAT_DTYPE)
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
        payoff_result = binary.payoff(instrument, self.numeraire)

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
        instrument = tf.constant([[90, 93,  86],
                                  [93., 100, 122]], FLOAT_DTYPE)

        barrier = 95
        binary = derivatives.BarrierCall(
            self.maturity,
            self.strike,
            barrier,
            self.rate,
            self.volatility,
            outin=1,
            updown=1)

        price_expected = tf.constant([
            [0., 0., 0.],
            [0., 0., 0.]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)


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
                                  [90., 121, 102]], FLOAT_DTYPE)

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
            [5.06817662477, 23.8937611227, 2]
            ], FLOAT_DTYPE) / self.numeraire

        self.run_value_delta(instrument, binary, price_expected)
