# -*- coding: utf-8 -*-
from unittest import TestCase
from tensorflow.debugging import assert_near

from market_models import BlackScholes

# ==============================================================================
# === BlackScholes
class test_BlackScholes(TestCase):
    def test_call_price(self):
        rate, vol = 0.05, 0.2
        model = BlackScholes(rate, vol) # rate, vol

        # test one dimension (itm)
        price_expected = 21.176511764526367
        price_result = model.call_price(0.25, 110, 90)
        assert_near(price_result, price_expected)

        # test two dimensions (itm, atm, otm)
        price_expected = [15.765978813171387, 10.450580596923828,
                         8.45120620727539]
        price_result = model.call_price([0.25, 1.0, 1.25],
                                        [75, 100, 125],
                                        [60, 100, 140])
        assert_near(price_result, price_expected)

    def test_call_delta(self):
        rate, vol = 0.05, 0.2
        model = BlackScholes(rate, vol) # rate, vol

        # test two dimensions (itm, atm, otm)
        delta_expected = [0.9919454455375671, 0.6368306875228882,
                         0.45402082800865173]
        delta_result = model.call_delta([0.25, 1.0, 1.25],
                                        [75, 100, 125],
                                        [60, 100, 140])
        assert_near(delta_result, delta_expected)