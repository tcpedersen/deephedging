# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import erf, erfinv

ONE_OVER_SQRT_TWO_PI = 1. / np.sqrt(2. * np.pi)
SQRT_TWO = np.sqrt(2.)

# ==============================================================================
# === Gaussian
def norm_pdf(x):
    return ONE_OVER_SQRT_TWO_PI * np.exp(-0.5 * x**2.)

def norm_cdf(x):
    return 0.5 * (1. + erf(x / SQRT_TWO))

def norm_qdf(x):
    return erfinv(2. * x - 1.) * SQRT_TWO