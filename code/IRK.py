import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange


def IRK_weight(t0, t1, deg):
    c, b = get_c(deg)

def get_c(deg):
    """
    Get the Legendre poly on [0,1].
    """
    xg, wg = leggauss(deg)
    c = 0.5 * xg + 0.5
    b = 0.5 * wg
    return c, b

def get_lang(c):
    a = np.identity(len(c))
    poly = lagrange(c, y)