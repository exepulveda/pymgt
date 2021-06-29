import numpy as np

from pymgt import *
from pymgt.ppmt_utils import *

from test_utils import *


def test_r():
    x = np.random.uniform(10.0, 20.0, size=500)

    r = r_transform(x)
    assert np.all(r >= -1.0)
    assert np.all(r <= 1.0)


def test_pi():
    np.random.seed(1)
    for size in [1000, 10000, 100000]:
        x = np.random.randn(size)
        r = r_transform(x)

        p = legendre_poly(r)
        pi = friedman_index_internal(p)

        assert pi < 0.008

    for size in [100, 1000, 10000]:
        x = np.random.uniform(-4.0, 4.0, size=size)
        r = r_transform(x)

        p = legendre_poly(r)
        pi = friedman_index_internal(p)

        assert pi > 0.2

