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

def test_next_direction():
    ndata = 1000
    degree = 10
    maxiter = 100
    trace = True
    step = 0.01    

    np.random.seed(1)
    x = np.empty((ndata, 2))
    x[:, 0] = np.random.uniform(-4.0, 4.0, size=ndata)
    x[:, 1] = np.random.normal(size=ndata)

    best_direction, best_pi = find_next_best_direction_gd(x, degree=degree, maxiter=maxiter, trace=trace, step=step)
    assert len(best_direction) == 2
    np.testing.assert_almost_equal(np.linalg.norm(best_direction), 1.0)
    np.testing.assert_almost_equal(np.abs(best_direction[0]), 1.0, decimal=2)
    np.testing.assert_almost_equal(np.abs(best_direction[1]), 0.0, decimal=2)
    assert best_pi >= 4.0

    popsize = 50
    is_maximising_better = True
    index_func = friedman_index
    best_direction, best_pi = find_next_best_direction(x, index_func, is_maximising_better=is_maximising_better, maxiter=maxiter, trace=trace, popsize=popsize)
    assert len(best_direction) == 2
    np.testing.assert_almost_equal(np.linalg.norm(best_direction), 1.0)
    np.testing.assert_almost_equal(np.abs(best_direction[0]), 1.0, decimal=2)
    np.testing.assert_almost_equal(np.abs(best_direction[1]), 0.0, decimal=2)
    assert best_pi >= 4.0
