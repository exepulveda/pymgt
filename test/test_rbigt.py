import numpy as np

from pymgt import *
from test_utils import *

def test_rbigt():
    x = np.loadtxt("../data/synthetic_minerals.csv", delimiter=',', skiprows=1, usecols=[9, 10, 11, 12, 13, 14, 15, 16]) 
    ndata, ndim = x.shape

    x_copy = np.copy(x)

    maxiter=1

    metric = FRIEDMAN_METRIC

    pi = mv_index_distribution(ndata, ndim, metric, ndir=1000)
    target = np.median(pi)

    t = RBIGTransform(objective=metric, target=target, maxiter=maxiter)

    y,  x_back = assert_reversibility(t,  x, decimal=4)

    # x should not have changed
    np.testing.assert_array_almost_equal(x, x_copy, decimal=4)

    # forward transform should return the same in consecutive runs
    y_forward1 = t.transform(x)
    y_forward2 = t.transform(x)
    np.testing.assert_array_almost_equal(y_forward2, y_forward1, decimal=4)

    # forward transform should return the same y
    for dim in range(2, ndim+1):
        print(dim)
        x = np.random.random(size=(ndata, dim))
        t = RBIGTransform(objective=metric, target=target, maxiter=maxiter)
        y = t.fit_transform(x)
        y_forward = t.transform(x)
        assert y.shape ==  y_forward.shape
        np.testing.assert_array_almost_equal(y, y_forward, decimal=4)
