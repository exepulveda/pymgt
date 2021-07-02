import numpy as np

from pymgt import *
from test_utils import *

def test_ppmt():
    ndim = 4
    ndata = 1000
    maxiters = 10

    np.random.seed(1)

    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))
    pidist = mv_index_distribution(ndata, ndim, friedman_index, ndir=1000)
    target = np.median(pidist)
    
    print(f"target for ({ndata},{ndim})={target}")

    t = PPMTransform(maxiter=maxiters, target=target)

    y, x_back = assert_reversibility(t, x)        

    for dim in range(ndim):
        assert_normality(y[:, dim])

    # forward transform should return the same y

    y_forward = t.transform(x)

    np.testing.assert_array_almost_equal(y, y_forward)


def test_state():
    ndim = 4
    ndata = 1000
    maxiters = 10

    np.random.seed(1)

    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))
    pidist = mv_index_distribution(ndata, ndim, friedman_index, ndir=1000)
    target = np.median(pidist)
    
    print(f"target for ({ndata},{ndim})={target}")

    t1 = PPMTransform(maxiter=maxiters, target=target)

    t1.fit(x)
    y1 = t1.transform(x)

    state = t1.state
    assert state is not None

    t2 = PPMTransform(maxiter=maxiters, target=target)
    t2.state = state
    y2 = t2.transform(x)

    np.testing.assert_array_almost_equal(y1, y2)


def test_ppmt_de():
    ndim = 4
    ndata = 1000
    maxiters = 10

    np.random.seed(1)

    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))

    pidist = mv_index_distribution(ndata, ndim, friedman_index, ndir=1000)
    target = np.median(pidist)
    
    print(f"target for ({ndata},{ndim})={target}")        

    t = PPMTransform(maxiter=maxiters, target=target, optimiser="de")

    y, x_back = assert_reversibility(t, x)        

    ndim = x.shape[1]
    for dim in range(ndim):
        assert_normality(y[:, dim])
