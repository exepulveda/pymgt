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


def test_jura_de():
    #xloc,yloc,long,lat,landuse,rock,Cd,Co,Cr,Cu,Ni,Pb,Zn
    x = np.loadtxt("data/jura.csv", delimiter=' ', skiprows=1, usecols=[6, 7, 8, 9, 10, 11, 12]) 
    ndata, ndim = x.shape        

    maxiters = 10

    np.random.seed(1)

    pidist = mv_index_distribution(ndata, ndim, friedman_index, ndir=1000)
    target = np.median(pidist)
    
    print(f"target for ({ndata},{ndim})={target}")        

    t = PPMTransform(maxiter=maxiters, target=target, optimiser="de")

    y, x_back = assert_reversibility(t, x)        

    ndim = x.shape[1]
    for dim in range(ndim):
        assert_normality(y[:, dim])

def test_synthetic_minerals():
    x = np.loadtxt("data/synthetic_minerals.csv", delimiter=',', skiprows=1, usecols=[9, 10, 11, 12, 13, 14, 15, 16]) 
    ndata, ndim = x.shape

    maxiters = 10

    np.random.seed(1)

    pidist = mv_index_distribution(ndata, ndim, friedman_index, ndir=1000)
    target = np.median(pidist)
    
    print(f"target for ({ndata},{ndim})={target}")

    t = PPMTransform(maxiter=maxiters, target=target)

    y, x_back = assert_reversibility(t, x)        

    ndim = x.shape[1]
    for dim in range(ndim):
        assert_normality(y[:, dim])
