import numpy as np

from pymgt import *
from test_utils import *

def test_rbigt():
    x = np.loadtxt("../data/synthetic_minerals.csv", delimiter=',', skiprows=1, usecols=[9, 10, 11, 12, 13, 14, 15, 16]) 
    ndata, ndim = x.shape

    maxiter=10

    metric = FRIEDMAN_METRIC

    pi = mv_index_distribution(ndata, ndim, metric, ndir=1000)
    target = np.median(pi)

    t = RBIGTransform(objective=metric, target=target, maxiter=maxiter)

    y,  x_back = assert_reversibility(t,  x, decimal=4)
