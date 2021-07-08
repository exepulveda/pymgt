import numpy as np

from pymgt import *
from pymgt.metrics import *
from pymgt.ppmt_utils import friedman_index


def test_():
    ndata = 1000
    np.random.seed(1)

    x = np.random.uniform(0.0, 1.0, size=ndata)


    metrics = [
        ("friedman", FRIEDMAN_METRIC, False),
        ("kstest", KS_METRIC, False),
        ("anderson", ANDERSON_METRIC, False),
        ("shapiro", SHAPIRO_METRIC, True),
        ("jarque", JARQUE_METRIC, True),
    ]

    for (name, metric, maximising) in metrics:
        assert metric.name == name
        assert metric.maximising == maximising

        pi1 = metric(x)
        pi2, test = metric.compute_test_best(x, pi1/2.0 if maximising else pi1*2.0)

        assert pi1 == pi2
        assert test