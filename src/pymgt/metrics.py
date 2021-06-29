"""Module for implementing metrics
"""

from .interface import AbstractMetric
from .tindex import jarque_bera_index
from .tindex import shapiro_index
from .tindex import anderson_index
from .tindex import ks_index
from .ppmt_utils import friedman_index


class Metric(AbstractMetric):
    """This class implement a metric based on a gaussianity index or test
    """
    def __init__(self, name, index_func, maximising=True):
        super(Metric, self).__init__()
        self.__name = name
        self.__index_func = index_func
        self.__maximising = maximising

    @property
    def name(self):
        """Name of the metric"""
        return self.__name

    @property
    def maximising(self):
        """If the metric must be maximised (if False will be minimised)"""
        return self.__maximising

    def __call__(self, x):
        """Make it callable"""
        return self.__index_func(x)

    def compute_test_best(self, x, target):
        """Compute the test and return if the result is better
        than the `target`"""

        index = self(x)
        if self.__maximising:
            test = index > target
        else:
            test = index < target

        return index, test


FRIEDMAN_METRIC = Metric("friedman", friedman_index, False)
KS_METRIC = Metric("kstest", ks_index, False)
ANDERSON_METRIC = Metric("anderson", anderson_index, False)
SHAPIRO_METRIC = Metric("shapiro", shapiro_index, True)
JARQUE_METRIC = Metric("jarque", jarque_bera_index, True)

DEFAULT_METRICS = [FRIEDMAN_METRIC, KS_METRIC, ANDERSON_METRIC, JARQUE_METRIC]
