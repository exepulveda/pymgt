"""
Package pymgt import most used costants, classes and functions
"""
__version__ = "0.1.1"

from .transform import Transform
from .transform import MDMetric

from .nscores import UnivariateGaussianTransform
from .nscores import MarginalGaussianTransform

from .sphering import SpheringTransform
from .rbigt import RBIGTransform

from .ppmt import PPMTransform
from .ppmt import friedman_index

from .tindex import Projectable
from .tindex import jarque_bera_index
from .tindex import shapiro_index
from .tindex import anderson_index
from .tindex import ks_index
from .tindex import mv_index_distribution

from .metrics import Metric, DEFAULT_METRICS
from .metrics import FRIEDMAN_METRIC, KS_METRIC, ANDERSON_METRIC, SHAPIRO_METRIC
