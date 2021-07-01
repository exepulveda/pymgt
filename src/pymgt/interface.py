"""Abstract interfaces (definitions) used in this package.
AbstractMetric is for defining metrics
AbstractTransform is for defining multivariate Gaussian transforms
"""

from abc import ABC, abstractmethod
from typing import NewType, Tuple, List

Vector = List[float]
Array2D = List[Vector]


class AbstractMetric(ABC):
    """Metric: abstract definition of any metric.
    A metric is a measure of departure from a standard
    Guassian distribution of 1-D samples.

    Parameters
    ----------
    name : string, optional
        User name of transform instance

    Attributes
    ----------
    state : any
        The internal representation of the transform. Property that uses
        must be implemented by subclasses.
    """

    @abstractmethod
    def __call__(self, x: Array2D) -> float:
        """Callable interface
        """

    @abstractmethod
    def compute_test_best(self,
                          x: Array2D,
                          target: float) -> Tuple[float, bool]:
        """Compute the metric for `x` and test the `target`
        """


MetricType = NewType('MetricType', AbstractMetric)


class AbstractState(ABC):
    """AbstractState: abstract definition of any state used for serialising.
    """


StateType = NewType('StateType', AbstractState)


class AbstractTransform(ABC):
    """AbstractTransform: abstract definition of any tranformer.
    A transformer is fitted by using an array-like data and then
    it is able to convert any input `x` into a new
    array-like `y`. The transform is perfectable invertible, i.e., x = g(f(x)),
    where f and g are the forward and inverse transform respectively.
    """

    @abstractmethod
    def fit_transform(self, x: Array2D) -> Array2D:
        """"Fit and apply the transform. It should be equivalent to `fit`
        followed bt `transform`.

        Parameters
        ----------
        x : array-like
            Values used to fit the transform.

        Returns
        -------
        y: array-like, same shape as `x`.
            The transformed values applied to `x`.
        """

    @abstractmethod
    def transform(self, x: Array2D) -> Array2D:
        """"Apply the fitted transform to `x`

        Parameters
        ----------
        x : array-like
            Values to transform.

        Returns
        -------
        y: array-like, same shape as `x`.
            The transformed values applied to `x`.
        """

    @abstractmethod
    def inverse_transform(self, y: Array2D) -> Array2D:
        """"Apply the inverse transform to `y`

        Parameters
        ----------
        y : array-like
            Values to transform.

        Returns
        -------
        x: array-like, same shape as `y`.
            The back transformed values applied to `y`.
        """


TransformType = NewType('TransformType', AbstractTransform)
