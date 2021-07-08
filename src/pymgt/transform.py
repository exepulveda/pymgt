"""Module to define Transform class and related classes
"""
from abc import abstractmethod
import numpy as np

from .interface import AbstractTransform, MetricType
from .tindex import uniform_on_surface


class MDMetric:
    """A Multidimensional metric class
    """
    def __init__(self, reduce_func=np.mean, ndir=100, seed=None):
        """Create an object with `reduce_func` for `ndir` directions and `seed` number
        """
        self.__reduce_func = reduce_func
        self.__directions = None
        self.__seed = seed
        self.__ndir = ndir

    def init_directions(self, x, force=False):
        """Initialise the directions according to the dimension of `x`.
        `force` will initialise the directions even if they are already
        initialised
        """
        if force or self.__directions is None:
            _, ndim = x.shape

            if self.__seed is not None:
                np.random.seed(self.__seed)

            # generate the directions
            self.__directions = uniform_on_surface(ndim, self.__ndir)

    def __call__(self, x, metric: MetricType):
        """Compute the `metric` to `x`.
        """
        self.init_directions(x)

        ret = np.array([metric(x@d) for d in self.__directions])

        return self.__reduce_func(ret)

    def compute_test_best(self, x, metric, target):
        """Compute the `metric` to `x` and test it against `target`.
        """
        index = self(x, metric)
        if metric.maximising:
            test = index > target
        else:
            test = index < target

        return index, test


class Transform(AbstractTransform):
    """Transform: still abstract implementation of any transform. A transformer is fitted
    by using an array-like data and then it is able to convert any input `x`
    into a new array-like `y`. The transform is perfectable invertible, i.e.,
    x = g(f(x)), where f and g are the forward and
    inverse transform respectively.

    Attributes
    ----------
    name : string
        User name of transform instance.
    trace : string
        User name of transform instance.
    metrics : list of Metrics
        User name of transform instance.
    _fitted : bool
        The logical indicator if the transform has been fitted. Initialised to `False`.
    state : any
        The internal representation of the transform. Property that uses
        must be implemented by subclasses.
    """

    def __init__(self, **kargs):
        """

        Attributes
        ----------
        name : string
            User name of transform instance. Default to `unnamed`
        trace : bool
            The logical indicator if the transform has been fitted. Default to `False`.
        metrics : list of Metrics
            A list of metrics to display. Default to `None`
        """
        self.__name = kargs.get('name', 'unnamed')
        self.__trace = kargs.get("trace", False)
        self.__metrics = kargs.get("metrics", None)
        self._state = None

        self._fitted = False
        self._mdmetric = MDMetric(kargs.get("reduce_func", np.mean), kargs.get("ndir", 100))
        super().__init__()

    @property
    def tracing(self):
        """Define if traceing is activated
        """
        return self.__trace

    def compute_metrics(self, x, extra=None):
        """Compute to `x` the defined metrics and if present the `extra` metric
        """
        if self.__metrics is None and extra is None:
            return None

        ret = {}
        if self.__metrics is not None:
            for m in self.__metrics:
                ret[m.name] = self._mdmetric(x, m)

        if extra is not None and extra.name not in ret:
            ret[extra.name] = self._mdmetric(x, extra)

        return ret

    def metrics_text(self, x, extra=None):
        """Compute and format a text for the defined metrics to `x`
        and if present to the `extra` metric
        """
        metrics = self.compute_metrics(x, extra=extra)
        if metrics is not None:
            s = ", ".join(["%s=%0.5f"%(k, v) for k, v in metrics.items()])
            return s
        return ""

    @property
    def name(self):
        """The name of the transform
        """
        return self.__name

    @property
    def metrics(self):
        """The metrics used for the transform
        """
        return self.__metrics

    @name.setter
    def name(self, name):
        """Set the transform name
        """
        self.__name = name

    @property
    def state(self):
        """Get the transform state
        """
        return self._state

    @state.setter
    def state(self, state):
        """Set the transform state
        """
        self._state = state
        self._fitted = True # if the state is set, we assume the model is now fitted

    def fit(self, x):
        """"Fit the transform. Uses by default `fit_tranform`

        Parameters
        ----------
        x : numpy array-like
            Values used to fit the transform.
        """
        x = np.asarray(x)
        _ = self.fit_transform(x)

