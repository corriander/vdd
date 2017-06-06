"""CODA Modelling Tools.

References
----------

  - M.H. Eres et al, 2014. Mapping Customer Needs to Engineering
    Characteristics: An Aerospace Perspective for Conceptual Design -
    Journal of Engineering Design pp. 1-24
"""
from __future__ import division

from operator import attrgetter
import abc

import numpy as np


class CODA(object):

    @property
    def array(self):
        """2D array of relationship functions."""
        try:
            array = self._array

        except AttributeError:
            array = self._array = self._create_array()

        shape = array.shape
        if shape != self.shape:
            new_array = self._create_array()
            new_array[0:shape[0],0:shape[1]] = array
            self._array = array = new_array

        return array

    @property
    def correlation(self):
        """Correlation matrix."""
        vfunc = np.vectorize(attrgetter('correlation'))
        return np.matrix(vfunc(self.array))

    @property
    def characteristics(self):
        """Modelled characteristics."""
        if not hasattr(self, '_characteristics'):
            self._characteristics = ()
        return self._characteristics

    @property
    def parameter_value(self):
        return np.array([c.value for c in self.characteristics])

    @property
    def requirements(self):
        """Modelled requirements."""
        if not hasattr(self, '_requirements'):
            self._requirements = ()
        return self._requirements

    @property
    def shape(self):
        """Return shape of model (M, N).

            M - Number of requirements
            N - Number of characteristics
        """
        return len(self.requirements), len(self.characteristics)

    @property
    def weight(self):
        return np.array([[reqt.weight for reqt in self.requirements]])

    def _create_array(self):
        # Create an array sized by the shape of the coda model and
        # populate with Null relationships.
        array = np.empty(self.shape, dtype=object)
        array[:] = CODANull()
        return array


class CODACharacteristic(object):

    _default_limits = (0.0, 1.0)

    def __init__(self, name, limits=None, value=None):
        """
            name: str
                Identifier/description

            limits: 2-tuple (or list)
                Constrains the value.

            value: real
                Characteristic parameter value, e.g. mass.
        """
        self.name = name
        if limits is not None:
            self.limits = limits
        if value is not None:
            self.value = value

    @property
    def limits(self):
        # TODO: Consider exposing limits elements (llim & ulim) also.
        try:
            return self._limits
        except AttributeError:
            self._limits = self._default_limits
            return self._limits
    @limits.setter
    def limits(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            self._limits = tuple(value)

    @property
    def value(self):
        """Characteristic parameter value."""
        # TODO: Support units, e.g. via pint
        try:
            return self._value
        except AttributeError:
            raise AttributeError("value not set.")
    @value.setter
    def value(self, x):
        llim, ulim = self.limits

        msg_base = "value must satisfy {}x{}"
        msg = msg_base.format(
            '{} <= '.format(llim) if llim is not None else '',
            ' <= {}'.format(ulim) if ulim is not None else ''
        )

        if llim is not None and x < llim:
            raise ValueError(msg)

        if ulim is not None and x > ulim:
            raise ValueError(msg)

        self._value = x


class CODARequirement(object):

    def __init__(self, name, weight=1.0):
        """
            name: str
                Identifier/description

            weight: fraction (0.0 - 1.0)
                Normalised importance weighting.
        """

        self.name = name	# Duplication with Characteristic
        self.weight = weight

    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, x):
        if x > 1 or x < 0:
            raise ValueError("Weight must be normalised 0 <= x <= 1.")
        self._weight = x


class CODARelationship(object):
    """Relationship between a requirement and characteristic.

    Concrete implementations of this class are callables returning
    merit.
    """
    __metaclass__ = abc.ABCMeta

    __correlation_map = {
        external: internal
        for internal, externals in {
            0.0: [0, None, 'none'],
            0.1: [1, 0.1, 'weak'],
            0.3: [3, 0.3, 'moderate', 'medium'],
            0.9: [9, 0.9, 'strong'],
        }.items()
        for external in externals
    }

    def __init__(self, correlation, target):
        """
            correlation: real in {0.0, 0.1, 0.3, 0.9}
                Correlation strength between requirement and
                characteristic.

            target: real
                Target value of characteristic parameter.
        """
        self.correlation = correlation
        self.target = target

    @property
    def correlation(self):
        return self._correlation
    @correlation.setter
    def correlation(self, value):
        try:
            self._correlation = self.__correlation_map[value]
        except KeyError:
            valid_set = set(self.__correlation_map.keys())
            raise ValueError(
                "Correlation must be in set {}".format(valid_set)
            )

    @property
    def target(self):
        return self._target
    @target.setter
    def target(self, value):
        self._target = value

    @abc.abstractmethod
    def __call__(self, x):
        return 0.0


class CODANull(CODARelationship):
    """Null relationship.

    Models the absence of a requirement-characteristic relationship,
    in other words the characteristic has no bearing on the
    requirement.
    """

    def __init__(self):
        super(CODANull, self).__init__(0.0, None)

    @CODARelationship.correlation.setter
    def correlation(self, value):
        if value != 0.0:
            raise TypeError("Fixed correlation value.")
        else:
            self._correlation = value

    @CODARelationship.target.setter
    def target(self, value):
        if value is not None:
            raise TypeError("Fixed target value.")
        else:
            self._target = value

    def __call__(self, x):
        return 0.0


class CODAMaximise(CODARelationship):
    """Models a maximising characteristic-requirement relationship.

    For this type of relationship, the goal is maximise the
    characteristic parameter value in order to best satisfy
    requirements.

    Here the target value is the neutral point where the target
    value represents 50% satisfaction of the requirement (i.e. not
    bad, not good).
    """

    def __call__(self, x):
        """Return the merit of parameter value to be maximised.

            x: real
                Parameter value; results in 50% merit if at the target
                point.
        """
        return 1 - (1. / 2**(x/self.target))


class CODAMinimise(CODARelationship):
    """Models a minimising characteristic-requirement relationship.

    For this type of relationship, the goal is minimise the
    characteristic parameter value in order to best satisfy
    requirements.

    Here the target value is the neutral point where the target
    value represents 50% satisfaction of the requirement (i.e. not
    bad, not good).
    """

    def __call__(self, x):
        """Return the merit of parameter value to be optimised.

            x: real
                Parameter value; results in 50% merit if at the target
                point.
        """
        return 1 - (1. / 2**(self.target/x))


class CODAOptimise(CODARelationship):
    """Models a opimitising characteristic-requirement relationship.

    For this type of relationship, the goal is optimise the
    characteristic parameter value in order to best satisfy
    requirements.

    Here the target value is the optimum point, op, where the target
    value represents 100% satisfaction of the requirement and
    tolerance is the variance either side of this optimum point
    representing 50% merit.
    """

    def __init__(self, correlation, target, tolerance=0):
        self.tolerance = tolerance
        super(CODAOptimise, self).__init__(correlation, target)

    def __call__(self, x):
        """Return the merit of parameter value to be optimised.

            x: real
                Parameter value; results in 100% merit if at the
                target point.
        """
        return 1. / (1 + ((x - self.target) / self.tolerance)**2)
