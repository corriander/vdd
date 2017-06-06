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
    def matrix(self):
        """Matrix of relationship functions.

        Matrix automatically adjusts to reflect the number of
        requirements, n, and characteristics, m, such that the shape
        is (n, m). CODANull relationships are used for element values
        by default.
        """
        try:
            matrix = self._matrix

        except AttributeError:
            matrix = self._matrix = self._create_base_matrix()

        shape = matrix.shape
        if shape != self.shape:
            new_matrix = self._create_base_matrix()
            new_matrix[0:shape[0],0:shape[1]] = matrix
            self._matrix = matrix = new_matrix

        return matrix

    @property
    def correlation(self):
        """Correlation matrix."""
        vfunc = np.vectorize(attrgetter('correlation'))
        return vfunc(self.matrix)

    @property
    def characteristics(self):
        """Modelled characteristics."""
        if not hasattr(self, '_characteristics'):
            self._characteristics = ()
        return self._characteristics

    @property
    def merit(self):
        """Overall design merit."""
        return self.satisfaction.sum()

    @property
    def parameter_value(self):
        return np.matrix([[c.value for c in self.characteristics]])

    @property
    def requirements(self):
        """Modelled requirements."""
        if not hasattr(self, '_requirements'):
            self._requirements = ()
        return self._requirements

    @property
    def satisfaction(self):
        """Satisfaction of the requirement for characteristic values.
        """
        nwt = self.weight
        cf = self.correlation
        scf = cf.sum(axis=1)
        mv = self._merit()
        return np.multiply(np.divide(nwt, scf),
                           np.multiply(mv, cf).sum(axis=1))

    @property
    def shape(self):
        """Return shape of model (M, N).

            M - Number of requirements
            N - Number of characteristics
        """
        return len(self.requirements), len(self.characteristics)

    @property
    def weight(self):
        vec = np.matrix([[reqt.weight for reqt in self.requirements]])
        return vec.T # Return as column vector

    def add_requirement(self, name, weight):
        tup = self.requirements
        if (self.weight.sum() + weight) > 1.0:
            raise RuntimeError(
                "Combined requirement weight exceeds unity."
            )
        self._requirements = tup + (CODARequirement(name, weight),)

    def add_characteristic(self, name, limits, value=None):
        tup = self.characteristics
        obj = CODACharacteristic(name, limits, value)
        self._characteristics = tup + (obj,)

    def add_relationship(self, rlkup, clkup, reltype, correlation,
                         target, tolerance=None):
        if reltype != 'opt' and tolerance is not None:
            raise TypeError("Tolerance only valid for optimising.")

        r = self._rc_lookup('r', rlkup)
        c = self._rc_lookup('c', clkup)

        relationships = {
            'max': (CODAMaximise, (correlation, target)),
            'min': (CODAMinimise, (correlation, target)),
            'opt': (CODAOptimise, (correlation, target, tolerance)),
        }

        cls, args = relationships[reltype]
        self.matrix[r,c] = cls(*args)

    def _create_base_matrix(self):
        # Create an array sized by the shape of the coda model and
        # populate with Null relationships.
        array = np.empty(self.shape, dtype=object)
        array[:] = CODANull()
        return np.matrix(array)

    def _merit(self):
        vfunc = np.vectorize(lambda f, x: f(x))
        return vfunc(self.matrix, self.parameter_value)

    def _rc_lookup(self, type_, value):
        switch = {
            'c': (CODACharacteristic, self.characteristics),
            'r': (CODARequirement, self.requirements),
        }

        cls, tup = switch[type_]

        if isinstance(value, cls):
            idx = tup.index(value)
        elif isinstance(value, int):
            shape = self.shape
            if value >= (shape[0] if type_ == 'r' else shape[1]):
                raise KeyError("{} index out of bounds.".format({
                    'r': 'Requirement',
                    'c': 'Characteristic'
                }[type_]))

            idx = value
        else:
            try:
                idx = [item.name for item in tup].index(value)
            except ValueError:
                raise KeyError(
                    "Lookup value must be known {}, its name or "
                    "index.".format(type_)
                )
        return idx


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
