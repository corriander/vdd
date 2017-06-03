"""CODA Modelling Tools.

References
----------

  - M.H. Eres et al, 2014. Mapping Customer Needs to Engineering
    Characteristics: An Aerospace Perspective for Conceptual Design -
    Journal of Engineering Design pp. 1-24
"""
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
    def characteristics(self):
        """Modelled characteristics."""
        if not hasattr(self, '_characteristics'):
            self._characteristics = ()
        return self._characteristics

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

    def _create_array(self):
        # Create an array sized by the shape of the coda model and
        # populate with Null relationships.
        array = np.empty(self.shape, dtype=object)
        array[:] = CODANull()
        return array


class CODACharacteristic(object):

    _default_limits = (0.0, 1.0)

    def __init__(self, name, limits=None, value=None):
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


class CODARelationship(object):
    """Relationship between a requirement and characteristic.

    Concrete implementations of this class are callables returning
    merit.
    """
    __metaclass__ = abc.ABCMeta

    __valid_correlations = {0.0, 0.1, 0.3, 0.9}

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
        if value not in self.__valid_correlations:
            raise ValueError("Invalid correlation value.")
        self._correlation = value

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
