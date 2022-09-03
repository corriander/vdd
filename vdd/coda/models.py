"""CODA Modelling Tools.

Defines classes required for building a Concept Design Analysis model.

References
----------

  - M.H. Eres et al, 2014. Mapping Customer Needs to Engineering
    Characteristics: An Aerospace Perspective for Conceptual Design -
    Journal of Engineering Design pp. 1-24
"""
from operator import attrgetter
import abc
from collections.abc import Sequence

import numpy as np

from . import io


class CODA(object):

    @classmethod
    def from_excel(cls, path, parser_class=io.CompactExcelParser):
        """Construct a CODA model from an Excel workbook.

        Parameters
        ----------

        path : str
            Filesystem path to the Excel workbook

        Returns
        -------

        CODA
            Populated CODA model
        """
        return cls.read_excel(path, parser_class)

    @classmethod
    def from_google_sheet(cls, workbook_name):
        """Construct a CODA model from a Google Sheet.

        Parameters
        ----------

        workbook_name : str
            Name of shared Google Sheet

        Returns
        -------

        CODA
            Populated CODA model

        Notes
        -----

        The Google Sheet must be shared with a service account for
        which credentials are provided in the configuration
        directory.
        """
        sheet = cls._get_sheet(workbook_name)
        model = cls()
        return cls._transfer_elements(model, sheet)

    @classmethod
    def read_excel(cls, path, parser_class=io.CompactExcelParser):
        """Import model from spreadsheet."""
        parser = parser_class(path)
        model = cls()
        return cls._transfer_elements(model, parser)

    @staticmethod
    def _transfer_elements(inst, source):
        # Helper method for the constructors.
        for element in 'requirement','characteristic','relationship':
            for args in getattr(source, 'get_{}s'.format(element))():
                if element == 'characteristic':
                    # add_characteristics has a bounds parameter, not
                    # separate min, max params.
                    args = (args[0],) + (args[1:3],)
                getattr(inst, 'add_{}'.format(element))(*args)
        return inst

    @staticmethod
    def _get_sheet(workbook_name):
        # TODO: remove redundancy with BinWM
        return io.GSheetCODA(workbook_name)

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
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
        """Correlation matrix.

        Defines the strength of the relationship between a requirement
        and characteristic.
        """
        vfunc = np.vectorize(attrgetter('correlation'))
        return vfunc(self.matrix)

    @property
    def characteristics(self):
        """Tuple of modelled characteristics."""
        if not hasattr(self, '_characteristics'):
            self._characteristics = ()
        return self._characteristics

    @property
    def merit(self):
        """Overall design merit.

        This is a value in the range [0.0, 1.0]. Note that this is a
        relative value and 50% merit can often represent a good
        mapping between requirements and characteristics.
        """
        # FIXME: Ignore requirements without relationships! They will
        #        result in nan and break this.
        return np.multiply(self.weight, self.satisfaction).sum()

    @property
    def parameter_value(self):
        """Row vector of characteristic parameter values.

        Parameter values for all characteristics may be set in bulk
        via this property via a sequence or matrix.
        """
        # XXX: It would be really nice if the mutability of this
        #	   propagated down.
        return np.array([[c.value for c in self.characteristics]])
    @parameter_value.setter
    def parameter_value(self, value):
        m = self.shape[1]

        if isinstance(value, (np.ndarray, np.matrix)):
            value = value.squeeze()	# Normalises the matrix to (1,N)

        if isinstance(value, np.matrix):
            value = value.tolist()[0]

        if (len(value) == m and
            isinstance(value, (Sequence, np.ndarray))):
            try:
                for x, c in zip(value, self.characteristics):
                    c.value = x
            except Exception as err:
                for x, c in zip(value, self.characteristics):
                    c.value = x
                raise
        else:
            raise ValueError("{} values must be provided as a 1D "
                             "sequence.".format(self.shape[1]))

    @property
    def requirements(self):
        """Tuple of modelled requirements."""
        if not hasattr(self, '_requirements'):
            self._requirements = ()
        return self._requirements

    @property
    def satisfaction(self):
        """Column vector of aggregate requirement satisfaction.

        Each characteristic meets the requirement to some level of
        satisfaction by virtue of its numerical value, the type of
        relationship and strength of correlation (possibly no
        satisfaction of a requirement for a null relationship). This
        is the aggregate contribution of all characteristics for each
        requirement.
        """
        cf = self.correlation
        scf = cf.sum(axis=1)
        mv = self._merit()
        array = np.divide(np.multiply(mv, cf).sum(axis=1), scf)
        vector = array[:,np.newaxis]
        return vector

    @property
    def shape(self):
        """Return shape of model (M, N).

            M - Number of requirements
            N - Number of characteristics
        """
        return len(self.requirements), len(self.characteristics)

    @property
    def weight(self):
        """Column vector of requirement weightings.

        Each requirement contributions to the overall model according
        to its weight.
        """
        vec = np.array([[reqt.weight for reqt in self.requirements]])
        return vec.T # Return as column vector

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def add_requirement(self, name, weight, normalise=True):
        """Add a requirement to the model.

            name: str
                Requirement name (e.g. 'lightweight')

            weight: real
                Weight of requirement.

            normalise: bool
                If true, the specified requirement weight does not
                need to be pre-normalised.

        Note: Null relationships between this requirement and
        characteristics are assumed initially.
        """
        tup = self.requirements
        if name in [c.name for c in tup]:
            raise ValueError("Requirement of this name exists.")

        if normalise:
            cls = CODARequirementNorm
            if any(map(lambda o: not isinstance(o, cls), tup)):
                raise RuntimeError(
                    "normalise parameter must be True for all requirements."
                )
        else:
            cls = CODARequirement
            if (self.weight.sum() + weight) > 1.0:
                raise RuntimeError(
                    "Combined requirement weight exceeds unity."
                )
        self._requirements = tup + (
            cls(context=self, name=name, weight=weight),
        )

    def add_characteristic(self, name, limits=None, value=None):
        """Add a characteristic to the model.

            name: str
                Name of the characteristic (e.g. 'mass')

            limits: 2-tuple
                Lower and upper limits for the characteristic
                numerical value (either or both may be None).

            value: real
                Value of the characteristic parameter. Does not need
                to be specified on characteristic creation.

        Note: Like requirements, new characteristics will initially
        have a null relationship with existing requirements.
        """
        tup = self.characteristics
        if name in [c.name for c in tup]:
            raise ValueError(
                "A characteristic with name '{}' has already been "
                "defined on this model.".format(name)
            )
        obj = CODACharacteristic(name, limits, value, context=self)
        self._characteristics = tup + (obj,)

    def add_relationship(self, rlkup, clkup, reltype, correlation,
                         target, tolerance=None):
        """Define a requirement-characteristic relationship.

            rlkup: int | str
                Index or name of the requirement.

            clkip: int | str
                Index or name of the characteristic.

            reltype: str {'min', 'max', 'opt'}
                Define the relationship as minimising, maximising or
                optimising.

            correlation: str | real
                Define the strength of the relationship (weak,
                moderate or strong). Must be in the set of supported
                correlation values.

            target: real
                If an optimising relationship, this is the value at
                which 100% requirement satisfaction is obtained. If
                minimising or maximising, this is the "neutral point"
                where 50% satisfaction is obtained (neither good or
                bad).

            tolerance: real
                If an optimising relationship, this is

                    tp +- tol = 50%

                An exception is raised if this is specified for a
                non-optimising relationship.
        """
        # TODO: Document the valid correlations!

        if reltype != 'opt' and tolerance is not None:
            raise TypeError("Tolerance only valid for optimising.")

        r = self._rc_lookup('requirement', rlkup)
        c = self._rc_lookup('characteristic', clkup)

        relationships = {
            'max': (CODAMaximise, (correlation, target)),
            'min': (CODAMinimise, (correlation, target)),
            'opt': (CODAOptimise, (correlation, target, tolerance)),
        }

        cls, args = relationships[reltype]
        self.matrix[r,c] = cls(*args)

    def compare(self, other):
        """Return True if the model matrix is the same as another's.
        """
        return self.matrix == other.matrix

    def _create_base_matrix(self):
        # Create an array sized by the shape of the coda model and
        # populate with Null relationships.
        array = np.empty(self.shape, dtype=object)
        array[:] = CODANull()
        return array

    def _merit(self):
        vfunc = np.vectorize(lambda f, x: f(x))
        return vfunc(self.matrix, self.parameter_value)

    def _rc_lookup(self, type_, value):
        type_title = type_.capitalize()
        cls = eval('CODA{}'.format(type_title))
        tup = getattr(self, '{}s'.format(type_))

        if isinstance(value, cls):
            idx = tup.index(value)
        elif isinstance(value, int):
            shape = self.shape
            if value >= shape[['r','c'].index(type_[0])]:
                raise KeyError(
                    "{} index out of bounds.".format(type_title)
                )

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


class CODAElement(object):

    def __init__(self, name, context=None):
        self.name = name
        self.context = context


class CODACharacteristic(CODAElement):

    _default_limits = (0.0, 1.0)

    def __init__(self, name, limits=None, value=None, context=None):
        """
            name: str
                Identifier/description

            limits: 2-tuple (or list)
                Constrains the value.

            value: real
                Characteristic parameter value, e.g. mass.
        """
        super(CODACharacteristic, self).__init__(name, context)
        if limits is not None:
            self.limits = limits
        if value is not None:
            self.value = value

    @property
    def limits(self):
        """Lower and upper bounds of the parameter value."""
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

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        s = self.name.encode('utf-8')
        # TODO: Distinguish between range types.
        s += u' [{},{}] '.format(*self.limits)

        pv = '?'
        try:
            if self.value is not None:
                pv = '{:.5f}'.format(self.value)
        except AttributeError:
            pass

        s += u': {}'.format(pv)
        return s


class CODARequirement(CODAElement):

    def __init__(self, name, weight, **kwargs):
        """
            name: str
                Identifier/description

            weight: fraction (0.0 - 1.0)
                Normalised importance weighting.
        """
        super(CODARequirement, self).__init__(name, **kwargs)
        self.weight = weight

    @property
    def weight(self):
        """Normalised requirement weight."""
        return self._weight
    @weight.setter
    def weight(self, x):
        if x > 1 or x < 0:
            raise ValueError("Weight must be normalised 0 <= x <= 1.")
        self._weight = x


class CODARequirementNorm(CODAElement):
    """Self-normalising requirement.

    Requirement weight is normalised against other requirements in the
    model.
    """

    def __init__(self, context, name, weight=1.0):
        """
            name: str
                Identifier/description

            context: CODA instance
                Provides access to the CODA model for improved
                functionality.

            weight: real (> 0)
                Importance weighting, this is normalised.
        """
        super(CODARequirementNorm, self).__init__(name, context)
        self.base_weight = weight

    @property
    def weight(self):
        """Normalised requirement weight."""
        sum_weights =  sum([r.base_weight
                            for r in self.context.requirements])
        return self.base_weight / sum_weights

    @property
    def base_weight(self):
        return self._base_weight
    @base_weight.setter
    def base_weight(self, x):
        if x < 0:
            raise ValueError("Weight must be positive.")
        self._base_weight = x


class CODARelationship(object):
    """Relationship between a requirement and characteristic.

    Concrete implementations of this class are callables returning
    merit.
    """
    __metaclass__ = abc.ABCMeta

    __correlation_map = {
        external: internal
        for internal, externals in {
            0.0: [0, None, 'none', ''],
            0.1: [1, 0.1, 'weak', '+', 'o', '-'],
            0.3: [3, 0.3, 'moderate', 'medium', '++', 'oo', '--'],
            0.9: [9, 0.9, 'strong', '+++', 'ooo', '---'],
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
        """Strength of requirement-characteristic relationship."""
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
        """Target value for relationship.

        This is 50% satisfaction for maximising and minimising
        relationships, 100% for optimising relationships.
        """
        return self._target
    @target.setter
    def target(self, value):
        self._target = value

    @abc.abstractmethod
    def __call__(self, x):
        return 0.0

    def __eq__(self, other):
        return (self.correlation == other.correlation and
                self.target == other.target)


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
    """Models an optimising characteristic-requirement relationship.

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

    def __eq__(self, other):
        return (super(CODAOptimise, self).__eq__(other) and
                self.tolerance == other.tolerance)

