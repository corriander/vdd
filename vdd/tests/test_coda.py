import unittest

import numpy as np
from mock import patch, Mock, PropertyMock
from ddt import data, unpack, ddt

from vdd import coda


@ddt
class TestCODA(unittest.TestCase):

    def setUp(self):
        self.inst = inst = coda.CODA()

        characteristics = []
        self.values = values = 1, 10, 2, 7.5, 0.3
        for x in values:
            char = Mock()
            char.value = x
            characteristics.append(char)

        inst._characteristics = tuple(characteristics)

        requirements = []
        self.weights = weights = 0.2, 0.1, 0.4, 0.3
        for wt in weights:
            reqt = Mock()
            reqt.weight = wt
            requirements.append(reqt)

        inst._requirements = tuple(requirements)

        correlation = np.array([[0.1, 0.0, 0.9, 0.3, 0.1],
                                [0.0, 0.9, 0.3, 0.1, 0.1],
                                [0.9, 0.3, 0.1, 0.1, 0.0],
                                [0.3, 0.1, 0.1, 0.0, 0.9]])
        self.correlation = correlation

        # Dummy merit values (these would normally be a fraction).
        self.merit = np.array([[24, 85, 78, 17,  5],
                               [99,  7,  3, 88, 13],
                               [41, 63, 52, 17, 31],
                               [51, 95, 53, 60, 36]])

        class MockRelationship(object):
            merit_preset = None
            def __call__(self, x):
                return self._merit_preset

        for	i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                relationship = MockRelationship()
                relationship.correlation = correlation[i,j]
                relationship._merit_preset = self.merit[i,j]
                inst.matrix[i,j] = relationship

    # ----------------------------------------------------------------
    # Test properties
    # ----------------------------------------------------------------
    def test_matrix__unset(self):
        """Matrix should match shape and contain CODANull by default.
        """
        temp_inst = coda.CODA()
        self.assertEqual(temp_inst.matrix.shape, (0, 0))

        self.assertEqual(self.inst.matrix.shape, (4, 5))

        temp_inst._requirements += (object(),)
        self.assertEqual(temp_inst.matrix.shape, (1, 0))

        for i, j in zip(*map(range, temp_inst.matrix.shape)):
            self.assertIsInstance(temp_inst.matrix[i,j], coda.CODANull)

    def test_characteristics__default(self):
        """Should be an empty tuple by default."""
        temp_inst = coda.CODA()
        self.assertIsInstance(temp_inst.characteristics, tuple)
        self.assertEqual(len(temp_inst.characteristics), 0.0)
        self.assertEqual(len(self.inst.requirements), 4)

    def test_correlation(self):
        """Property converts correlation values in array to a matrix.

        Each design relationship models a correlation between a
        requirement and a characteristic parameter. This  should
        therefore be the same dimensions as the overall coda model,
        i.e. (n, m) where n is the number of requirements, and m the
        number of characteristics.
        """
        self.assertIsInstance(self.inst.correlation, np.matrix)
        self.assertEqual(self.inst.correlation.shape, self.inst.shape)
        self.assertTrue((self.inst.correlation==self.correlation).all())

    @patch('vdd.coda.CODA.satisfaction', new_callable=PropertyMock)
    def test_merit(self, patch):
        """Sum total of weighted requirement satisfaction."""
        patch.return_value = np.arange(5)
        self.assertEqual(self.inst.merit, 10)

    def test_parameter_value(self):
        """A row vector containing characteristic parameter values.

        Characteristics are considered to be columns in the underlying
        coda matrix, so characterstic parameter values should reflect
        this to be unambiguous.
        """
        self.assertIsInstance(self.inst.parameter_value, np.matrix)
        self.assertEqual(self.inst.parameter_value.shape,
                         (1, self.inst.shape[1]))
        self.assertTrue((self.inst.parameter_value==self.values).all())

    @data(
        (np.array([2.0, 10, 2, 7.5, 0.3]), None),
        (np.array([[2.0, 10, 2, 7.5, 0.3]]), None),
        (np.array([[2.0, 10, 2, 7.5, 0.3]]).T, None),
        (np.matrix([2.0, 10, 2, 7.5, 0.3]), None),
        ([2.0, 10, 2, 7.5, 0.3], None),
        (tuple([2.0, 10, 2, 7.5, 0.3]), None),
        (set([2.0, 10, 2, 7.5, 0.3]), ValueError),
        ([2.0, 10, 2, 7.5], ValueError),
    )
    @unpack
    def test_parameter_value__set(self, value, exception):
        self.assertEqual(self.inst.parameter_value[0,0], 1.0)
        if exception is not None:
            self.assertRaises(exception, setattr, self.inst,
                              'parameter_value', value)
        else:
            setattr(self.inst, 'parameter_value', value)
            self.assertEqual(self.inst.parameter_value[0,0], 2.0)

    def test_requirements__default(self):
        """Should be an empty tuple by default."""
        temp_inst = coda.CODA()
        self.assertIsInstance(temp_inst.characteristics, tuple)
        self.assertEqual(len(temp_inst.characteristics), 0.0)
        self.assertEqual(len(self.inst.requirements), 4)

    @patch('vdd.coda.CODA._merit')
    @patch('vdd.coda.CODA.correlation', new_callable=PropertyMock)
    @patch('vdd.coda.CODA.weight', new_callable=PropertyMock)
    def test_satisfaction(self, *mocks):
        """Weighted requirement satisfactions.

        This is the merit of each characteristic parameter value for
        each requirement, weighted by both correlation factors and
        requirement importance weighting.
        """
        weight, correlation, merit = mocks
        weight.return_value = np.matrix([0, 0.4, 0.6]).T

        correlation.return_value = np.matrix(np.ones((3,2)))

        merit.return_value = np.matrix(np.ones((3,2)))

        expected = np.matrix([[0.0 / 2 * (1.0 + 1.0)],
                              [0.4 / 2 * (1.0 + 1.0)],
                              [0.6 / 2 * (1.0 + 1.0)]])

        self.assertIsInstance(self.inst.satisfaction, np.matrix)
        self.assertEqual(self.inst.satisfaction.shape, (3, 1))
        np.testing.assert_array_almost_equal(self.inst.satisfaction,
                                             expected)

    def test_shape(self):
        """Reflects the number of characteristics & requirements.

        A CODA model involves n requirements and m characteristics,
        modelled as an (n, m) array/matrix.
        """
        self.assertEqual(self.inst.shape, (4, 5))

    def test_weight(self):
        """A column vector containing requirement weightings.

        Requirements are considered to be rows in the underlying
        coda matrix, so requirement weights should reflect this to be
        unambiguous.
        """
        self.assertIsInstance(self.inst.weight, np.matrix)
        self.assertEqual(self.inst.weight.shape,
                         (self.inst.shape[0], 1))
        # Note we must transpose the weight column vector to compare
        # it properly with the simple input weights tuple because of
        # numpy broadcasting producing a boolean matrix.
        self.assertTrue((self.inst.weight.T==self.weights).all())

    # ----------------------------------------------------------------
    # Test methods
    # ----------------------------------------------------------------
    @data(
        #[('Irrelevant requirement', 0.0, ValueError),], # not enforced
        [('Unimportant requirement', 0.1, None),],
        [('Important requirement', 0.9, None),],
        [('Unimportant requirement', 0.1, None),
         ('Important requirement', 0.9, None),],
        [('Sole requirement', 1.0, None),],
        [('Sole requirement', 1.0, None),
         ('Another requirement', 0.1, RuntimeError)],
    )
    def test_add_requirement__prenormalised(self, reqts):
        inst = coda.CODA()
        i = 0
        for (name, normwt, exception) in reqts:
            if exception is None:
                inst.add_requirement(name, normwt, normalise=False)
                i += 1
                self.assertEqual(len(inst.requirements), i)
                self.assertEqual(inst.requirements[i-1].name, name)
                self.assertEqual(inst.requirements[i-1].weight,
                                 normwt)
            else:
                self.assertRaises(exception, inst.add_requirement,
                                  name, normwt, normalise=False)

    @data(
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.1, 0.2, 0.3, 0.4)
    )
    def test_add_requirement__unnormalised(self, weights):
        inst = coda.CODA()
        for wt in weights:
            inst.add_requirement('Blah', wt, normalise=True)

        self.assertAlmostEqual(
            sum([r.weight for r in inst.requirements]),
            1.0
        )

    @data(
        [('Characteristic', 0.0, 1.0, None, None),],
        [('Characteristic', 0.0, 1.0, 1.0, None),
         ('Another characteristic', -1.0, 11.0, None, None),],
    )
    def test_add_characteristic(self, chars):
        inst = coda.CODA()
        i = 0
        for (name, llim, ulim, value, exception) in chars:
            if exception is None:
                inst.add_characteristic(name, (llim, ulim), value)
                i += 1
                self.assertEqual(len(inst.characteristics), i)
                self.assertEqual(inst.characteristics[i-1].name, name)
                self.assertEqual(inst.characteristics[i-1].limits,
                                 (llim, ulim))
                # Value not set in these test data.
                #self.assertEqual(inst.characteristics[i-1].value,
                #				 value)
            else:
                self.assertRaises(exception, inst.add_characteristic,
                                  name, normwt)

    @data(
        [(0, 0, 'max', 0.1, 1.0, None, None),],
        [(0, 0, 'min', 0.1, 1.0, None, None),],
        [(0, 0, 'opt', 0.1, 1.0, 1.0, None),],
        [(0, 5, 'opt', 0.1, 1.0, 1.0, KeyError),],
        [(0, 0, 'max', 0.1, 1.0, None, None),
         (0, 1, 'max', 0.1, 1.0, None, None),],
    )
    def test_add_relationship(self, rels):
        inst = self.inst
        for (r, c, type_, corr, tv, tol, exception) in rels:
            if type_ == 'opt':
                cls = coda.CODAOptimise
                args = (r, c, type_, corr, tv, tol)
            else:
                args = (r, c, type_, corr, tv, tol)
                if type_ == 'max':
                    cls = coda.CODAMaximise
                else:
                    cls = coda.CODAMinimise

            if exception is None:
                inst.add_relationship(*args)
            else:
                self.assertRaises(exception, inst.add_relationship,
                                  *args)
                continue

            self.assertIsInstance(inst.matrix[r,c], cls)
            self.assertEqual(inst.matrix[r,c].correlation, corr)

    @data(
        ['Requirement0', 0, None],
        [0, 'Characteristic0', None],
        ['requirement0', 0, KeyError], # Case-sensitive for now.
        ['Requirement1', 0, KeyError], # Not present.
        ['Requirement0', 'Characteristic0', None],
    )
    @unpack
    def test_add_relationship__by_name(self, rlkup, clkup, exception):
        inst = coda.CODA()

        mock1 = Mock()
        mock1.name = 'Requirement0'
        inst._requirements = (mock1,)

        mock2 = Mock()
        mock2.name = 'Characteristic0'
        inst._characteristics = (mock2,)

        if exception is None:
            inst.add_relationship(rlkup, clkup, 'max', 1.0, 1.0)
            self.assertIsInstance(inst.matrix[0,0], coda.CODAMaximise)
        else:
            self.assertRaises(exception, inst.add_relationship,
                              rlkup, clkup, 'max', 1.0, 1.0)

    def test__merit(self):
        """Returns a matrix of merit values for design relationships.

        Each design relationship is a model providing a fractional
        decimal value representing the degree to which a requirement
        is satisfied by a given characteristic parameter value. This
        should therefore be the same dimensions as the overall coda
        model, i.e. (n, m) where n is the number of requirements, and
        m the number of characteristics.

        "Internal" method because raw merit values are not considered
        particularly useful on their own at this point.
        """
        self.assertIsInstance(self.inst._merit(), np.matrix)
        self.assertEqual(self.inst._merit().shape, self.inst.shape)
        self.assertTrue((self.inst._merit()==self.merit).all())


@ddt
class TestCODACharacteristic(unittest.TestCase):

    def setUp(self):
        class CODACharacteristic(coda.CODACharacteristic):
            def __init__(self):
                pass
        self.inst = CODACharacteristic()

    def test___init____omit_value(self):
        """Omitting the value on instantiation is valid.

        When modelling a set of designs (typical) we don't necessarily
        want to seed the model with characteristic values.
        """
        inst = coda.CODACharacteristic('Name')
        # This might want to be None? Requires everything supporting
        # that as an input though.
        self.assertRaises(AttributeError, getattr, inst, 'value')

    @data(
        (-0.01, ValueError),
        (0.0, None),
        (0.5, None),
        (1.0, None),
        (1.01, ValueError),
    )
    @unpack
    def test_value__set_with_default_limits(self, value, exception):
        if exception is not None:
            self.assertRaises(exception, setattr, self.inst, 'value',
                              value)

        else:
            self.inst.value = value
            self.assertEqual(self.inst.value, value)

    def test_limits__get__default(self):
        self.assertEqual(self.inst.limits, self.inst._default_limits)

    @data((0.0, 1.0), [0.0, 1.0], (None, None), (0, None), (None, 1))
    def test_limits__set__valid(self, value):
        self.inst.limits = value
        self.assertEqual(self.inst.limits, tuple(value))


@ddt
class TestCODARequirement(unittest.TestCase):

    def setUp(self):
        class CODARequirement(coda.CODARequirement):
            def __init__(self):
                pass
        self.inst = CODARequirement()

    @data((-0.01, False), (0.0, True), (0.5, True), (1.0, True),
          (1.1, False))
    @unpack
    def test_weight__set(self, wt, valid):
        # Prototypes used context to allow weights to be provided in a
        # non-normalised form and this property would handle the
        # normalisation by inspecting the weights of other
        # requirements. This functionality isn't implemented here, but
        # might still be useful.
        if not valid:
            self.assertRaises(ValueError, setattr, self.inst,
                              'weight', wt)
        else:
            self.inst.weight = wt
            self.assertEqual(self.inst.weight, wt)


@ddt
class TestCODARelationship(unittest.TestCase):

    def setUp(self):
        class Concrete(coda.CODARelationship):
            def __init__(self):
                pass

            def __call__(self, x):
                return 0.0

        self.cls = Concrete
        self.inst = Concrete()

    @data([0.0, 0.0, True],
          [0.1, 0.1,  True],
          [0.3, 0.3, True],
          [0.9, 0.9, True],
          [1.0, 0.1, True],
          [0.25, None, False],
          [-0.1, None, False],
          [0, 0.0, True],
          [1, 0.1, True],
          [3, 0.3, True],
          [9, 0.9, True],
          ['none', 0.0, True],
          [None, 0.0, True],
          ['weak', 0.1, True],
          ['moderate', 0.3, True],
          ['medium', 0.3, True],
          ['strong', 0.9, True],
    )
    @unpack
    def test_correlation(self, value, internal_value, valid):
        """Correlation value must be one of a restricted set."""
        # TODO: It might be more flexible to enforce this further up
        #		for different scaling systems. Could also be done with
        #		a mixin implementation implementation
        self.assertRaises(AttributeError, getattr, self.inst,
                          'correlation')
        if valid:
            self.inst.correlation = value
            self.assertEqual(self.inst.correlation, internal_value)
        else:
            self.assertRaises(ValueError, setattr, self.inst,
                              'correlation', value)

    def test_target(self):
        """Target value may be anything, but check it's settable."""
        self.assertRaises(AttributeError, getattr, self.inst,
                          'target')
        self.inst.target = 0.0
        self.assertEqual(self.inst.target, 0.0)


class TestCODANull(unittest.TestCase):

    def test___init__(self):
        """Takes no arguments, has a correlation and merit of zero."""
        null = coda.CODANull()
        self.assertEqual(null.correlation, 0.0)
        self.assertIs(null.target, None)
        self.assertEqual(null(None), 0.0)

    def test__attributes_not_settable(self):
        null = coda.CODANull()

        self.assertRaises(TypeError, setattr, null, 'correlation', 1)
        self.assertRaises(TypeError, setattr, null, 'target', 1)


class TestCODAMaximise(unittest.TestCase):

    # TODO: compare function over range.

    def test_merit(self):
        inst = coda.CODAMaximise(target=1.0, correlation=None)
        self.assertAlmostEqual(inst(1.0), 0.5)
        self.assertLess(inst(0.1), 0.5)
        self.assertGreater(inst(2.0), 0.5)


class TestCODAMinimise(unittest.TestCase):

    # TODO: compare function over range.

    def test_merit(self):
        inst = coda.CODAMinimise(target=1.0, correlation=None)
        self.assertAlmostEqual(inst(1.0), 0.5)
        self.assertGreater(inst(0.1), 0.5)
        self.assertLess(inst(2.0), 0.5)


class TestCODAOptimise(unittest.TestCase):

    # TODO: compare function over range.

    def test_merit(self):
        inst = coda.CODAOptimise(target=1.0, correlation=None,
                                 tolerance=0.2)
        self.assertAlmostEqual(inst(0.8), 0.5)
        self.assertAlmostEqual(inst(1.2), 0.5)
        self.assertAlmostEqual(inst(1.0), 1.0)
        self.assertGreater(inst(1.1), 0.5)
        self.assertGreater(inst(0.9), 0.5)
        self.assertLess(inst(2.0), 0.5)
        self.assertLess(inst(0.0), 0.5)



if __name__ == '__main__':
    unittest.main()
