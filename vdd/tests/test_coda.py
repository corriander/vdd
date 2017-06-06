import unittest

import numpy as np
from mock import Mock
from ddt import data, unpack, ddt

from vdd import coda


@ddt
class TestCODA(unittest.TestCase):

    def setUp(self):
        self.inst = coda.CODA()


        self.inst._requirements = tuple([object()
                                         for i in range(4)])
        self.inst._characteristics = tuple([object()
                                            for i in range(5)])

    def test_array__unset(self):
        """Array should reflect shape and contain CODANull by default.
        """
        temp_inst = coda.CODA()
        self.assertEqual(temp_inst.array.shape, (0, 0))

        self.assertEqual(self.inst.array.shape, (4, 5))

        self.inst._requirements += (object(),)
        self.assertEqual(self.inst.array.shape, (5, 5))

        for i, j in zip(*map(range, self.inst.array.shape)):
            self.assertIsInstance(self.inst.array[i,j], coda.CODANull)

    def test_characteristics__default(self):
        """Should be an empty tuple by default."""
        temp_inst = coda.CODA()
        self.assertIsInstance(temp_inst.characteristics, tuple)
        self.assertEqual(len(temp_inst.characteristics), 0.0)
        self.assertEqual(len(self.inst.requirements), 4)

    def test_correlation(self):
        """Property converts correlation values in array to a matrix.
        """
        inst = coda.CODA()
        inst._requirements = self.inst._requirements
        inst._characteristics = self.inst._characteristics
        correlation = np.array([[0.1, 0.0, 0.9, 0.3, 0.1],
                                [0.0, 0.9, 0.3, 0.1, 0.1],
                                [0.9, 0.3, 0.1, 0.1, 0.0],
                                [0.3, 0.1, 0.1, 0.0, 0.9]])
        relationship = Mock()

        for	i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                relationship = Mock()
                relationship.correlation = correlation[i,j]
                inst.array[i,j] = relationship

        self.assertTrue((inst.correlation == correlation).all())

    def test_parameter_value(self):
        """Property presents characteristic param. values as a vector.
        """
        inst = coda.CODA()
        characteristics = []
        values = 1, 10, 2, 7.5, 0.3
        for x in values:
            char = Mock()
            char.value = x
            characteristics.append(char)

        inst._requirements = self.inst._requirements
        inst._characteristics = tuple(characteristics)

        self.assertTrue((inst.parameter_value==values).all())

    def test_requirements__default(self):
        """Should be an empty tuple by default."""
        temp_inst = coda.CODA()
        self.assertIsInstance(temp_inst.characteristics, tuple)
        self.assertEqual(len(temp_inst.characteristics), 0.0)
        self.assertEqual(len(self.inst.requirements), 4)

    def test_shape(self):
        """Shape should reflect the characteristics & requirements."""
        self.assertEqual(self.inst.shape, (4, 5))

    def test_weight(self):
        """Property presents requirement weights as a vector."""
        inst = coda.CODA()
        requirements = []
        weights = 0.2, 0.1, 0.4, 0.3
        for wt in weights:
            reqt = Mock()
            reqt.weight = wt
            requirements.append(reqt)

        inst._requirements = tuple(requirements)
        inst._characteristics = self.inst._characteristics

        self.assertTrue((inst.weight==weights).all())


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
