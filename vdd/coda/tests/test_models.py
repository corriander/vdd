import os

import numpy as np
import pytest

from .. import models
from .. import io
from . import DATA_DIR


class TestCODA:

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.inst = inst = models.CODA()

        characteristics = []
        self.values = values = 1, 10, 2, 7.5, 0.3
        for x in values:
            char = mocker.Mock()
            char.value = x
            characteristics.append(char)

        inst._characteristics = tuple(characteristics)

        requirements = []
        self.weights = weights = 0.2, 0.1, 0.4, 0.3
        for wt in weights:
            reqt = mocker.Mock()
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

        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                relationship = MockRelationship()
                relationship.correlation = correlation[i, j]
                relationship._merit_preset = self.merit[i, j]
                inst.matrix[i, j] = relationship

    # ----------------------------------------------------------------
    # Test properties
    # ----------------------------------------------------------------
    def test_matrix__unset(self):
        """Matrix should match shape and contain CODANull by default."""
        temp_inst = models.CODA()
        assert temp_inst.matrix.shape == (0, 0)
        assert self.inst.matrix.shape == (4, 5)

        temp_inst._requirements += (object(),)
        assert temp_inst.matrix.shape == (1, 0)

        for i, j in zip(*map(range, temp_inst.matrix.shape)):
            assert isinstance(temp_inst.matrix[i, j], models.CODANull)

    def test_characteristics__default(self):
        """Should be an empty tuple by default."""
        temp_inst = models.CODA()
        assert isinstance(temp_inst.characteristics, tuple)
        assert len(temp_inst.characteristics) == 0
        assert len(self.inst.requirements) == 4

    def test_correlation(self):
        """Property converts correlation values in array to a matrix.

        Each design relationship models a correlation between a
        requirement and a characteristic parameter. This should
        therefore be the same dimensions as the overall coda model,
        i.e. (n, m) where n is the number of requirements, and m the
        number of characteristics.
        """
        assert isinstance(self.inst.correlation, np.ndarray)
        assert self.inst.correlation.ndim == 2
        assert self.inst.correlation.shape == self.inst.shape
        assert (self.inst.correlation == self.correlation).all()

    def test_merit(self, mocker):
        """Sum total of weighted requirement satisfaction."""
        patch = mocker.patch.object(
            models.CODA, 'satisfaction', new_callable=mocker.PropertyMock
        )
        patch.return_value = np.arange(5)
        assert self.inst.merit == pytest.approx(10)

    def test_parameter_value(self):
        """A row vector containing characteristic parameter values.

        Characteristics are considered to be columns in the underlying
        coda matrix, so characteristic parameter values should reflect
        this to be unambiguous.
        """
        assert isinstance(self.inst.parameter_value, np.ndarray)
        assert self.inst.parameter_value.shape == (1, self.inst.shape[1])
        assert (self.inst.parameter_value == self.values).all()

    @pytest.mark.parametrize("value, exception", [
        (np.array([2.0, 10, 2, 7.5, 0.3]), None),
        (np.array([[2.0, 10, 2, 7.5, 0.3]]), None),
        (np.array([[2.0, 10, 2, 7.5, 0.3]]).T, None),
        ([2.0, 10, 2, 7.5, 0.3], None),
        (tuple([2.0, 10, 2, 7.5, 0.3]), None),
        (set([2.0, 10, 2, 7.5, 0.3]), ValueError),
        ([2.0, 10, 2, 7.5], ValueError),
    ])
    def test_parameter_value__set(self, value, exception):
        assert self.inst.parameter_value[0, 0] == 1.0
        if exception is not None:
            with pytest.raises(exception):
                setattr(self.inst, 'parameter_value', value)
        else:
            setattr(self.inst, 'parameter_value', value)
            assert self.inst.parameter_value[0, 0] == 2.0

    def test_requirements__default(self):
        """Should be an empty tuple by default."""
        temp_inst = models.CODA()
        assert isinstance(temp_inst.characteristics, tuple)
        assert len(temp_inst.characteristics) == 0
        assert len(self.inst.requirements) == 4

    def test_satisfaction(self, mocker):
        """Weighted requirement satisfactions.

        This is the merit of each characteristic parameter value for
        each requirement, weighted by correlation factors.

        .. math::

            \\frac{\\sum_{j=1}^{M} cf .* \\eta}{{scf}_i}

        Where

            i = [1..n]
            j = [1..m]

        and

            n = number of requirements
            m = number of characteristics
        """
        mock_merit = mocker.patch.object(models.CODA, '_merit')
        mock_correlation = mocker.patch.object(
            models.CODA, 'correlation', new_callable=mocker.PropertyMock
        )

        a = np.random.rand(3, 2)
        mock_correlation.return_value = mock_merit.return_value = a

        # numerator
        num = np.multiply(a, a).sum(axis=1, keepdims=True)

        # denominator
        den = a.sum(axis=1, keepdims=True)

        expected = np.divide(num, den)

        assert isinstance(self.inst.satisfaction, np.ndarray)
        assert self.inst.satisfaction.shape == (3, 1)
        np.testing.assert_array_almost_equal(self.inst.satisfaction, expected)

    def test_shape(self):
        """Reflects the number of characteristics & requirements.

        A CODA model involves n requirements and m characteristics,
        modelled as an (n, m) array/matrix.
        """
        assert self.inst.shape == (4, 5)

    def test_weight(self):
        """A column vector containing requirement weightings.

        Requirements are considered to be rows in the underlying
        coda matrix, so requirement weights should reflect this to be
        unambiguous.
        """
        assert isinstance(self.inst.weight, np.ndarray)
        assert self.inst.weight.shape == (self.inst.shape[0], 1)
        # Note we must transpose the weight column vector to compare
        # it properly with the simple input weights tuple because of
        # numpy broadcasting producing a boolean matrix.
        assert (self.inst.weight.T == self.weights).all()

    # ----------------------------------------------------------------
    # Test methods
    # ----------------------------------------------------------------
    @pytest.mark.parametrize("reqts", [
        #[('Irrelevant requirement', 0.0, ValueError),], # not enforced
        [('Unimportant requirement', 0.1, None),],
        [('Important requirement', 0.9, None),],
        [('Unimportant requirement', 0.1, None),
         ('Important requirement', 0.9, None),],
        [('Sole requirement', 1.0, None),],
        [('Sole requirement', 1.0, None),
         ('Another requirement', 0.1, RuntimeError)],
    ])
    def test_add_requirement__prenormalised(self, reqts):
        inst = models.CODA()
        i = 0
        for (name, normwt, exception) in reqts:
            if exception is None:
                inst.add_requirement(name, normwt, normalise=False)
                i += 1
                assert len(inst.requirements) == i
                assert inst.requirements[i-1].name == name
                assert inst.requirements[i-1].weight == normwt
            else:
                with pytest.raises(exception):
                    inst.add_requirement(name, normwt, normalise=False)

    @pytest.mark.parametrize("weights", [
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.1, 0.2, 0.3, 0.4)
    ])
    def test_add_requirement__unnormalised(self, weights):
        inst = models.CODA()
        for i, wt in enumerate(weights):
            inst.add_requirement('Blah'+str(i), wt, normalise=True)

        assert sum([r.weight for r in inst.requirements]) == pytest.approx(1.0)

    @pytest.mark.parametrize("chars", [
        [('Characteristic', 0.0, 1.0, None, None),],
        [('Characteristic', 0.0, 1.0, 1.0, None),
         ('Another characteristic', -1.0, 11.0, None, None),],
    ])
    def test_add_characteristic(self, chars):
        inst = models.CODA()
        i = 0
        for (name, llim, ulim, value, exception) in chars:
            if exception is None:
                inst.add_characteristic(name, (llim, ulim), value)
                i += 1
                assert len(inst.characteristics) == i
                assert inst.characteristics[i-1].name == name
                assert inst.characteristics[i-1].limits == (llim, ulim)
                # Value not set in these test data.
                #assert inst.characteristics[i-1].value == value
            else:
                with pytest.raises(exception):
                    inst.add_characteristic(name, (llim, ulim), value)

    @pytest.mark.parametrize("rels", [
        [(0, 0, 'max', 0.1, 1.0, None, None),],
        [(0, 0, 'min', 0.1, 1.0, None, None),],
        [(0, 0, 'opt', 0.1, 1.0, 1.0, None),],
        [(0, 5, 'opt', 0.1, 1.0, 1.0, KeyError),],
        [(0, 0, 'max', 0.1, 1.0, None, None),
         (0, 1, 'max', 0.1, 1.0, None, None),],
    ])
    def test_add_relationship(self, rels):
        inst = self.inst
        for (r, c, type_, corr, tv, tol, exception) in rels:
            if type_ == 'opt':
                cls = models.CODAOptimise
                args = (r, c, type_, corr, tv, tol)
            else:
                args = (r, c, type_, corr, tv, tol)
                if type_ == 'max':
                    cls = models.CODAMaximise
                else:
                    cls = models.CODAMinimise

            if exception is None:
                inst.add_relationship(*args)
            else:
                with pytest.raises(exception):
                    inst.add_relationship(*args)
                continue

            assert isinstance(inst.matrix[r, c], cls)
            assert inst.matrix[r, c].correlation == corr

    @pytest.mark.parametrize("rlkup, clkup, exception", [
        ['Requirement0', 0, None],
        [0, 'Characteristic0', None],
        ['requirement0', 0, KeyError],  # Case-sensitive for now.
        ['Requirement2', 0, KeyError],  # Not present.
        ['Requirement0', 'Characteristic0', None],
        ['Requirement1', 'Characteristic0', None],
        [1, 'Characteristic0', None],
    ])
    def test_add_relationship__by_name(self, rlkup, clkup, exception, mocker):
        """Given two requirements, 1 characteristic - add relations."""
        inst = models.CODA()

        mock1 = mocker.Mock()
        mock1.name = 'Requirement0'
        mock3 = mocker.Mock()
        mock3.name = 'Requirement1'
        inst._requirements = (mock1, mock3)

        mock2 = mocker.Mock()
        mock2.name = 'Characteristic0'
        inst._characteristics = (mock2,)

        if exception is None:
            inst.add_relationship(rlkup, clkup, 'max', 1.0, 1.0)
            r = rlkup if isinstance(rlkup, int) else int(rlkup[-1])
            assert isinstance(inst.matrix[r, 0], models.CODAMaximise)
        else:
            with pytest.raises(exception):
                inst.add_relationship(rlkup, clkup, 'max', 1.0, 1.0)

    def test_read_excel(self, mocker):
        """Constructor adds elements in turn from the parser.

        The parser provides three methods:

          - get_requirements
          - get_characteristics
          - get_relationships

        These all return records defined within io.CODASheet.

        The constructor calls these methods on the parser and uses the
        results to feed arguments to the add_requirement,
        add_characteristic and add_relationship methods on the CODA
        class.

        This unit test mocks the parser and ensures the known return
        values for these get methods are passed to the add methods in
        the correct fashion.
        """
        mock_add_requirement = mocker.patch.object(
            models.CODA, 'add_requirement'
        )
        mock_add_characteristic = mocker.patch.object(
            models.CODA, 'add_characteristic'
        )
        mock_add_relationship = mocker.patch.object(
            models.CODA, 'add_relationship'
        )

        dummy_records = {
            'requirements': [
                io.CODASheet.ReqRecord('Requirement 1', 0.33),
                io.CODASheet.ReqRecord('Requirement 2', 0.5),
                io.CODASheet.ReqRecord('Requirement 3', 0.17),
            ],
            'characteristics': [
                io.CODASheet.CDefRecord('Characteristic 1', 1, 5),
                io.CODASheet.CDefRecord('Characteristic 2', 10, 20),
            ],
            'relationships': [
                io.CODASheet.MinMaxRelRecord(
                    'Requirement 1',
                    'Characteristic 1',
                    'min',
                    '---',  # TODO: Remove redundant information
                    3
                ),
                io.CODASheet.OptRelRecord(
                    'Requirement 2',
                    'Characteristic 1',
                    'opt',
                    'ooo',
                    13,
                    1,
                ),
                io.CODASheet.MinMaxRelRecord(
                    'Requirement 3',
                    'Characteristic 2',
                    'max',
                    '+++',
                    3
                ),
            ]
        }

        stub_parser = mocker.MagicMock(spec_set=io.CompactExcelParser)
        for s in 'requirements', 'characteristics', 'relationships':
            method = getattr(stub_parser, 'get_{}'.format(s))
            method.return_value = dummy_records[s]
        mock_parser_class = mocker.Mock()
        mock_parser_class.return_value = stub_parser

        sut = models.CODA.read_excel('/dummy/path',
                                     parser_class=mock_parser_class)

        mock_parser_class.assert_called_once_with('/dummy/path')
        mock_add_requirement.assert_has_calls([
            mocker.call('Requirement 1', 0.33),
            mocker.call('Requirement 2', 0.5),
            mocker.call('Requirement 3', 0.17),
        ])
        mock_add_characteristic.assert_has_calls([
            mocker.call('Characteristic 1', (1, 5)),
            mocker.call('Characteristic 2', (10, 20)),
        ])
        mock_add_relationship.assert_has_calls([
            mocker.call('Requirement 1', 'Characteristic 1', 'min',
                        '---', 3),
            mocker.call('Requirement 2', 'Characteristic 1', 'opt',
                        'ooo', 13, 1),
            mocker.call('Requirement 3', 'Characteristic 2', 'max',
                        '+++', 3),
        ])

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
        assert isinstance(self.inst._merit(), np.ndarray)
        assert self.inst._merit().ndim == 2
        assert self.inst._merit().shape == self.inst.shape
        assert (self.inst._merit() == self.merit).all()


class TestCODACaseStudy1:
    """Case study of a bicycle wheel design based on ref 1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.wheel = wheel = models.CODA()
        self._setup_requirements(wheel)
        self._setup_characteristics(wheel)
        self._setup_relationships(wheel)

    def _setup_requirements(self, wheel):
        for name in ('Stiffness', 'Friction', 'Weight',
                     'Manufacturability', 'Repairability'):
            wheel.add_requirement(name, 0.2)

    def _setup_characteristics(self, wheel):
        wheel.add_characteristic('Tyre Diameter', (24, 29), 24)
        wheel.add_characteristic('Tyre Width', (11, 18), 13)
        wheel.add_characteristic('Spoke Thickness', (2.8, 5), 4.3)
        wheel.add_characteristic('Use of Composites', (0.05, 0.8), 0.2)

    def _setup_relationships(self, wheel):
        reqt = 'Stiffness'
        wheel.add_relationship(reqt, 0, 'min', 'strong', 29)
        wheel.add_relationship(reqt, 1, 'max', 'moderate', 12)
        wheel.add_relationship(reqt, 2, 'max', 'strong', 3)
        wheel.add_relationship(reqt, 3, 'opt', 'moderate', 0.5, 0.2)

        reqt = 'Friction'
        wheel.add_relationship(reqt, 'Tyre Diameter', 'max',
                               'moderate', 25)
        wheel.add_relationship(reqt, 'Tyre Width', 'max', 'strong',
                               11)

        reqt = 'Weight'
        wheel.add_relationship(reqt, 'Tyre Diameter', 'min', 'strong',
                               26)
        wheel.add_relationship(reqt, 'Tyre Width', 'min', 'strong',
                               15)
        wheel.add_relationship(reqt, 'Spoke Thickness', 'min',
                               'moderate', 3.5)
        wheel.add_relationship(reqt, 'Use of Composites', 'max',
                               'strong', 0.3)

        reqt = 'Manufacturability'
        wheel.add_relationship(reqt, 'Tyre Width', 'max', 'weak', 12)
        wheel.add_relationship(reqt, 'Spoke Thickness', 'max',
                               'moderate', 2.9)
        wheel.add_relationship(reqt, 'Use of Composites', 'min',
                               'strong', 0.5)

        reqt = 'Repairability'
        wheel.add_relationship(reqt, 'Tyre Width', 'max', 'weak', 14)
        wheel.add_relationship(reqt, 'Spoke Thickness', 'max',
                               'moderate', 3.8)
        wheel.add_relationship(reqt, 'Use of Composites', 'min',
                               'strong', 0.25)

    def test_merit(self):
        assert self.wheel.merit == pytest.approx(0.5788, abs=1e-4)

    def test_sum_of_correlations(self):
        """Sum of correlation factors for all requirements."""
        np.testing.assert_array_almost_equal(
            self.wheel.correlation.sum(axis=1, keepdims=True),
            np.array([[2.4, 1.2, 3.0, 1.3, 1.3]]).T
        )

    def test_read_excel(self):
        pytest.importorskip('pandas', reason="`pandas` required for spreadsheet parsing")
        pytest.importorskip('xlrd', reason="`xlrd` required for spreadsheet parsing")
        model = models.CODA.read_excel(
            os.path.join(DATA_DIR, 'demo_model_casestudy1.xlsx')
        )

        for char, ref in zip(model.characteristics,
                             self.wheel.characteristics):
            char.value = ref.value
        assert self.wheel.merit == model.merit


class TestCODACharacteristic:

    @pytest.fixture(autouse=True)
    def setup(self):
        class CODACharacteristic(models.CODACharacteristic):
            def __init__(self):
                pass
        self.inst = CODACharacteristic()

    def test___init____omit_value(self):
        """Omitting the value on instantiation is valid.

        When modelling a set of designs (typical) we don't necessarily
        want to seed the model with characteristic values.
        """
        inst = models.CODACharacteristic('Name')
        # This might want to be None? Requires everything supporting
        # that as an input though.
        with pytest.raises(AttributeError):
            _ = inst.value

    @pytest.mark.parametrize("value, exception", [
        (-0.01, ValueError),
        (0.0, None),
        (0.5, None),
        (1.0, None),
        (1.01, ValueError),
    ])
    def test_value__set_with_default_limits(self, value, exception):
        if exception is not None:
            with pytest.raises(exception):
                setattr(self.inst, 'value', value)
        else:
            self.inst.value = value
            assert self.inst.value == value

    def test_limits__get__default(self):
        assert self.inst.limits == self.inst._default_limits

    @pytest.mark.parametrize("value", [
        (0.0, 1.0), [0.0, 1.0], (None, None), (0, None), (None, 1)
    ])
    def test_limits__set__valid(self, value):
        self.inst.limits = value
        assert self.inst.limits == tuple(value)


class TestCODARequirement:

    @pytest.fixture(autouse=True)
    def setup(self):
        class CODARequirement(models.CODARequirement):
            def __init__(self):
                pass
        self.inst = CODARequirement()

    @pytest.mark.parametrize("wt, valid", [
        (-0.01, False), (0.0, True), (0.5, True), (1.0, True),
        (1.1, False)
    ])
    def test_weight__set(self, wt, valid):
        # Prototypes used context to allow weights to be provided in a
        # non-normalised form and this property would handle the
        # normalisation by inspecting the weights of other
        # requirements. This functionality isn't implemented here, but
        # might still be useful.
        if not valid:
            with pytest.raises(ValueError):
                setattr(self.inst, 'weight', wt)
        else:
            self.inst.weight = wt
            assert self.inst.weight == wt


class TestCODARelationship:

    @pytest.fixture(autouse=True)
    def setup(self):
        class Concrete(models.CODARelationship):
            def __init__(self):
                pass

            def __call__(self, x):
                return 0.0

        self.cls = Concrete
        self.inst = Concrete()

    @pytest.mark.parametrize("value, internal_value, valid", [
        [0.0, 0.0, True],
        [0.1, 0.1, True],
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
        ['---', 0.9, True],
        ['+++', 0.9, True],
        ['ooo', 0.9, True],
        ['--', 0.3, True],
        ['++', 0.3, True],
        ['oo', 0.3, True],
        ['o', 0.1, True],
        ['o', 0.1, True],
        ['o', 0.1, True],
    ])
    def test_correlation(self, value, internal_value, valid):
        """Correlation value must be one of a restricted set."""
        # TODO: It might be more flexible to enforce this further up
        #       for different scaling systems. Could also be done with
        #       a mixin implementation
        with pytest.raises(AttributeError):
            _ = self.inst.correlation
        if valid:
            self.inst.correlation = value
            assert self.inst.correlation == internal_value
        else:
            with pytest.raises(ValueError):
                setattr(self.inst, 'correlation', value)

    def test_target(self):
        """Target value may be anything, but check it's settable."""
        with pytest.raises(AttributeError):
            _ = self.inst.target
        self.inst.target = 0.0
        assert self.inst.target == 0.0


class TestCODANull:

    def test___init__(self):
        """Takes no arguments, has a correlation and merit of zero."""
        null = models.CODANull()
        assert null.correlation == 0.0
        assert null.target is None
        assert null(None) == 0.0

    def test__attributes_not_settable(self):
        null = models.CODANull()

        with pytest.raises(TypeError):
            null.correlation = 1
        with pytest.raises(TypeError):
            null.target = 1


class TestCODAMaximise:

    # TODO: compare function over range.

    def test_merit(self):
        inst = models.CODAMaximise(target=1.0, correlation=None)
        assert inst(1.0) == pytest.approx(0.5)
        assert inst(0.1) < 0.5
        assert inst(2.0) > 0.5


class TestCODAMinimise:

    # TODO: compare function over range.

    def test_merit(self):
        inst = models.CODAMinimise(target=1.0, correlation=None)
        assert inst(1.0) == pytest.approx(0.5)
        assert inst(0.1) > 0.5
        assert inst(2.0) < 0.5


class TestCODAOptimise:

    # TODO: compare function over range.

    def test_merit(self):
        inst = models.CODAOptimise(target=1.0, correlation=None,
                                   tolerance=0.2)
        assert inst(0.8) == pytest.approx(0.5)
        assert inst(1.2) == pytest.approx(0.5)
        assert inst(1.0) == pytest.approx(1.0)
        assert inst(1.1) > 0.5
        assert inst(0.9) > 0.5
        assert inst(2.0) < 0.5
        assert inst(0.0) < 0.5
