import json
import os
import unittest

import numpy as np
import mock
from ddt import ddt, data, unpack

from .. import io

from . import DATA_DIR, FIXTURES_DIR


class TestBinWMSource(unittest.TestCase):
    """Test case uses a concrete impl to test shared behaviour."""
    pass


@ddt
class TestBinWMGSheet(unittest.TestCase):

    def get_mock_facade(self, fixture):
        mock_facade = mock.Mock(spec_set=io.GSheetsFacade)
        fixture_path = os.path.join(FIXTURES_DIR, fixture)
        with open(fixture_path) as fhandle:
            mock_facade.get_rows.return_value = json.load(fhandle)
        return mock_facade

    def get_subject_under_test(self, fixture):
        sut = io.BinWMGSheet('dummy_workbook_name')
        sut._facade = self.get_mock_facade(fixture)
        return sut

    @data(
        ('minimal_example_incomplete_triu.json', False),
        ('minimal_example_incomplete_tril.json', True),
        ('minimal_example_populated.json', True),
        ('minimal_example_zeros.json', True),
        ('minimal_example_blank.json', True),
        ('minimal_example_different_cell_a1.json', True),
        ('minimal_example_extra_column.json', False),
        ('minimal_example_extra_row.json', False)
    )
    @unpack
    def test_is_valid(self, fixture, expected):
        """Method returns a boolean indicating source is valid.
        """
        sut = self.get_subject_under_test(fixture)
        self.assertIs(sut.is_valid(), expected)

    @data(
        'minimal_example_incomplete_tril.json',
        'minimal_example_populated.json',
        'minimal_example_zeros.json',
        'minimal_example_blank.json',
        'minimal_example_different_cell_a1.json',
    )
    def test_get_requirements__valid(self, fixture):
        """A list of requirements is obtainable from valid sources."""
        expected = ['Requirement {}'.format(x) for x in range(1, 4)]
        sut = self.get_subject_under_test(fixture)
        self.assertEqual(sut.get_requirements(), expected)

    @data(
        'minimal_example_incomplete_triu.json',
        'minimal_example_extra_column.json',
        'minimal_example_extra_row.json',
        'minimal_example_tril_ones.json'
    )
    def test_get_requirements__invalid(self, fixture):
        """The requirements list can't be read; raise an exception."""
        sut = self.get_subject_under_test(fixture)
        self.assertRaises(
            io.BinWMGSheet.InvalidSource,
            sut.get_requirements
        )

    @data(
        ('minimal_example_incomplete_tril.json', (0, 1, 1)),
        ('minimal_example_populated.json', (0, 1, 1)),
        ('minimal_example_zeros.json', (0, 0, 0)),
        ('minimal_example_blank.json', (0, 0 ,0)),
        ('minimal_example_different_cell_a1.json', (0, 1, 1))
    )
    @unpack
    def test_get_value_matrix__valid(self, fixture, expected_triu):
        """The binary matrix is obtained from valid sources."""
        a, b, c = expected_triu
        expected = np.matrix([
            [0, a, b],
            [0, 0, c],
            [0, 0, 0]
        ])

        sut = self.get_subject_under_test(fixture)
        np.testing.assert_array_almost_equal(sut.get_value_matrix(),
                                             expected)

    @data(
        'minimal_example_incomplete_triu.json',
        'minimal_example_extra_column.json',
        'minimal_example_extra_row.json',
        'minimal_example_tril_ones.json'
    )
    def test_get_value_matrix__invalid(self, fixture):
        """The binary matrix can't be read; raise an exception."""
        sut = self.get_subject_under_test(fixture)
        self.assertRaises(
            io.BinWMGSheet.InvalidSource,
            sut.get_value_matrix
        )


if __name__ == '__main__':
    unittest.main()
