import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pygsheets
from ddt import ddt, data, unpack


from ... import common
from .. import io
from . import DATA_DIR, FIXTURES_DIR


class TestBinWMSource(unittest.TestCase):
    """Test case uses a concrete impl to test shared behaviour."""
    pass


@ddt
class TestGSheetBinWM(unittest.TestCase):

    def get_mock_facade(self, fixture):
        mock_facade = mock.Mock(spec_set=common.io.GSheetsFacade)
        fixture_path = os.path.join(FIXTURES_DIR, fixture)
        with open(fixture_path) as fhandle:
            mock_facade.get_rows.return_value = json.load(fhandle)
        return mock_facade

    def get_subject_under_test(self, fixture):
        sut = io.GSheetBinWM('dummy_workbook_name')
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
        ('minimal_example_extra_row.json', False),
        ('minimal_example_duplicate_requirement.json', False),
        ('minimal_example_misaligned_requirements.json', False),
        ('minimal_example_score_column.json', True),
    )
    @unpack
    def test_is_valid(self, fixture, expected):
        """Method returns a boolean indicating source is valid.
        """
        sut = self.get_subject_under_test(fixture)
        self.assertIs(sut.is_valid(), expected)

    def test_get_label(self):
        """Check the custom label is picked up properly.

        This is critical functionality for easy copy pasting in
        jupyter notebooks; no label was OK but distortion happens very
        easily if you copy back into the source spreadsheet due to the
        missing cell in the dataframe visualisation.
        """
        sut = self.get_subject_under_test(
            'minimal_example_custom_label.json'
        )

        actual = sut.get_label()
        expected = 'Custom Label'

        self.assertEqual(actual, expected)

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
            io.GSheetBinWM.InvalidSource,
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
        expected = np.array([
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
            io.GSheetBinWM.InvalidSource,
            sut.get_value_matrix
        )

    def test_update(self):
        fixture = 'case__minimal_example.json'
        sut = self.get_subject_under_test(fixture)

        dummy_df = mock.MagicMock(spec_set=pd.DataFrame)
        sut.update(dummy_df)
        sut._facade.write_dataframe.assert_called_once_with(
            dummy_df, position='A1'
        )

    def test__score_column_is_ignored(self):
        """Score columns in the source spreadsheet are dropped.

        This is so they can be re-created in a controlled fashion.
        """
        sut = self.get_subject_under_test(
            'minimal_example_score_column.json'
        )

        actual = sut.df.columns.values
        expected = np.array([
            'Requirement 1',
            'Requirement 2',
            'Requirement 3'
            # No score column
        ])
        np.testing.assert_array_equal(actual, expected)

        # And just to check the pathological case where our test data
        # has got out of whack and we've missed this:
        assert 'Score' in sut._facade.get_rows()[0]


if __name__ == '__main__':
    unittest.main()
