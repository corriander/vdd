import unittest
from unittest import mock

import pygsheets
import pandas as pd

from .. import io


@mock.patch.object(io.GSheetsFacade, '_sheet',
                   new_callable=mock.PropertyMock)
class TestGSheetsFacade(unittest.TestCase):

    def setup_mock_sheet(self, mock_property):
        mock_property.return_value = mock_sheet = mock.MagicMock(
            spec=io.WorksheetAdapter
        )
        # Ensure we can call the methods we actually use in the facade
        # because the adapter passes through attribute lookups to the
        # underlying implementation. mock doesn't know this so will
        # complain (rightfully). Using spec (rather than spec_set)
        # allows this and balances between strict and useful.
        mock_sheet.set_dataframe = mock.Mock()
        return mock_sheet

    def setUp(self):
        self.sut = io.GSheetsFacade('dummy workbook name')

    def test_get_rows(self, mock_sheet_property):
        """Utilises our adaptation of 'Worksheet.get_all_values'."""
        mock_sheet = self.setup_mock_sheet(mock_sheet_property)

        retval = self.sut.get_rows()

        mock_sheet.get_all_values.assert_called_once_with()
        self.assertIs(retval, mock_sheet.get_all_values.return_value)

    def test_write_dataframe(self, mock_sheet_property):
        """Utilises the pygsheets method 'Worksheet.set_dataframe'"""
        mock_sheet = self.setup_mock_sheet(mock_sheet_property)

        dummy_df = mock.MagicMock(spec_set=pd.DataFrame)

        retval = self.sut.write_dataframe(dummy_df, 'A1')

        mock_sheet.set_dataframe.assert_called_once_with(
                dummy_df,
                start='A1',
                copy_index=True,
                copy_head=True,
                fit=True
        )
        self.assertIs(retval, None)


class TestWorksheetAdapter(unittest.TestCase):
    """Adapter for external (pygsheets) worksheet model."""

    def setUp(self):
        self.mock_sheet = mock.MagicMock(
            spec_set=pygsheets.worksheet.Worksheet
        )
        self.sut = io.WorksheetAdapter(self.mock_sheet)

    def test_attribute_passthrough(self):
        """Attributes not explicitly overriden are passed through."""
        mock_sheet = self.mock_sheet

        adapter = self.sut
        adapter.refresh(update_grid=True)

        mock_sheet.refresh.assert_called_once_with(update_grid=True)

    def test_get_all_values__binwm_source_structure(self):
        """Method papers over some deficiencies in the wrapped method.

        Specifically, the original didn't provide enough control over
        trimming blank columns. This wrapper strips trailing columns.

        This test checks using source data typical of a binary
        weighting matrix
        """
        self.mock_sheet.get_all_values.return_value = [
            ['Requirements',
             'Requirement 1',
             'Requirement 2',
             'Requirement 3',
             '',
             '',
             '',
             ''],
            ['Requirement 1', '', '', '', '', '', '', ''],
            ['Requirement 2', '', '', '', '', '', '', ''],
            ['Requirement 3', '', '', '', '', '', '', '']
        ]

        adapter = self.sut
        actual = adapter.get_all_values()
        expected = [
            ['Requirements',
             'Requirement 1',
             'Requirement 2',
             'Requirement 3'],
            ['Requirement 1', '', '', ''],
            ['Requirement 2', '', '', ''],
            ['Requirement 3', '', '', '']
        ]

        self.assertEqual(actual, expected)

    def test_get_all_values__coda_source_structure(self):
        """Method papers over some deficiencies in the wrapped method.

        Specifically, the original didn't provide enough control over
        trimming blank columns. This wrapper strips trailing columns.

        This test checks using a source data structure typical of a
        coda model.
        """
        self.mock_sheet.get_all_values.return_value = [
                [  '', 'B1', 'C1',   '', 'E1',   '', ''],
                ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', ''],
                ['A3', 'B3', 'C3',   '', 'E3', 'F3', ''],
                ['A4', 'B4', 'C4', 'D4', 'E4',   '', ''],
        ]

        adapter = self.sut
        actual = adapter.get_all_values()
        expected = [
            [  '', 'B1', 'C1',   '', 'E1',   ''],
            ['A2', 'B2', 'C2', 'D2', 'E2', 'F2'],
            ['A3', 'B3', 'C3',   '', 'E3', 'F3'],
            ['A4', 'B4', 'C4', 'D4', 'E4',   ''],
        ]

        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
