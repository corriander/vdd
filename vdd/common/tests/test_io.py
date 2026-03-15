import pytest
import pygsheets
import pandas as pd

from .. import io


class TestGSheetsFacade:

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.sut = io.GSheetsFacade('dummy workbook name')
        self.mock_sheet_property = mocker.patch.object(
            io.GSheetsFacade, '_sheet', new_callable=mocker.PropertyMock
        )
        mock_sheet = mocker.MagicMock(spec=io.WorksheetAdapter)
        # Ensure we can call the methods we actually use in the facade
        # because the adapter passes through attribute lookups to the
        # underlying implementation. mock doesn't know this so will
        # complain (rightfully). Using spec (rather than spec_set)
        # allows this and balances between strict and useful.
        mock_sheet.set_dataframe = mocker.Mock()
        self.mock_sheet_property.return_value = mock_sheet
        self.mock_sheet = mock_sheet

    def test_get_rows(self):
        """Utilises our adaptation of 'Worksheet.get_all_values'."""
        retval = self.sut.get_rows()

        self.mock_sheet.get_all_values.assert_called_once_with()
        assert retval is self.mock_sheet.get_all_values.return_value

    def test_write_dataframe(self, mocker):
        """Utilises the pygsheets method 'Worksheet.set_dataframe'"""
        dummy_df = mocker.MagicMock(spec_set=pd.DataFrame)

        retval = self.sut.write_dataframe(dummy_df, 'A1')

        self.mock_sheet.set_dataframe.assert_called_once_with(
            dummy_df,
            start='A1',
            copy_index=True,
            copy_head=True,
            fit=True
        )
        assert retval is None


class TestWorksheetAdapter:
    """Adapter for external (pygsheets) worksheet model."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.mock_sheet = mocker.MagicMock(
            spec_set=pygsheets.worksheet.Worksheet
        )
        self.sut = io.WorksheetAdapter(self.mock_sheet)

    def test_attribute_passthrough(self):
        """Attributes not explicitly overriden are passed through."""
        self.sut.refresh(update_grid=True)

        self.mock_sheet.refresh.assert_called_once_with(update_grid=True)

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

        actual = self.sut.get_all_values()
        expected = [
            ['Requirements',
             'Requirement 1',
             'Requirement 2',
             'Requirement 3'],
            ['Requirement 1', '', '', ''],
            ['Requirement 2', '', '', ''],
            ['Requirement 3', '', '', '']
        ]

        assert actual == expected

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

        actual = self.sut.get_all_values()
        expected = [
            [  '', 'B1', 'C1',   '', 'E1',   ''],
            ['A2', 'B2', 'C2', 'D2', 'E2', 'F2'],
            ['A3', 'B3', 'C3',   '', 'E3', 'F3'],
            ['A4', 'B4', 'C4', 'D4', 'E4',   ''],
        ]

        assert actual == expected
