from __future__ import absolute_import

import abc
import os

import platformdirs
import pygsheets

from .abstract import ABC


class AbstractGSheet(ABC):

    class InvalidSource(Exception): pass

    def __init__(self, workbook_name):
        self._workbook_name = workbook_name
        self._facade = GSheetsFacade(workbook_name)

    @abc.abstractmethod
    def is_valid(self):
        """Method reports whether the source worksheet is valid."""
        return False

    @abc.abstractmethod
    def update(self, df):
        """Method updates the source worksheet with a dataframe."""
        return None


class GSheetsFacade(object):
    """Facade providing restricted API to Google Sheets."""

    _credentials_path = os.path.join(
        platformdirs.user_config_dir('vdd'),
        'gsheets_credentials.json'
    )

    def __init__(self, workbook_name):
        # We shouldn't really init the facade with a workbook name.
        # It's here to support the retention of the sheet, but perhaps
        # we just store fetched sheets and make them available for
        # referencing, specifying the workbook name each time we want
        # to operate on it.
        self._workbook_name = workbook_name

    @property
    def _client(self):
        # google sheets client (cached)
        try:
            return self._cached_client
        except AttributeError:
            self._cached_client = pygsheets.authorize(
                service_account_file=self._credentials_path
            )
            return self._cached_client

    @property
    def _sheet(self):
        try:
            return self._cached_sheet
        except AttributeError:
            sheet = self._client.open(self._workbook_name).sheet1
            self._cached_sheet = WorksheetAdapter(sheet)
            return self._cached_sheet


    def get_rows(self):
        """Return a 2D list of populated rows/columns."""
        return self._sheet.get_all_values()

    def write_dataframe(self, df, position):
        """Write a dataframe to the worksheet at position.

        Parameters
        ----------

        position : str
            Upper left cell for the dataframe position.
        """
        self._sheet.set_dataframe(df, start=position, copy_index=True,
                                  copy_head=True, fit=True)


class WorksheetAdapter(object):
    # pygsheet's sheet api is a bit poor and does odd things. this
    # class brings it a little closer to the gspread behaviour in some
    # respects.

    def __init__(self, pygsheets_sheet):
        self._sheet = pygsheets_sheet

    def __getattr__(self, attr):
        return getattr(self._sheet, attr)

    def get_all_values(self):
        # TODO: Consider tackling this method's API upstream
        sheet = self._sheet
        rows = sheet.get_all_values(include_tailing_empty_rows=False)
        # Strip all empty trailing columns
        rows = zip(*rows)
        rows = [cell for cell in rows if any(cell)]
        rows = [list(row) for row in zip(*rows)]
        return rows
