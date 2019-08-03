import abc
import os

import pygsheets
import numpy as np
import pandas as pd
import xdg


# Python 2 & 3 compatible ABC superclass.
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class BinWMSource(ABC):

    @abc.abstractmethod
    def is_valid(self):
        """Validate the source data."""
        return True

    @abc.abstractmethod
    def get_requirements(self):
        """Get the list of requirements from the source data."""
        return []

    @abc.abstractmethod
    def get_value_matrix(self):
        """Get the matrix of decisions from the source data."""
        return np.matrix([])


class GSheetBinWM(BinWMSource):

    class InvalidSource(Exception): pass

    def __init__(self, workbook_name):
        self._workbook_name = workbook_name
        self._facade = GSheetsFacade()

    @property
    def df(self):
        try:
            return self._cached_df
        except AttributeError:
            rows = self.get_rows()
            header = rows[0]
            records = rows[1:]

            try:
                df = pd.DataFrame.from_records(records)
                df.columns = header
                df = df.set_index(df.columns[0])
            except:
                raise self.InvalidSource("Can't construct dataframe.")

            x, y = df.shape
            if x*y != x**2:
                raise self.InvalidSource("Source matrix not square.")

            mat = df.to_numpy()

            lower_tri = mat[np.tril_indices(n=x, k=0)]
            upper_tri = mat[np.triu_indices(n=x, k=1)]
            # check all elements are '', '0' or '1'
            # if all zero, break
            # check for 1s in the lower tri (bad)
            # Set lower tri to ''
            # If all blank; set to '0'; break
            # If not all blank, we have values in upper.
            # Check for blanks in upper tri (bad)
            # Only '0' and '1' left in upper tri (fine)
            # Replace '' with '0' in whole
            if not ((mat == '') | (mat == '0') | (mat == '1')).all():
                raise self.InvalidSource(
                    "Valid matrix values are empty, 0 or 1"
                )

            elif (mat == '0').all():
                df[:] = '0'

            elif (lower_tri == '1').any():
                raise self.InvalidSource(
                    "Lower triangular matrix should be 0 or empty"
                )

            else:
                mat[np.tril_indices(n=x, k=0)] = ''
                if (mat == '').all():
                    df[:] = '0'

                elif (upper_tri == '').any():
                    raise self.InvalidSource(
                        "Upper triangular matrix must be complete."
                    )
            df[df == ''] = '0'

            df = self._cached_df = df.astype(int)
            return df

    def is_valid(self):
        try:
            x, y = self.df.shape
        except self.InvalidSource:
            return False
        else:
            return True

    def get_requirements(self):
        return list(self.df.columns.values)

    def get_value_matrix(self):
        return np.matrix(self.df.to_numpy())

    def get_rows(self):
        return self._facade.get_rows(self._workbook_name)


class GSheetsFacade(object):

    _credentials_path = os.path.join(
        xdg.XDG_CONFIG_HOME,
        'vdd',
        'gsheets_credentials.json'
    )

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

    def get_rows(self, workbook_name):
        """Return a 2D list of populated rows/columns."""
        sheet = self._client.open(workbook_name).sheet1
        return PyGSheetsGSpreadAdapter(sheet).get_all_values()


class PyGSheetsGSpreadAdapter(object):
    # pygsheet's sheet api is a bit poor and does odd things. this
    # class provides a get_all_values method that behaves a little
    # more like gspread's default behaviour for the similar method.

    def __init__(self, pygsheets_sheet):
        self.sheet = pygsheets_sheet

    def get_all_values(self):
        sheet = self.sheet
        rows = sheet.get_all_values(include_tailing_empty_rows=False)
        # Strip all empty trailing columns
        rows = zip(*rows)
        rows = [cell for cell in rows if any(cell)]
        rows = [list(row) for row in zip(*rows)]
        return rows
