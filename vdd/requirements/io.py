import abc
import os

import gspread
import numpy as np
import pandas as pd
import xdg
from  oauth2client.service_account import ServiceAccountCredentials


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

    @property
    def _client(self):
        try:
            return self._cached_client
        except AttributeError:
            creds = GSheetsAuth().get_credentials()
            self._cached_client = gspread.authorize(creds)
            return self._cached_client

    def get_rows(self, workbook_name):
        """Return the list of populated rows."""
        sheet = self._client.open(workbook_name).sheet1
        return sheet.get_all_values()


class GSheetsAuth(object):

    AUTH_SCOPES = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]

    @property
    def _config_dir(self):
        return os.path.join(xdg.XDG_CONFIG_HOME, 'vdd')

    def get_client_secret_path(self):
        return os.path.join(self._config_dir, 'gsheets_credentials.json')

    def get_credentials(self):
        return ServiceAccountCredentials.from_json_keyfile_name(
            self.get_client_secret_path(),
            self.AUTH_SCOPES
        )
