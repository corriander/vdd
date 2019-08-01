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


class BinWMGSheet(BinWMSource):

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

            df = pd.DataFrame.from_records(records, index=0)
            df.columns = header
            df = df.set_index('Requirements')
            df = self._cached_df = df.astype(int)
            return df

    def is_valid(self):
        x, y = self.df.shape
        return x**2 == x*y
    
    def get_requirements(self):
        return self.df.columns

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
