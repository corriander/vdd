import pandas as pd


class ExcelParser(object):

    def __init__(self, path):
        self.path = path

    @property
    def df(self):
        try:
            return self._df
        except AttributeError:
            df = self._df = pd.read_excel(self.path)
            return df

    def get_requirements(self):
        return
