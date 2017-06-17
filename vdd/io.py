import collections

import pandas as pd


class ExcelParser(object):

    # 20 characteristic definitions are supported ((4*26)/5 cols)
    _MAX_COL = 'ZZZZ'
    _NCOLS_CHAR = 4

    def __init__(self, path):
        self.path = path

    @property
    def df(self):
        """DataFrame of requirement-characteristic relationships."""
        try:
            return self._df
        except AttributeError:
            df = self._df = pd.read_excel(self.path, skiprows=[0,1])
            return df

    @property
    def cdf(self):
        """DataFrame of characteristic definitions.

        This augments the regular DataFrame.
        """
        try:
            return self._cdf
        except AttributeError:
            df = pd.read_excel(
                self.path,
                parse_cols="C:{}".format(self._MAX_COL)
            )[:1]

            names = filter(lambda s: 'Unnamed' not in s, df.columns)

            dd = collections.defaultdict(list)
            for i, s in enumerate(df.columns):

                ridx = i % self._NCOLS_CHAR # Relative index

                if ridx == 0:
                    # Initial column of group; begin construct.
                    dd['name'].append(s)

                elif ridx == 1:
                    dd['min'].append(df.loc[0, s])

                elif ridx == 3:
                    # Final column of group; add construct to list.
                    dd['max'].append(df.loc[0, s])

            self._cdf = tdf = pd.DataFrame.from_dict(dd)
            return tdf

    def get_requirements(self):
        cols = ('Weighting', 'Requirements')
        return [tuple(reversed(tuple(rec)[1:]))	# Exclude idx
                for rec in self.df.loc[:,cols].to_records()]

    def get_characteristics(self):
        """Returns a 3-tuple: (<name>, <minvalue>, <maxvalue>).

        If the minimum or maximum values are omitted, NaN(s) will be
        returned.
        """
        return [(rec['name'], rec['min'], rec['max'])
                for rec in self.cdf.to_records()]

