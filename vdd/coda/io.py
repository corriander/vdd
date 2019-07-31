import re
import collections
import itertools
import warnings

import numpy as np

try:
    import pandas as pd
    import xlrd
except ImportError:
    warnings.warn('`pandas` and `xlrd` packages required for '
                  'spreadsheet support.')


class ExcelParser(object):

    # 20 characteristic definitions are supported ((4*26)/5 cols)
    _MAX_COL = 'CZ'
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
                usecols="C:{}".format(self._MAX_COL)
            )[:1]

        return self._cdf_base(df)

    def _cdf_base(self, df):
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

    def get_characteristics(self):
        """Returns a 3-tuple: (<name>, <minvalue>, <maxvalue>).

        If the minimum or maximum values are omitted, NaN(s) will be
        returned.
        """
        l = []
        for rec in self.cdf.to_records():
            if re.match(r'^(Unnamed: \d+|Characteristic \d+)$',
                        rec['name']) is not None:
                warnings.warn("Picked up a default column name")
            else:
                l.append((rec['name'], rec['min'], rec['max']))
        return l

    def get_relationships(self):
        """Get relationships defined a 4/5-tuple.

        Size of tuple depends on the type of relationship.
        """
        # TODO: Yeah I know, variable return type.
        reqts = [tup[0] for tup in self.get_requirements()]
        chars = [tup[0] for tup in self.get_characteristics()]

        return self._parse_row(reqts, chars)

    def _parse_row(self, reqts, chars):

        n = self._NCOLS_CHAR
        df = self.df.loc[:,'Correlation':]

        relationships = []
        for (i, r), (j, c) in itertools.product(enumerate(reqts),
                                                enumerate(chars)):
            row = df.loc[i,:].values
            base_tup = (r, c, row[j*n+1], row[j*n+0], row[j*n+2])

            if np.isnan(base_tup[4]):
                # The target value is always a quantity.
                continue

            if base_tup[2] == 'opt':
                tup = base_tup + (row[j*n+3],)
            else:
                tup = base_tup

            relationships.append(tup)

        return relationships

    def get_requirements(self):
        cols = ('Weighting', 'Requirements')
        return [tuple(reversed(tuple(rec)[1:]))	# Exclude idx
                for rec in self.df.loc[:,cols].to_records()]


class CompactExcelParser(ExcelParser):

    _NCOLS_CHAR = 3

    def _cdf_base(self, df):
            dd = collections.defaultdict(list)
            for i, s in enumerate(df.columns):

                ridx = i % self._NCOLS_CHAR # Relative index

                if ridx == 0:
                    # Initial column of group; begin construct.
                    dd['name'].append(s)

                elif ridx == 1:
                    dd['min'].append(df.loc[0, s])

                elif ridx == 2:
                    # Final column of group; add construct to list.
                    dd['max'].append(df.loc[0, s])

            self._cdf = tdf = pd.DataFrame.from_dict(dd)
            return tdf

    def _parse_row(self, reqts, chars):
        n = self._NCOLS_CHAR
        df = self.df.loc[:,'Relationship Type':]

        relationships = []
        for (i, r), (j, c) in itertools.product(enumerate(reqts),
                                                enumerate(chars)):
            row = df.loc[i,:].values
            rel = row[j*n]

            try:
                type_ = {'+': 'max', 'o': 'opt', '-': 'min'}[rel[0]]
            except (TypeError, IndexError):
                # rel is not a recognised string.
                continue

            base_tup = (r, c, type_, rel, row[j*n+1])

            if np.isnan(base_tup[4]):
                # The target value is always a quantity.
                continue

            if base_tup[2] == 'opt':
                tup = base_tup + (row[j*n+2],)
            else:
                tup = base_tup

            relationships.append(tup)

        return relationships

