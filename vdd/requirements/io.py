import abc
import os

import numpy as np
import pandas as pd

from .. import common


class BinWMSheet(common.ABC):

    score_column_name = 'Score'

    @abc.abstractmethod
    def get_requirements(self):
        """Get the list of requirements from the source data."""
        return []

    @abc.abstractmethod
    def get_value_matrix(self):
        """Get the matrix of decisions from the source data."""
        return np.array([[]])


class GSheetBinWM(BinWMSheet, common.io.AbstractGSheet):
    # TODO: This level of abstraction may not be necessary.
    #       pygsheets supports getting as a dataframe; inconsistency
    #       of method names aside in 2.0.2 (get_as_df, set_dataframe),
    #       the index_col parameter doesn't seem to work as expected.
    #       If this can be fixed in the adapter (or upstream) we might
    #       just be able to grab the dataframe directly.

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
                df.index.name = header[0]
            except:
                raise self.InvalidSource("Can't construct dataframe.")

            self._validate_df_shape(df)

            # Check integrity of axes.
            for r, c in zip(df.index.values, df.columns.values):
                if r != c:
                    raise self.InvalidSource(
                        "Row and column labels misaligned."
                    )
            if len(set(df.index.values)) != len(df.index.values):
                raise self.InvalidSource("Duplicate row labels.")
            if len(set(df.columns.values)) != len(df.columns.values):
                raise self.InvalidSource("Duplicate column labels.")

            mat = df.to_numpy()

            lower_tri = mat[np.tril_indices(n=len(df.index), k=0)]
            upper_tri = mat[np.triu_indices(n=len(df.index), k=1)]
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
                mat[np.tril_indices(n=len(df.index), k=0)] = ''
                if (mat == '').all():
                    df[:] = '0'

                elif (upper_tri == '').any():
                    raise self.InvalidSource(
                        "Upper triangular matrix must be complete."
                    )
            df[df == ''] = '0'

            df = self._cached_df = df.astype(int)
            return df

    def _validate_df_shape(self, df):
        # Checks the dataframe shape is valid
        #
        # Side-effects
        # ------------
        #
        # Checks for a final 'Score' column and drops it if it exists
        #
        # Raises
        # ------
        #
        # self.InvalidSource
        #   If not valid shape
        x, y = df.shape
        if x*y != x**2:
            if df.columns[-1] == self.score_column_name:
                df.drop(columns=self.score_column_name, inplace=True)
                self._validate_df_shape(df)
            else:
                raise self.InvalidSource("Source matrix not square.")

    def is_valid(self):
        try:
            x, y = self.df.shape
        except self.InvalidSource:
            return False
        else:
            return True

    def get_label(self):
        """Get label for the axes (upper left cell)."""
        return self.df.index.name

    def get_requirements(self):
        """Read the requirements from the dataframe representation."""
        return list(self.df.columns.values)

    def get_value_matrix(self):
        """Read the value matrix from the dataframe representation."""
        return self.df.to_numpy()

    def get_rows(self):
        """Return the rows in the source spreadsheet."""
        return self._facade.get_rows()

    def update(self, df):
        """Bulk update the contents of the source spreadsheet."""
        self._facade.write_dataframe(df, position='A1')
