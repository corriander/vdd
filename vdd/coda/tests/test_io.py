import unittest
import os
import sys
from unittest import mock

import numpy as np
try:
    import pandas as pd
    import xlrd
    deps_present = True
except ImportError:
    deps_present = False

from ... import common
from .. import io
from . import DATA_DIR


class TestExcelParser(unittest.TestCase):
    """Test case for importing a coda model definition from Excel."""

    @classmethod
    def setUpClass(cls):
        cls.path = path = os.path.join(DATA_DIR, 'demo_model.xlsx')
        cls.parser = io.ExcelParser(path)

    def __getattr__(self, attr):
        # Helper to support both Python 2 and 3
        if attr == 'assertCountEqual' and sys.version_info[0] == 2:
            return getattr(self, 'assertItemsEqual')

    def setUp(self):
        if not deps_present:
            self.skipTest("`pandas` package required for tests.")

    def test_cdf(self):
        """This should return a pandas dataframe.

        Only basic structure is checked here.
        """
        df = self.parser.cdf

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.shape, (3,)) # 2 chars
        self.assertEqual(len(df.columns), 3) # name, min, max

    def test_df(self):
        """This should return a pandas dataframe.

        Only basic structure is checked here.
        """
        df = self.parser.df

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.shape, (3,))	# 3 reqts

    def test_get_requirements(self):
        """Should return requisite information for requirements."""
        retval = self.parser.get_requirements()

        self.assertCountEqual(retval, [('Stiffness', 0.2),
                                       ('Friction', 0.3),
                                       ('Weight', 0.5)])

    def test_get_characteristics(self):
        retval = self.parser.get_characteristics()

        self.assertCountEqual(retval[:2], [('Tyre Diameter', 24, 29),
                                           ('Tyre Width', 11, 18)])

        # Check the dummpy which contains NaNs.
        self.assertEqual(retval[2][0], 'Dummy Characteristic')
        self.assertTrue(np.isnan(retval[2][1]))
        self.assertTrue(np.isnan(retval[2][2]))

    def test_get_relationships(self):
        """Three relationships are defined in the source spreadsheet.
        """
        # NOTE: The Weight-Tyre Width relationship is artificial for
        #		testing optimial relationships.
        retval = self.parser.get_relationships()
        self.assertCountEqual(
            retval,
            [('Stiffness', 'Tyre Diameter', 'min', 0.9, 29),
             ('Friction', 'Tyre Diameter', 'max', 0.3, 12),
             ('Weight', 'Tyre Width', 'opt', 0.1, 14, 1)]
        )


class TestCompactExcelParser(unittest.TestCase):
    """Functionally similar, but diff. source format to io.ExcelParser.
    """

    @classmethod
    def setUpClass(cls):
        cls.regular = io.ExcelParser(
            os.path.join(DATA_DIR, 'demo_model.xlsx')
        )
        cls.compact = io.CompactExcelParser(
            os.path.join(DATA_DIR, 'demo_model_compact.xlsx')
        )

    def setUp(self):
        if not deps_present:
            self.skipTest("`pandas` package required for tests.")

    def test_get_requirements(self):
        self.assertEqual(self.regular.get_requirements(),
                         self.compact.get_requirements())

    def test_get_characteristics(self):
        l1 = self.regular.get_characteristics()
        l2 = self.compact.get_characteristics()
        self.assertEqual(l1[:2], l2[:2])

        # Now check the last characteristic spec, as it contains NaNs.
        for i in 1, 2:
            self.assertTrue(
                np.isnan(l1[-1][i]) and np.isnan(l1[-1][i])
            )
        self.assertEqual(l1[-1][0], l2[-1][0])

    def test_get_relationships(self):
        self.maxDiff = None
        l1 = self.regular.get_relationships()
        l2 = self.compact.get_relationships()

        for t1, t2 in zip(l1, l2):
            self.assertEqual(t1[:3], t2[:3])
            self.assertAlmostEqual(t1[3],
                                   [0.1, 0.3, 0.9][len(t2[3])-1])
            self.assertAlmostEqual(t1[4], t2[4])


@mock.patch.object(io.GSheetCODA, 'df',
                   new_callable=mock.PropertyMock)
class TestGSheetCODA(unittest.TestCase):
    """Provides an interface to a compact form model in Google Sheets.

    The adapter behaves similarly to CompactExcelParser with scope for
    extension to support updating the remote data. With this in mind,
    these tests check that the subject matches or exceeds the
    functionality of the reference implementation. The specifics of
    the implementation will vary, however, as approach taken by
    CompactExcelParser is brittle and results in tight coupling
    (specifics of the format it returns, etc.).
    """

    @classmethod
    def setUpClass(cls):
        demo_model_path = os.path.join(DATA_DIR,
                                       'demo_model_compact.xlsx')
        cls.compact_excel_parser = io.CompactExcelParser(
            demo_model_path
        )

        df = pd.read_excel(demo_model_path)
        df = df.fillna('')
        df = df.astype(str)

        df.columns = [
            column if not column.startswith('Unnamed') else ''
            for column in df.columns
        ]
        cls.reference_df = df

    def setUp(self):
        self.mock_facade = mock.MagicMock(
            spec_set=common.io.GSheetsFacade
        )
        self.sut = io.GSheetCODA('dummy workbook name')
        self.sut._facade = self.mock_facade

    def test_get_characteristics(self, mock_df_property):
        """Expect a list of 3-tuples describing characteristics.

        The 3-tuples contain the name, and the min and max values of
        the characteristics defined in the source.

        There are three characteristics in the source, with the third
        having no min/max values (NaN).
        """
        mock_df_property.return_value = self.reference_df
        actual = self.sut.get_characteristics()
        expected = self.compact_excel_parser.get_characteristics()

        # We use numpy testing here because data contains NaNs.
        np.testing.assert_array_equal(np.array(actual),
                                      np.array(expected))

    def test_get_requirements(self, mock_df_property):
        """Expect a list of 2-tuples describing requirements.

        The 2-tuples contain each requirement and its
        weighting/scoring.
        """
        mock_df_property.return_value = self.reference_df
        actual = self.sut.get_requirements()
        expected = self.compact_excel_parser.get_requirements()
        self.assertEqual(actual, expected)

    def test_get_relationships(self, mock_df_property):
        """Expect a list of 5/6-tuples describing relationships.

        The tuples take one of two forms depending on the type of
        relationship:

         1. Min/Max: Correlation strength, relationship type, neutral
            point.
         2. Optimum: Correlation strength, relationship type, optimum
            point, tolerance.
        """
        self.maxDiff = None
        mock_df_property.return_value = self.reference_df
        actual = self.sut.get_relationships()
        expected = self.compact_excel_parser.get_relationships()

        np.testing.assert_array_equal(
            np.array(actual, dtype=object),
            np.array(expected, dtype=object)
        )

    def test_is_valid(self, mock_df_property):
        """Check the source sheet is valid.

        Note that only relationship_type fields are checks for symbol
        correctness currently.
        """
        mock_df_property.return_value = self.reference_df
        self.assertTrue(self.sut.is_valid())


if __name__ == '__main__':
    unittest.main()
