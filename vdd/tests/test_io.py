import unittest
import os

import numpy as np
try:
    import pandas as pd
    import xlrd
    deps_present = True
except ImportError:
    deps_present = False

import vdd


DATAD = os.path.join(os.path.dirname(__file__), 'data')


class TestExcelParser(unittest.TestCase):
    """Test case for importing a coda model definition from Excel."""

    @classmethod
    def setUpClass(cls):
        cls.path = path = os.path.join(DATAD, 'demo_model.xlsx')
        cls.parser = vdd.io.ExcelParser(path)

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

        self.assertItemsEqual(retval, [('Stiffness', 0.2),
                                       ('Friction', 0.3),
                                       ('Weight', 0.5)])

    def test_get_characteristics(self):
        retval = self.parser.get_characteristics()

        self.assertItemsEqual(retval[:2], [('Tyre Diameter', 24, 29),
                                           ('Tyre Width', 11, 18)])

        # Check the dummpy which contains NaNs.
        self.assertEqual(retval[2][0], 'Dummy Characteristic')
        self.assertTrue(pd.np.isnan(retval[2][1]))
        self.assertTrue(pd.np.isnan(retval[2][2]))

    def test_get_relationships(self):
        """Three relationships are defined in the source spreadsheet.
        """
        # NOTE: The Weight-Tyre Width relationship is artificial for
        #		testing optimial relationships.
        retval = self.parser.get_relationships()
        self.assertItemsEqual(
            retval,
            [('Stiffness', 'Tyre Diameter', 'min', 0.9, 29),
             ('Friction', 'Tyre Diameter', 'max', 0.3, 12),
             ('Weight', 'Tyre Width', 'opt', 0.1, 14, 1)]
        )


class TestCompactExcelParser(unittest.TestCase):
    """Functionally similar, but diff. source format to ExcelParser.
    """

    @classmethod
    def setUpClass(cls):
        cls.regular = vdd.io.ExcelParser(
            os.path.join(DATAD, 'demo_model.xlsx')
        )
        cls.compact = vdd.io.CompactExcelParser(
            os.path.join(DATAD, 'demo_model_compact.xlsx')
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


if __name__ == '__main__':
    unittest.main()
