import unittest
import os

import pandas as pd

import vdd


DATAD = os.path.join(os.path.dirname(__file__), 'data')


class TestExcelParser(unittest.TestCase):
    """Test case for importing a coda model definition from Excel."""

    @classmethod
    def setUpClass(cls):
        cls.path = path = os.path.join(DATAD, 'demo_model.xlsx')
        cls.parser = vdd.io.ExcelParser(path)

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



if __name__ == '__main__':
    unittest.main()
