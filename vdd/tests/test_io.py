import unittest
import os

import pandas as pd

import vdd


DATAD = os.path.join(os.path.dirname(__file__), 'data')


class TestExcelParser(unittest.TestCase):
    """Test case for importing a coda model definition from Excel."""

    def setUp(self):
        self.path = path = os.path.join(DATAD, 'demo_model.xlsx')
        self.parser = vdd.io.ExcelParser(path)

    def test_df(self):
        """This should return a pandas dataframe.

        Only basic structure is checked here.
        """
        df = self.parser.df

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.shape, (3,))	# 3 reqts

    def test_get_requirements(self):
        """This should return all requirements defined in the source.
        """
        requirements = self.parser.get_requirements()

        self.assertItemsEqual(requirements, [('Stiffness', 0.2),
                                             ('Friction', 0.3),
                                             ('Weight', 0.5)])


if __name__ == '__main__':
    unittest.main()
