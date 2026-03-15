import os

import numpy as np
import pytest
try:
    import pandas as pd
    import xlrd
    deps_present = True
except ImportError:
    deps_present = False

from ... import common
from .. import io
from . import DATA_DIR


@pytest.mark.skipif(not deps_present,
                    reason="`pandas` package required for tests.")
class TestExcelParser:
    """Test case for importing a coda model definition from Excel."""

    @pytest.fixture(autouse=True, scope='class')
    def setup_class(self):
        path = os.path.join(DATA_DIR, 'demo_model.xlsx')
        self.__class__.parser = io.ExcelParser(path)
        self.__class__.path = path

    def test_cdf(self):
        """This should return a pandas dataframe.

        Only basic structure is checked here.
        """
        df = self.parser.cdf

        assert isinstance(df, pd.DataFrame)
        assert df.index.shape == (3,)  # 2 chars
        assert len(df.columns) == 3   # name, min, max

    def test_df(self):
        """This should return a pandas dataframe.

        Only basic structure is checked here.
        """
        df = self.parser.df

        assert isinstance(df, pd.DataFrame)
        assert df.index.shape == (3,)  # 3 reqts

    def test_get_requirements(self):
        """Should return requisite information for requirements."""
        retval = self.parser.get_requirements()

        assert sorted(retval) == sorted([
            ('Stiffness', 0.2),
            ('Friction', 0.3),
            ('Weight', 0.5),
        ])

    def test_get_characteristics(self):
        retval = self.parser.get_characteristics()

        assert sorted(retval[:2]) == sorted([
            ('Tyre Diameter', 24, 29),
            ('Tyre Width', 11, 18),
        ])

        # Check the dummy which contains NaNs.
        assert retval[2][0] == 'Dummy Characteristic'
        assert np.isnan(retval[2][1])
        assert np.isnan(retval[2][2])

    def test_get_relationships(self):
        """Three relationships are defined in the source spreadsheet."""
        # NOTE: The Weight-Tyre Width relationship is artificial for
        #       testing optimal relationships.
        retval = self.parser.get_relationships()
        assert sorted(retval) == sorted([
            ('Stiffness', 'Tyre Diameter', 'min', 0.9, 29),
            ('Friction', 'Tyre Diameter', 'max', 0.3, 12),
            ('Weight', 'Tyre Width', 'opt', 0.1, 14, 1),
        ])


@pytest.mark.skipif(not deps_present,
                    reason="`pandas` package required for tests.")
class TestCompactExcelParser:
    """Functionally similar, but diff. source format to io.ExcelParser."""

    @pytest.fixture(autouse=True, scope='class')
    def setup_class(self):
        self.__class__.regular = io.ExcelParser(
            os.path.join(DATA_DIR, 'demo_model.xlsx')
        )
        self.__class__.compact = io.CompactExcelParser(
            os.path.join(DATA_DIR, 'demo_model_compact.xlsx')
        )

    def test_get_requirements(self):
        assert self.regular.get_requirements() == self.compact.get_requirements()

    def test_get_characteristics(self):
        l1 = self.regular.get_characteristics()
        l2 = self.compact.get_characteristics()
        assert l1[:2] == l2[:2]

        # Now check the last characteristic spec, as it contains NaNs.
        for i in 1, 2:
            assert np.isnan(l1[-1][i]) and np.isnan(l1[-1][i])
        assert l1[-1][0] == l2[-1][0]

    def test_get_relationships(self):
        l1 = self.regular.get_relationships()
        l2 = self.compact.get_relationships()

        for t1, t2 in zip(l1, l2):
            assert t1[:3] == t2[:3]
            assert t1[3] == pytest.approx([0.1, 0.3, 0.9][len(t2[3])-1])
            assert t1[4] == pytest.approx(t2[4])


@pytest.mark.skipif(not deps_present,
                    reason="`pandas` package required for tests.")
class TestGSheetCODA:
    """Provides an interface to a compact form model in Google Sheets.

    The adapter behaves similarly to CompactExcelParser with scope for
    extension to support updating the remote data. With this in mind,
    these tests check that the subject matches or exceeds the
    functionality of the reference implementation. The specifics of
    the implementation will vary, however, as approach taken by
    CompactExcelParser is brittle and results in tight coupling
    (specifics of the format it returns, etc.).
    """

    @pytest.fixture(autouse=True, scope='class')
    def setup_class(self):
        demo_model_path = os.path.join(DATA_DIR, 'demo_model_compact.xlsx')
        self.__class__.compact_excel_parser = io.CompactExcelParser(
            demo_model_path
        )

        df = pd.read_excel(demo_model_path)
        df = df.fillna('')
        df = df.astype(str)

        df.columns = [
            column if not column.startswith('Unnamed') else ''
            for column in df.columns
        ]
        self.__class__.reference_df = df

    @pytest.fixture(autouse=True)
    def setup_sut(self, mocker):
        """Create the subject under test with a mock facade and a patched df property."""
        self.mock_facade = mocker.MagicMock(spec_set=common.io.GSheetsFacade)
        self.sut = io.GSheetCODA('dummy workbook name')
        self.sut._facade = self.mock_facade
        self.mock_df_property = mocker.patch.object(
            io.GSheetCODA, 'df', new_callable=mocker.PropertyMock
        )

    def test_get_characteristics(self):
        """Expect a list of 3-tuples describing characteristics.

        The 3-tuples contain the name, and the min and max values of
        the characteristics defined in the source.

        There are three characteristics in the source, with the third
        having no min/max values (NaN).
        """
        self.mock_df_property.return_value = self.reference_df
        actual = self.sut.get_characteristics()
        expected = self.compact_excel_parser.get_characteristics()

        # We use numpy testing here because data contains NaNs.
        np.testing.assert_array_equal(np.array(actual),
                                      np.array(expected))

    def test_get_requirements(self):
        """Expect a list of 2-tuples describing requirements.

        The 2-tuples contain each requirement and its
        weighting/scoring.
        """
        self.mock_df_property.return_value = self.reference_df
        actual = self.sut.get_requirements()
        expected = self.compact_excel_parser.get_requirements()
        assert actual == expected

    def test_get_relationships(self):
        """Expect a list of 5/6-tuples describing relationships.

        The tuples take one of two forms depending on the type of
        relationship:

         1. Min/Max: Correlation strength, relationship type, neutral
            point.
         2. Optimum: Correlation strength, relationship type, optimum
            point, tolerance.
        """
        self.mock_df_property.return_value = self.reference_df
        actual = self.sut.get_relationships()
        expected = self.compact_excel_parser.get_relationships()

        np.testing.assert_array_equal(
            np.array(actual, dtype=object),
            np.array(expected, dtype=object)
        )

    def test_is_valid(self):
        """Check the source sheet is valid.

        Note that only relationship_type fields are checks for symbol
        correctness currently.
        """
        self.mock_df_property.return_value = self.reference_df
        assert self.sut.is_valid()
