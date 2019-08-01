import unittest

from ddt import ddt, data, unpack

from . import DATA_DIR


class TestBinWMSource(unittest.TestCase):
    """Test case uses a concrete impl to test shared behaviour."""
    pass


@ddt
class TestGSheetsIntf(unittest.TestCase):

    @data(
        ('Motorcycle Helmet', True),
        ('Simple Aircraft', True),
        ('Minimal Example', True),
        ('Missing column', False),
        ('Lower triangle populated', False),
    )
    @unpack
    def test_is_valid(self):
        self.skipTest()

    def test_get_requirements(self):
        self.skipTest()

    def test_get_value_matrix(self):
        self.skipTest()


if __name__ == '__main__':
    unittest.main()
