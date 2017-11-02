import unittest

import numpy as np

import vdd


class TestWtBinaryMatrix(unittest.TestCase):

    def setUp(self):

        self.requirements = [
            'Light weight',
            'Impact resistance',
            'Good visibility',
            'Low noise',
            'Easy to put on/remove',
            'Comfortable'
        ]

        self.binary_matrix = np.matrix([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        self.wbm = vdd.wbm.WtBinaryMatrix()
        self.wbm._matrix = self.binary_matrix
        self.wbm.requirements = self.requirements

    def test__base_score(self):
        # Implementation detail, used to confirm intermediate calc.
        self.assertEqual(
            self.wbm.score,
            np.matrix([[1, 5, 2, 2, 2, 3]])
        )

    def test__biased_score(self):
        # Implementation detail, used to confirm intermediate calc.
        self.wbm._base_score = np.matrix([[1, 5, 2, 2, 2, 3]])
        self.assertEqual(
            self.wbm._biased_score,
            np.matrix([[2, 6, 3, 3, 3, 4]])
        )

    def test_score(self):
        self.wbm._biased_score = np.matrix([[2, 6, 3, 3, 3, 4]])
        np.testing.assert_allclose(
            self.wbm.score,
            np.matrix([[0.095, 0.286, 0.143,  0.143, 0.143, 0.19]])
        )


if __name__ == '__main__':
    unittest.main()
