"""
References
----------

 1. M.H. Eres et al, 2014. Mapping Customer Needs to Engineering
    Characteristics: An Aerospace Perspective for Conceptual Design -
    Journal of Engineering Design pp. 1-24
"""
import unittest

import mock
import numpy as np
from ddt import ddt, unpack, data

import vdd

@ddt
class TestBinWM(unittest.TestCase):

    models = {
        'Motorcycle Helmet': {
            'requirements': [
                'Light weight',
                'Impact resistance',
                'Good visibility',
                'Low noise',
                'Easy to put on/remove',
                'Comfortable'
                'Light'
            ],
            'binary_matrix': np.matrix([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        },
        'Simple Aircraft': {
            'requirements': [
                'Easy to extend operational life',
                'Green aircraft',
                'Easy to operate',
                'Cheap to maintain and repair',
                'Lowest consumption per PAX-km',
                'No A/C on ground',
                'Unlimited use of Internet',
                'Extremely comfortable',
                'Sufficient range'
            ],
            'binary_matrix': np.matrix([
                [0, 0, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        },
        'Minimal Example': {
            'requirements': [
                'Requirement 1',
                'Requirement 2',
                'Requirement 3'
            ],
            'binary_matrix': np.matrix([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        }
    }

    def setup_binary_weighting_matrix(self, key):
        data = self.models[key]
        bwm = vdd.requirements.BinWM(*data['requirements'])
        bwm._matrix = data['binary_matrix']
        return bwm

    def test_score__motorcycle_helmet(self):
        bwm = self.setup_binary_weighting_matrix('Motorcycle Helmet')

        np.testing.assert_allclose(
            bwm.score,
            np.array([0.095, 0.286, 0.143,  0.143, 0.143, 0.19]),
            atol=0.01
        )

    def test_score__simple_aircraft(self):
        bwm = self.setup_binary_weighting_matrix('Simple Aircraft')

        np.testing.assert_allclose(
            bwm.score,
            np.array([0.13, 0.16, 0.13, 0.04, 0.13, 0.09, 0.07, 0.09, 0.16]),
            atol=0.1
        )

    @data(
        [('n', 'n', 'n'), (0.17, 0.33, 0.5)],
        [('y', 'n', 'n'), (0.33, 0.17, 0.5)],
        [('n', 'y', 'n'), (0.33, 0.33, 0.33)],
        [('n', 'y', 'y'), (0.33, 0.5, 0.17)],
        [('y', 'y', 'y'), (0.5, 0.33, 0.17)]
    )
    @unpack
    @mock.patch('vdd.requirements.BinWM._print')
    @mock.patch('vdd.requirements.BinWM._input')
    def test_prompt(self, answers, score, mock_input, mock_print):
        mock_input.side_effect = answers
        bwm = self.setup_binary_weighting_matrix('Minimal Example')

        bwm.prompt(shuffle=False)

        mock_input.assert_has_calls([
            mock.call('Requirement 1 is more important than Requirement 2: '),
            mock.call('Requirement 1 is more important than Requirement 3: '),
            mock.call('Requirement 2 is more important than Requirement 3: ')
        ])

        np.testing.assert_allclose(bwm.score, np.array(score), atol=0.01)

    @mock.patch('random.shuffle')
    @mock.patch('vdd.requirements.BinWM._print')
    @mock.patch('vdd.requirements.BinWM._input')
    def test_prompt__shuffle(self, mock_input, mock_print, mock_shuffle):
        mock_input.side_effect = ['y'] * 3
        bwm = self.setup_binary_weighting_matrix('Minimal Example')

        bwm.prompt(shuffle=True)

        mock_shuffle.assert_called_with([
            (0, 1, 'Requirement 1', 'Requirement 2'),
            (0, 2, 'Requirement 1', 'Requirement 3'),
            (1, 2, 'Requirement 2', 'Requirement 3')
        ])


if __name__ == '__main__':
    unittest.main()