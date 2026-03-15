import json
import os

import numpy as np
import pandas as pd
import pytest

from .. import io
from .. import models

from . import FIXTURES_DIR


def get_fixture_data(fname):
    """Load a JSON fixture file and return its parsed contents."""
    path = os.path.join(FIXTURES_DIR, fname)
    with open(path) as f:
        return json.load(f)


class TestBinWM:

    model_data_fixtures = {
        'Minimal Example': 'case__minimal_example.json',
        'Motorcycle Helmet': 'case__motorcycle_helmet.json',
        'Simple Aircraft': 'case__simple_aircraft.json'
    }

    def setup_binary_weighting_matrix(self, key):
        fixture_fname = self.model_data_fixtures[key]
        data = get_fixture_data(fixture_fname)
        bwm = models.BinWM(*data['requirements'])
        bwm._matrix = np.array(data['binary_matrix'])
        bwm.label = "My Requirements"
        return bwm

    def test_score__motorcycle_helmet(self):
        bwm = self.setup_binary_weighting_matrix('Motorcycle Helmet')

        np.testing.assert_allclose(
            bwm.score,
            np.array([0.095, 0.286, 0.143, 0.143, 0.143, 0.19]),
            atol=0.01
        )

    def test_score__simple_aircraft(self):
        bwm = self.setup_binary_weighting_matrix('Simple Aircraft')

        np.testing.assert_allclose(
            bwm.score,
            np.array([0.13, 0.16, 0.13, 0.04, 0.13, 0.09, 0.07, 0.09, 0.16]),
            atol=0.1
        )

    @pytest.mark.parametrize("answers, score", [
        [('n', 'n', 'n'), (0.17, 0.33, 0.5)],
        [('y', 'n', 'n'), (0.33, 0.17, 0.5)],
        [('n', 'y', 'n'), (0.33, 0.33, 0.33)],
        [('n', 'y', 'y'), (0.33, 0.5, 0.17)],
        [('y', 'y', 'y'), (0.5, 0.33, 0.17)]
    ])
    def test_prompt(self, answers, score, mocker):
        mock_input = mocker.patch.object(models.BinWM, '_input')
        mocker.patch.object(models.BinWM, '_print')
        mock_input.side_effect = answers
        bwm = self.setup_binary_weighting_matrix('Minimal Example')

        bwm.prompt(shuffle=False)

        mock_input.assert_has_calls([
            mocker.call("'Requirement 1' is more important than "
                        "'Requirement 2': "),
            mocker.call("'Requirement 1' is more important than "
                        "'Requirement 3': "),
            mocker.call("'Requirement 2' is more important than "
                        "'Requirement 3': ")
        ])

        np.testing.assert_allclose(bwm.score, np.array(score), atol=0.01)

    def test_prompt__shuffle(self, mocker):
        mock_input = mocker.patch.object(models.BinWM, '_input')
        mocker.patch.object(models.BinWM, '_print')
        mock_shuffle = mocker.patch('random.shuffle')
        mock_input.side_effect = ['y'] * 3
        bwm = self.setup_binary_weighting_matrix('Minimal Example')

        bwm.prompt(shuffle=True)

        mock_shuffle.assert_called_with([
            (0, 1, 'Requirement 1', 'Requirement 2'),
            (0, 2, 'Requirement 1', 'Requirement 3'),
            (1, 2, 'Requirement 2', 'Requirement 3')
        ])

    def test_to_dataframe(self):
        """Method coerces the matrix to a pandas dataframe.

        Test creates a matrix from source data and checks the
        dataframe looks right.
        """
        bwm = self.setup_binary_weighting_matrix('Minimal Example')
        expected_scores = bwm.score

        actual = bwm.to_dataframe()

        expected_requirement_labels = [
            'Requirement ' + str(x) for x in range(1, 4)
        ]

        expected = pd.DataFrame(
            data=[
                [0, 0, 1, expected_scores[0]],
                [0, 0, 1, expected_scores[1]],
                [0, 0, 0, expected_scores[2]]
            ],
            columns=expected_requirement_labels + ['Score'],
            index=expected_requirement_labels
        )
        expected.index.name = 'My Requirements'

        pd.testing.assert_frame_equal(actual, expected)

    def test_save(self):
        """Method is only implemented in special cases."""
        bwm = self.setup_binary_weighting_matrix('Minimal Example')
        bwm._matrix[2, 0] = 1
        with pytest.raises(NotImplementedError):
            bwm.save()


class TestBinWM_GoogleSheetsIntegration:

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Patch _get_sheet and configure a mock sheet from fixture data."""
        self.mock_getter = mocker.patch.object(models.BinWM, '_get_sheet')
        self._setup_mock_sheet(mocker)

    def _setup_mock_sheet(self, mocker):
        data = get_fixture_data('case__minimal_example.json')
        requirements = data['requirements']
        binary_matrix = np.array(data['binary_matrix'])

        mock_sheet = mocker.MagicMock(spec_set=io.GSheetBinWM)
        self.mock_getter.return_value = mock_sheet
        mock_sheet.get_requirements.return_value = requirements
        mock_sheet.get_value_matrix.return_value = binary_matrix
        self.mock_sheet = mock_sheet

    def test_from_google_sheet(self):
        """Constructor uses and links a google sheet to instantiate.

        Requirements and binary matrix are fetched from the
        io.BinWMSheet interface to populate the object.
        """
        bwm = models.BinWM.from_google_sheet('dummy name')

        actual_requirements = bwm.requirements
        expected_requirements = tuple(self.mock_sheet.get_requirements())
        assert actual_requirements == expected_requirements

        actual_matrix = bwm.matrix
        expected_matrix = self.mock_sheet.get_value_matrix()
        np.testing.assert_allclose(actual_matrix, expected_matrix)

    def test_access_sheet_model(self):
        """Instances access linked sheets through a generic interface."""
        bwm = models.BinWM.from_google_sheet('dummy name')

        assert bwm._sheet is self.mock_sheet

    def test_save__triggers_update(self, mocker):
        """Save method wraps the google sheet update method."""
        mock_to_dataframe = mocker.patch.object(models.BinWM, 'to_dataframe')
        mock_to_dataframe.return_value = blank_df = pd.DataFrame()

        bwm = models.BinWM.from_google_sheet('dummy name')
        bwm.save()

        self.mock_sheet.update.assert_called_once_with(blank_df)


class TestBinWM_ExcelIntegration:
    # TODO: BinWM is not currently integrated with Excel
    pass
