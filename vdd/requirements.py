from __future__ import division

import itertools
import random

import numpy as np


class BinWM(object):
    """Binary Weighting Matrix

    Used to model relative importance of requirements. Each
    requirement is assessed to be less or more important than every
    other requirement in turn. This allows us to calculate a weighted
    set of requirements.
    """

    def __init__(self, *args):
        self.requirements = args
        self._matrix = np.matrix(np.zeros([len(args), len(args)]))

    @property
    def matrix(self):
        """Copy of the weighting matrix."""
        return np.copy(self._matrix)

    @property
    def score(self):
        """Calculate the relative score."""
        sum_x = self.matrix.sum(axis=1)
        sum_y = np.triu(1 - self.matrix, k=1).sum(axis=0).T

        sum_combined = sum_x + sum_y
        sum_biased = sum_combined + 1

        return sum_biased / sum_biased.sum()

    def _input(prompt_string):
        # Wrapper for testing
        return input(prompt_string)

    def _print(string):
        # Wrapper for testing
        print(string)

    def prompt(self, shuffle=True):
        """Step through an interactive prompt to calculate weighting.

        Parameters
        ----------

        shuffle: bool
            Shuffle the comparisons so each decision is presented in
            random order.
        """
        reqs = self.requirements
        combinations = itertools.combinations(reqs, 2)
        coordinates = itertools.combinations(range(len(reqs)), 2)

        self._print("Please agree (y) or disagree (n) with the "
                    "following statements:\n")
        iterable = zip(coordinates, combinations)
        decisions = []
        for x, grp in itertools.groupby(iterable, lambda o: o[1][0]):
            for (i, j), (this, other) in grp:
                decisions.append((i, j, this, other))

        if shuffle:
            random.shuffle(decisions)

        for i, j, this, other in decisions:

                # Keep asking until response is valid
                while True:
                    response = self._input(
                        "{} is more important than {}: "
                        .format(this, other)
                    )
                    if response in 'yn':
                        break
                    else:
                        self._print(
                            "Sorry I didn't understand...\n\n"
                        )

                self._matrix[i, j] = 1 if response == 'y' else 0
