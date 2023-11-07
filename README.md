Tools for Value-Driven Design
=============================

<!-- badges -->

[![cc-angular][cc-angular-badge]][cc-home]
[![release-tool][release-please-badge]][release-please-gh]

<!-- / -->


Tools intended to help with modelling decisions in a value centric
design process. The intent is to keep this as generic as possible, as
some of this decision modelling is suited to generic decision-making,
non-design activities with a little massaging.


Features
-------

  - Concept Design Analysis (CODA) method implementation
  - Requirements weighting with a Binary Weighting Matrix
  - Programmatic or Spreadsheet based model creation (via Excel
    workbooks or Google Sheets).


Install
-------

    pip install vdd


Documentation
-------

Currently just stored in the repo.

  - [Using Google Sheets for requirements matrices][binwm-gsheets]
  - tbc


Development
-----------

`poetry` must be installed in the local development environment as a pre-requisite. In the repository root:


	poetry install


### Releases

Managed by [Release Please][release-please-gh] with auto-versioning.
Changes to default branch will be accumulated in a release PR based on [Conventional Commits][cc-home].
Merging the release PR will automatically publish the versioned package to PyPI.


Roadmap
-------

  - Model sets for comparative work (rather than a single set of
	characteristic parameter values)
  - Improved visualisation
  - Export CODA models to Excel template
  - House of Quality style requirement/characteristic weighting
  - Pandas everywhere (v1.x)


References
----------

Based on my own degree notes and open access literature:

  - M.H. Eres et al, 2014. Mapping Customer Needs to Engineering
	Characteristics: An Aerospace Perspective for Conceptual Design -
	Journal of Engineering Design pp. 1-24
	<http://eprints.soton.ac.uk/id/eprint/361442>




<!-- links -->

[cc-angular-badge]: https://img.shields.io/badge/conventional_commits-angular-FE5196?logo=conventionalcommits&
[cc-home]: https://www.conventionalcommits.org
[release-please-badge]: https://img.shields.io/badge/release--please-python-4285F4?logo=google
[release-please-gh]: https://github.com/googleapis/release-please

[binwm-gsheets]: ./docs/gsheets-integration.md
