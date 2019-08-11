Tools for Value-Driven Design
=============================

[![Build Status][master-build-status]][azure-pipeline]


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

Roadmap
-------

![Azure DevOps builds (branch)][develop-build-status]

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

<!-- statuses -->
[azure-pipeline]: https://dev.azure.com/corriander/github-public/_build/latest?definitionId=2&branchName=master
[master-build-status]: https://dev.azure.com/corriander/github-public/_apis/build/status/corriander.vdd?branchName=master
[develop-build-status]: https://img.shields.io/azure-devops/build/corriander/8c97c580-4bf1-4e14-80b2-1be44ecc86f6/2/develop?label=develop
[binwm-gsheets]: ./docs/gsheets-integration.md
