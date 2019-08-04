Google Sheets and Jupyter as a Requirements Weighting UI
===

`BinWM` is a class implementing a binary weighting matrix for
analysing the relative importance of requirements based on a set of
individual comparisons between each requirement in turn.


Configure Google Sheets
---

Follow the [excellent instructions by Twilio][twilio], bearing in mind
the following

  - Setting up a developer account and creating the service account is
    the same.
  - Sharing a spreadsheet with the service account is the same (and
    necessary!).
  - Save the credentials as `gsheets_credentials.json` in the
	config directory (default: `~/.config/vdd/gsheets_credentials.json`)
  - We don't need `oauth2client` or `gspread`. Dependencies are
    handled.


Jupyter
---

Assuming familiarity with Jupyter, the following should get the ball
rolling:

> Step 1 is optional, if skipped there's another way to assess the
> requirements.

 1. Optional: Weight the requirements in the spreadsheet, filling the
	upper triangular matrix with 1s or 0s depdending whether the
	requirement on the row is more important than the one on the 
	column (or vice versa).

	For example, the following matrix shows a Requirement 1 is more
	important than Requirement 2, and both 1 & 2 are less important
	than Requirement 3.

		Requirements  | Requirement 1 | Requirement 2 | Requirement 3
		-------------------------------------------------------------
		Requirement 1 |               |             1 |             0
		Requirement 2 |               |               |             0
		Requirement 3 |               |               |             

	Leave the blank cells in this example (the diagnoal and everything
	to the lower left) empty or filled with zeros.

 2. Fire up Jupyter (or IPython, or whatever) and read from the
	spreadsheet (e.g. called 'MySpreadsheet' - it must be shared with
	the service account created earlier!):

    	from vdd.requirements import BinWM

		binary_matrix = BinWM.from_google_sheets('MySpreadsheet')
		
 3. If (1) was skipped, `BinWM` provides a prompt based approach to
	populating the matrix:

		binary_matrix.prompt()
		Please agree (y) or disagree (n) with the following statements:
		'Requirement 1' is more important than 'Requirement 2':
		...

	This can be a bit tedious for large sets of requirements, as the
	approach requires comparing every combination. `prompt` does 
	accept `shuffle=True` to mix it up and reduce bias/tunnel vision.
 
 4. Once weighted, the scores can be calculated and saved back to the
	spreadsheet:

		print(binary_matrix.score)
		binary_matrix.save()

	`pandas` is a powerful tool here as scores can be plotted on bar
	charts, the binary matrix represented as a dataframe and
	visualised (`BinWM.to_dataframe()`) etc.


[twilio]: https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html
