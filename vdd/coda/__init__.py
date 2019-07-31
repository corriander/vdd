import os

from .models import CODA
from .io import CompactExcelParser, ExcelParser

DATAD = os.path.join(os.path.dirname(__file__), 'data')
