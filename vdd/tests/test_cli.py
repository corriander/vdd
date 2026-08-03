import os
import pathlib
import subprocess
import sys

import pytest

from .. import cli


DATA_DIR = (
    pathlib.Path(__file__).parent.parent / 'coda' / 'tests' / 'data'
)


class TestForceUTF8Stdio:
    """Stdio is coerced to UTF-8 so rich output can be encoded."""

    class _Stream:
        """Minimal reconfigurable stream stand-in."""

        def __init__(self, encoding):
            self.encoding = encoding
            self.reconfigured_to = None

        def reconfigure(self, encoding):
            self.reconfigured_to = encoding
            self.encoding = encoding

    def test_non_utf8_stream_is_reconfigured(self, monkeypatch):
        """A cp1252 stream (Windows default when piped) is switched."""
        stream = self._Stream('cp1252')
        monkeypatch.setattr(cli.sys, 'stdout', stream)
        monkeypatch.setattr(cli.sys, 'stderr', stream)

        cli._force_utf8_stdio()

        assert stream.reconfigured_to == 'utf-8'

    @pytest.mark.parametrize('encoding', ['utf-8', 'UTF-8', 'utf8'])
    def test_utf8_stream_is_left_alone(self, monkeypatch, encoding):
        """An already-UTF-8 stream is not needlessly reconfigured."""
        stream = self._Stream(encoding)
        monkeypatch.setattr(cli.sys, 'stdout', stream)
        monkeypatch.setattr(cli.sys, 'stderr', stream)

        cli._force_utf8_stdio()

        assert stream.reconfigured_to is None

    def test_stream_without_reconfigure_is_tolerated(self, monkeypatch):
        """A replaced stream (e.g. a test harness capture) can't fail."""
        class Captured:
            encoding = 'cp1252'

        monkeypatch.setattr(cli.sys, 'stdout', Captured())
        monkeypatch.setattr(cli.sys, 'stderr', Captured())

        cli._force_utf8_stdio()  # must not raise


class TestCODACommandEncoding:
    """Regression: rendering must survive a non-UTF-8 stdout.

    The table borders and satisfaction bars use box-drawing and block
    characters. On Windows a redirected/piped stdout defaults to the
    locale encoding (cp1252), which cannot represent them. Forcing the
    child's encoding reproduces that condition on any platform.
    """

    def test_coda_renders_to_a_non_utf8_pipe(self, tmp_path):
        env = dict(os.environ, PYTHONIOENCODING='cp1252')
        model = DATA_DIR / 'demo_model_compact.xlsx'

        proc = subprocess.run(
            [sys.executable, '-c', 'from vdd.cli import app; app()',
             'coda', str(model)],
            capture_output=True,
            env=env,
        )

        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        stdout = proc.stdout.decode('utf-8')
        assert 'Merit:' in stdout
        assert '█' in stdout  # the satisfaction bar survived
