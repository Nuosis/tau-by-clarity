import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_active_compressor_parses_on_supported_python_311(tmp_path: Path) -> None:
    path = ROOT / "src" / "pi_coding_agent" / "active_compression" / "compressor.py"
    python311 = (
        sys.executable
        if sys.version_info[:2] == (3, 11)
        else shutil.which("python3.11")
    )
    if python311 is None:
        pytest.skip("Python 3.11 is unavailable")

    result = subprocess.run(
        [python311, "-m", "py_compile", str(path)],
        env={**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
