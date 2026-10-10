import subprocess
import sys
from importlib.metadata import version

import aloeforge_reference


def test_installed_reference():
    assert aloeforge_reference.__version__ == version("aloeforge-bootstrap")


def test_reference_import_does_not_load_worker():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import aloeforge_reference, sys; "
            "assert 'aloeforge_worker' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )
