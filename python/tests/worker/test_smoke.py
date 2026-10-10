import subprocess
import sys
from importlib.metadata import version

import aloeforge_worker


def test_installed_worker():
    assert aloeforge_worker.__version__ == version("aloeforge-bootstrap")


def test_worker_import_does_not_load_reference():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import aloeforge_worker, sys; "
            "assert 'aloeforge_reference' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )
