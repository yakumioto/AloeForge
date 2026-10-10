"""Measure one Linux inspection child in a fresh process (RSS is reported in KiB).

Run only a trusted, locally built AloeForge binary. This offline developer tool
executes that binary; path validation does not authenticate its contents.
"""

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def executable_path(value: str) -> Path:
    """Require an explicit existing executable file; never search PATH."""
    try:
        path = Path(value).resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError
    except (OSError, ValueError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError("binary must be an executable file") from exc
    return path


def directory_path(value: str) -> Path:
    """Normalize even leading-dash paths into an absolute argument value."""
    try:
        path = Path(value).resolve(strict=True)
        if not path.is_dir():
            raise ValueError
    except (OSError, ValueError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError("root must be an existing directory") from exc
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=executable_path, required=True)
    parser.add_argument("--root", type=directory_path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    # Audited: trusted local executable, fixed command/flags, literal argv, no shell.
    # Shell quoting would change filenames. See docs/P1_VALIDATION.md and RSS tests.
    # nosemgrep: dangerous-subprocess-use-audit
    result = subprocess.run(
        # nosemgrep: dangerous-subprocess-use-tainted-env-args
        [str(args.binary), "inspect", "--model-root", str(args.root)],
        shell=False,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    report.update(
        exit_code=result.returncode,
        elapsed_seconds=round(time.monotonic() - start, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
