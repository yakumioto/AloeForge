"""Measure one Linux inspection child in a fresh process (RSS is reported in KiB)."""

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = subprocess.run(
        [str(args.binary.resolve()), "inspect", "--model-root", str(args.root)],
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
