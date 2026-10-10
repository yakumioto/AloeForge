import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "p1_measure_rss.py"


@pytest.fixture
def measure():
    return runpy.run_path(str(SCRIPT))


def write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return path


@pytest.mark.parametrize(
    "name",
    [
        "ordinary path",
        "--leading-dash",
        "literal ; touch INJECTED ; $(touch SUBSTITUTED) `touch BACKTICK` '\"",
    ],
)
def test_paths_are_literal_arguments(measure, tmp_path, monkeypatch, capsys, name):
    monkeypatch.chdir(tmp_path)
    binary = write_executable(
        tmp_path / f"binary {name}",
        "import json, sys; print(json.dumps({'argv': sys.argv[1:]}))",
    )
    root = tmp_path / name
    root.mkdir()
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--binary", binary.name, f"--root={root.name}"]
    )

    measure["main"]()

    report = json.loads(capsys.readouterr().out)
    assert report["argv"] == ["inspect", "--model-root", str(root.resolve())]
    assert report["exit_code"] == 0
    assert report["elapsed_seconds"] >= 0
    assert report["peak_rss_kib"] > 0
    for marker in ("INJECTED", "SUBSTITUTED", "BACKTICK"):
        assert not (tmp_path / marker).exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "missing-binary",
        "directory-binary",
        "nonexecutable-binary",
        "fifo-binary",
        "missing-root",
        "file-root",
        "nul-binary",
        "nul-root",
    ],
)
def test_invalid_paths_fail_before_launch(
    measure, tmp_path, monkeypatch, capsys, invalid
):
    binary = write_executable(tmp_path / "trusted", "print('{}')")
    root = tmp_path / "root"
    root.mkdir()
    bad_file = tmp_path / "not-executable"
    bad_file.write_text("data", encoding="utf-8")
    bad_file.chmod(0o600)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    binary_arg = {
        "missing-binary": str(tmp_path / "missing"),
        "directory-binary": str(root),
        "nonexecutable-binary": str(bad_file),
        "fifo-binary": str(fifo),
        "nul-binary": "bad\0binary",
    }.get(invalid, str(binary))
    root_arg = {
        "missing-root": str(tmp_path / "missing"),
        "file-root": str(bad_file),
        "nul-root": "bad\0root",
    }.get(invalid, str(root))

    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid paths must fail before launching a child")

    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--binary", binary_arg, "--root", root_arg]
    )
    with pytest.raises(SystemExit) as exc:
        measure["main"]()
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("import sys; sys.exit(7)", subprocess.CalledProcessError),
        ("print('not JSON')", json.JSONDecodeError),
    ],
)
def test_child_failure_is_not_reported_as_success(
    measure, tmp_path, monkeypatch, capsys, body, error
):
    binary = write_executable(tmp_path / "trusted", body)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--binary", str(binary), "--root", str(tmp_path)]
    )
    with pytest.raises(error) as exc:
        measure["main"]()
    if error is subprocess.CalledProcessError:
        assert exc.value.returncode == 7
    assert capsys.readouterr().out == ""
