import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/p5_validate_qwen25.py"
SPEC = importlib.util.spec_from_file_location("p5_external", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "bad", ["version", "revision", "duplicate", "empty", "bool", "extra", "oversized"]
)
def test_external_dataset_admission(bad):
    revision = "b" * 40
    document = {
        "schema_version": 1,
        "checkpoint_revision": revision,
        "cases": [{"input_ids": [0, 1], "target_ids": [2]}],
    }
    assert MODULE.dataset(json.dumps(document).encode(), revision) == document["cases"]
    if bad == "version":
        document["schema_version"] = True
    elif bad == "revision":
        document["checkpoint_revision"] = "c" * 40
    elif bad == "empty":
        document["cases"] = []
    elif bad == "bool":
        document["cases"][0]["input_ids"] = [True]
    elif bad == "extra":
        document["cases"][0]["prompt"] = "private"
    elif bad == "oversized":
        document["cases"][0]["input_ids"] = [0] * 129
    raw = json.dumps(document).encode()
    if bad == "duplicate":
        raw = raw.replace(
            b'"schema_version": 1', b'"schema_version": 1,"schema_version": 1'
        )
    with pytest.raises(ValueError):
        MODULE.dataset(raw, revision)
