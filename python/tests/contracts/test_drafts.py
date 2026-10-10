"""Test-only draft structures; deliberately not a production schema validator."""

import copy
import json
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

DRAFT = Path(__file__).resolve().parents[3] / "docs/contracts/bootstrap.json"


@dataclass
class ModelIR:
    architecture_id: None = None
    tokenizer_fingerprint: None = None
    source_tensor_inventory: None = None
    output_tensor_inventory: None = None
    logical_hidden_dimensions: None = None
    physical_hidden_dimensions: None = None
    attention_layout: None = None
    normalization: None = None
    rope_layout: None = None
    ffn_layout: None = None
    layer_tensor_roles: None = None
    moe_layout: None = None
    mla_layout: None = None
    sharding: None = None
    quantization: None = None


@dataclass
class MethodContract:
    method_revision: None = None
    evidence_label: None = None
    paper_reference: None = None
    profile: None = None
    rng_algorithm: None = None
    rng_revision: None = None
    transform_specification: None = None
    acceptance_gates: None = None


@dataclass
class RuntimeContract:
    protocol_revision: None = None
    artifact_id: None = None
    method_revision: None = None
    binding_id: None = None
    tokenizer_fingerprint: None = None
    loader_revision: None = None
    cache_layout: None = None
    mapped_special_token_ids: None = None
    context_bounds: None = None
    supported_features: None = None


@dataclass
class SecretBinding:
    artifact_id: None = None
    binding_id: None = None
    tokenizer_fingerprint: None = None
    method_revision: None = None
    protocol_revision: None = None


STRUCTURES = (ModelIR, MethodContract, RuntimeContract, SecretBinding)


def validate_draft(structure, value):
    """Reject deviations from the documented P0 placeholder, not real models."""
    names = {field.name for field in fields(structure)}
    if set(value) != names | {"schema_version", "status"}:
        raise ValueError("draft fields differ")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported draft version")
    if value["status"] != "unimplemented":
        raise ValueError("draft cannot claim implementation")
    if any(value[name] is not None for name in names):
        raise ValueError("P0 fields must be unimplemented")
    return structure(**{name: value[name] for name in names})


@pytest.fixture
def draft():
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def test_exact_contract_set(draft):
    assert set(draft) == {structure.__name__ for structure in STRUCTURES}


@pytest.mark.parametrize("structure", STRUCTURES)
def test_documented_structure(draft, structure):
    assert validate_draft(structure, draft[structure.__name__]) == structure()


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize(
    "mutation",
    ["version", "boolean_version", "status", "missing", "unknown", "populated"],
)
def test_reject_invalid_drafts(draft, structure, mutation):
    value = copy.deepcopy(draft[structure.__name__])
    first_field = fields(structure)[0].name
    if mutation == "version":
        value["schema_version"] = 2
    elif mutation == "boolean_version":
        value["schema_version"] = True
    elif mutation == "status":
        value["status"] = "implemented"
    elif mutation == "missing":
        del value[first_field]
    elif mutation == "unknown":
        value["unexpected"] = None
    else:
        value[first_field] = "placeholder-must-stay-null"
    with pytest.raises(ValueError):
        validate_draft(structure, value)


def test_public_binding_contains_only_identity_fields(draft):
    assert set(draft["SecretBinding"]) == {
        "schema_version",
        "status",
        "artifact_id",
        "binding_id",
        "tokenizer_fingerprint",
        "method_revision",
        "protocol_revision",
    }
