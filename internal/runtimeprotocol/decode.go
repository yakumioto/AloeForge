package runtimeprotocol

import (
	"bytes"
	"encoding/json"
	"errors"

	"github.com/yakumioto/AloeForge/internal/strictjson"
)

// Wire fields are required and non-null, including zero/false/empty fields.
// encoding/json alone treats null scalar fields as zero values; the presence
// gate prevents a missing/null field from accidentally matching an expectation.
func fields(data []byte, names ...string) (map[string]json.RawMessage, error) {
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) != nil || len(object) != len(names) {
		return nil, errors.New("invalid wire fields")
	}
	for _, name := range names {
		value, ok := object[name]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return nil, errors.New("invalid wire fields")
		}
	}
	return object, nil
}

func idArray(data []byte) error {
	var values []json.RawMessage
	if json.Unmarshal(data, &values) != nil || values == nil {
		return errors.New("invalid wire ID array")
	}
	for _, value := range values {
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return errors.New("invalid wire ID array")
		}
	}
	return nil
}
func DecodeRuntime(data []byte, r *Runtime) error {
	if strictjson.Decode(data, r, true) != nil {
		return errors.New("invalid runtime JSON")
	}
	bindingField := "token_binding"
	if r.Profile == LocalDenseProfile {
		bindingField = "dense_contract"
	}
	object, err := fields(data, "protocol_version", "profile", "model_id", "artifact_id", bindingField, "supported_features", "max_prompt_tokens", "max_output_tokens", "context_tokens")
	if err != nil {
		return err
	}
	var binding map[string]json.RawMessage
	if bindingField == "dense_contract" {
		binding, err = fields(object[bindingField], "schema_version", "profile", "method_revision", "rng_algorithm", "rng_revision", "loader_revision", "binding_id", "tokenizer_sha256", "vocabulary_size", "logical_hidden_size", "physical_hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings", "logical_norm_epsilon", "rope_theta", "rope_layout", "cache_layout", "source_tied_embeddings", "mapped_special_token_ids")
	} else {
		binding, err = fields(object[bindingField], "schema_version", "profile", "binding_id", "tokenizer_sha256", "method_revision", "rng_algorithm", "rng_revision", "vocabulary_size", "hidden_size", "tied_embeddings", "mapped_special_token_ids")
	}
	if err != nil {
		return err
	}
	special, err := fields(binding["mapped_special_token_ids"], "bos", "eos", "pad")
	if err != nil {
		return err
	}
	for _, data := range special {
		if err := idArray(data); err != nil {
			return err
		}
	}
	return nil
}
func DecodeRequest(data []byte, r *Request) error {
	if strictjson.Decode(data, r, true) != nil {
		return errors.New("invalid request JSON")
	}
	object, err := fields(data, "protocol_version", "model_id", "artifact_id", "binding_id", "request_id", "prompt_token_ids", "stop_token_ids", "sampling", "stream")
	if err != nil {
		return err
	}
	_, err = fields(object["sampling"], "temperature", "max_tokens")
	if err != nil {
		return err
	}
	if err := idArray(object["prompt_token_ids"]); err != nil {
		return err
	}
	return idArray(object["stop_token_ids"])
}
func DecodeEvent(data []byte, e *Event) error {
	if strictjson.Decode(data, e, true) != nil {
		return errors.New("invalid event JSON")
	}
	object, err := fields(data, "request_id", "sequence", "output_token_ids", "finish_reason", "error")
	if err != nil {
		return err
	}
	return idArray(object["output_token_ids"])
}
