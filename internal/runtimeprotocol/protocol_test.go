package runtimeprotocol

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

func runtimeFixture() Runtime {
	return Runtime{ProtocolVersion: Version, Profile: Profile, ModelID: "fake", ArtifactID: strings.Repeat("a", 64), Binding: artifact.TokenBinding{SchemaVersion: 1, Profile: artifact.TokenProfile, BindingID: strings.Repeat("b", 64), TokenizerFingerprint: strings.Repeat("c", 64), MethodRevision: artifact.TokenMethod, RNGAlgorithm: artifact.TokenRNG, RNGRevision: artifact.TokenRNGRevision, VocabularySize: 16, HiddenSize: 2, MappedSpecialIDs: map[string][]int{"bos": {0}, "eos": {1}, "pad": {2}}}, Features: []string{"greedy", "sse", "stop_token_ids"}, MaxPromptTokens: 8, MaxOutputTokens: 8, ContextTokens: 12}
}
func TestRuntimeGatesAndDetachedCopy(t *testing.T) {
	r := runtimeFixture()
	if r.Validate() != nil {
		t.Fatal("valid runtime rejected")
	}
	c := Clone(r)
	c.Binding.MappedSpecialIDs["bos"][0] = 3
	c.Features[0] = "text"
	if r.Binding.MappedSpecialIDs["bos"][0] != 0 || r.Features[0] != "greedy" {
		t.Fatal("capability aliases caller")
	}
	for _, mutate := range []func(*Runtime){func(r *Runtime) { r.Profile = "production" }, func(r *Runtime) { r.ModelID = "../text" }, func(r *Runtime) { r.Binding.MethodRevision = "future" }, func(r *Runtime) { r.Binding.VocabularySize = 1 }, func(r *Runtime) { r.Binding.HiddenSize = 0 }, func(r *Runtime) { r.Features = []string{"greedy", "sse"} }, func(r *Runtime) { r.ContextTokens = MaxTokens*2 + 1 }, func(r *Runtime) { r.MaxPromptTokens = 0 }, func(r *Runtime) { r.MaxOutputTokens = r.ContextTokens + 1 }, func(r *Runtime) { r.Binding.MappedSpecialIDs["eos"] = []int{16} }} {
		c := Clone(r)
		mutate(&c)
		if c.Validate() == nil {
			t.Fatal("unsupported capability admitted")
		}
	}
}
func TestAllWireFieldsAreRequiredAndNonNull(t *testing.T) {
	r := runtimeFixture()
	temp := 0.0
	q := Request{ProtocolVersion: 1, ModelID: r.ModelID, ArtifactID: r.ArtifactID, BindingID: r.Binding.BindingID, RequestID: strings.Repeat("a", 32), PromptTokenIDs: []int{3}, StopTokenIDs: []int{}, Sampling: Sampling{Temperature: &temp, MaxTokens: 3}, Stream: true}
	e := Event{RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{3}, FinishReason: "", Error: ""}
	for _, tc := range []struct {
		value  any
		decode func([]byte) error
	}{{r, func(b []byte) error { var x Runtime; return DecodeRuntime(b, &x) }}, {q, func(b []byte) error { var x Request; return DecodeRequest(b, &x) }}, {e, func(b []byte) error { var x Event; return DecodeEvent(b, &x) }}} {
		b, _ := json.Marshal(tc.value)
		if tc.decode(b) != nil {
			t.Fatal("valid schema rejected")
		}
		var object map[string]json.RawMessage
		_ = json.Unmarshal(b, &object)
		for name := range object {
			for _, null := range []bool{false, true} {
				var copy map[string]json.RawMessage
				_ = json.Unmarshal(b, &copy)
				if null {
					copy[name] = json.RawMessage("null")
				} else {
					delete(copy, name)
				}
				bad, _ := json.Marshal(copy)
				if tc.decode(bad) == nil {
					t.Fatalf("missing/null field admitted: %s", name)
				}
			}
		}
	}
	b, _ := json.Marshal(r)
	b = []byte(strings.Replace(string(b), `"tied_embeddings":false`, `"tied_embeddings":null`, 1))
	var got Runtime
	if DecodeRuntime(b, &got) == nil {
		t.Fatal("null nested boolean admitted")
	}
	b, _ = json.Marshal(q)
	b = []byte(strings.Replace(string(b), `"temperature":0`, `"temperature":null`, 1))
	var req Request
	if DecodeRequest(b, &req) == nil {
		t.Fatal("null sampling admitted")
	}
}
func TestContextBoundAndEventForms(t *testing.T) {
	r := runtimeFixture()
	temp := 0.0
	q := Request{ProtocolVersion: 1, ModelID: r.ModelID, ArtifactID: r.ArtifactID, BindingID: r.Binding.BindingID, RequestID: strings.Repeat("a", 32), PromptTokenIDs: make([]int, 8), StopTokenIDs: []int{}, Sampling: Sampling{Temperature: &temp, MaxTokens: 8}, Stream: true}
	if q.Validate(r) == nil {
		t.Fatal("combined context bound ignored")
	}
	q.Sampling.MaxTokens = 4
	if q.Validate(r) != nil {
		t.Fatal("exact context bound rejected")
	}
	for _, e := range []Event{{RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{1, 2}}, {RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{}, FinishReason: "error", Error: "plaintext"}, {RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{1}, FinishReason: "eos"}, {RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{}, FinishReason: "future"}} {
		if e.Validate(q.RequestID, 1, 16) == nil {
			t.Fatal("invalid SSE form accepted")
		}
	}
}

func TestNullIDElementsDoNotBecomeZero(t *testing.T) {
	r := runtimeFixture()
	data, _ := json.Marshal(r)
	bad := []byte(strings.Replace(string(data), `"bos":[0]`, `"bos":[null]`, 1))
	var runtime Runtime
	if DecodeRuntime(bad, &runtime) == nil {
		t.Fatal("null special ID became zero")
	}
	temp := 0.0
	q := Request{ProtocolVersion: 1, ModelID: r.ModelID, ArtifactID: r.ArtifactID, BindingID: r.Binding.BindingID, RequestID: strings.Repeat("a", 32), PromptTokenIDs: []int{0}, StopTokenIDs: []int{0}, Sampling: Sampling{Temperature: &temp, MaxTokens: 1}, Stream: true}
	data, _ = json.Marshal(q)
	for _, key := range []string{"prompt_token_ids", "stop_token_ids"} {
		bad = []byte(strings.Replace(string(data), `"`+key+`":[0]`, `"`+key+`":[null]`, 1))
		var req Request
		if DecodeRequest(bad, &req) == nil {
			t.Fatal("null request ID became zero")
		}
	}
	e := Event{RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: []int{0}}
	data, _ = json.Marshal(e)
	bad = []byte(strings.Replace(string(data), `"output_token_ids":[0]`, `"output_token_ids":[null]`, 1))
	var event Event
	if DecodeEvent(bad, &event) == nil {
		t.Fatal("null output ID became zero")
	}
}
