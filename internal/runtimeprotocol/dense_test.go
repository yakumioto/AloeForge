package runtimeprotocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

func localDenseFixture() Runtime {
	r := runtimeFixture()
	r.Profile = LocalDenseProfile
	r.Binding = artifact.TokenBinding{}
	r.Dense = &artifact.DenseContract{SchemaVersion: 1, Profile: "diagnostic", MethodRevision: artifact.DenseMethod, RNGAlgorithm: artifact.DenseRNG, RNGRevision: "1", LoaderRevision: "numpy-qwen25-v1", BindingID: strings.Repeat("b", 64), TokenizerFingerprint: strings.Repeat("c", 64), Vocabulary: 16, LogicalHidden: 8, PhysicalHidden: 12, Intermediate: 10, Layers: 2, Heads: 2, KVHeads: 1, HeadDim: 4, Context: 64, NormEpsilon: 1e-5, RopeTheta: 10000, RopeLayout: "split-half", CacheLayout: "layer-kv-head-position-dim-v1", MappedSpecialIDs: map[string][]int{"bos": {0}, "eos": {1}, "pad": {0}}}
	return r
}

func TestLocalDenseIdentityAndDetachedCopy(t *testing.T) {
	r := localDenseFixture()
	if r.Validate() != nil || r.BindingID() != r.Dense.BindingID || r.VocabularySize() != 16 {
		t.Fatal("dense fixture rejected")
	}
	data, _ := json.Marshal(r)
	if bytes.Contains(data, []byte("token_binding")) {
		t.Fatal("mixed wire binding")
	}
	var got Runtime
	if DecodeRuntime(data, &got) != nil || got.Validate() != nil || !Equal(got, r) {
		t.Fatal("dense wire mismatch")
	}
	copy := Clone(r)
	copy.Dense.MappedSpecialIDs["eos"][0] = 3
	if r.Dense.MappedSpecialIDs["eos"][0] != 1 {
		t.Fatal("contract aliases caller")
	}
	for _, mutate := range []func(*Runtime){
		func(r *Runtime) { r.Profile = Profile },
		func(r *Runtime) { r.Binding = runtimeFixture().Binding },
		func(r *Runtime) { r.Dense = nil },
		func(r *Runtime) { r.Dense.Profile = "production" },
		func(r *Runtime) { r.ContextTokens = r.Dense.Context + 1 },
		func(r *Runtime) { r.Dense.HeadDim = 3 },
	} {
		x := Clone(r)
		mutate(&x)
		if x.Validate() == nil {
			t.Fatal("invalid dense runtime admitted")
		}
	}
}

func TestDenseWireAllFieldsStrict(t *testing.T) {
	raw, _ := json.Marshal(localDenseFixture())
	var document map[string]any
	_ = json.Unmarshal(raw, &document)
	// Every field at every level is required, including false and special arrays.
	var check func(map[string]any)
	check = func(object map[string]any) {
		for key, value := range object {
			for _, missing := range []bool{false, true} {
				object[key] = nil
				if missing {
					delete(object, key)
				}
				bad, _ := json.Marshal(document)
				var got Runtime
				if DecodeRuntime(bad, &got) == nil {
					t.Fatal("missing/null dense field admitted")
				}
				object[key] = value
			}
			if child, ok := value.(map[string]any); ok {
				check(child)
			}
		}
	}
	check(document)
	for _, bad := range [][]byte{
		bytes.Replace(raw, []byte(`"profile":"p6-local-dense-reference"`), []byte(`"profile":"p6-local-dense-reference","token_binding":{}`), 1),
		bytes.Replace(raw, []byte(`"schema_version":1`), []byte(`"schema_version":1,"schema_version":1`), 1),
		bytes.Replace(raw, []byte(`"bos":[0]`), []byte(`"bos":[null]`), 1),
		bytes.Replace(raw, []byte(`"source_tied_embeddings":false`), []byte(`"source_tied_embeddings":0`), 1),
	} {
		var got Runtime
		if DecodeRuntime(bad, &got) == nil {
			t.Fatal("invalid dense schema admitted")
		}
	}
}
