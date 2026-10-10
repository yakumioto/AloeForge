package artifact

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tokenPlan() Plan {
	return Plan{SchemaVersion: 1, SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("1", 64), MethodRevision: TokenMethod, RNGAlgorithm: TokenRNG, RNGRevision: TokenRNGRevision,
		Outputs: []OutputTensor{{Name: "embedding.weight", Shard: "model.safetensors", DType: "F32", Shape: []int64{4, 2}, PayloadHash: payloadHash(make([]byte, 32))}},
		Binding: &TokenBinding{SchemaVersion: 1, Profile: TokenProfile, BindingID: strings.Repeat("2", 64), TokenizerFingerprint: strings.Repeat("3", 64), MethodRevision: TokenMethod, RNGAlgorithm: TokenRNG, RNGRevision: TokenRNGRevision, VocabularySize: 4, HiddenSize: 2, TiedEmbeddings: true, MappedSpecialIDs: map[string][]int{"bos": {}, "eos": {1, 3}, "pad": {1}}}}
}

func TestNonfiniteTokenOutputCannotPublish(t *testing.T) {
	parent := privateParent(t)
	p := tokenPlan()
	payload := make([]byte, 32)
	binary.LittleEndian.PutUint32(payload, math.Float32bits(float32(math.NaN())))
	p.Outputs[0].PayloadHash = payloadHash(payload)
	stage, err := Create(parent, ".stage-nonfinite", p)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if err := stage.WriteShard("model.safetensors", map[string]io.Reader{"embedding.weight": bytes.NewReader(payload)}); err != nil {
		t.Fatal(err)
	}
	if _, err := stage.Publish("visible"); err == nil {
		t.Fatal("nonfinite P2 output published through lower-level writer")
	}
	if _, err := os.Stat(filepath.Join(parent, "visible")); !os.IsNotExist(err) {
		t.Fatal("failed validation made output visible")
	}
}

func TestTokenPlanNegativeGates(t *testing.T) {
	tests := map[string]func(*Plan){
		"profile":        func(p *Plan) { p.Binding.Profile = "production" },
		"schema":         func(p *Plan) { p.Binding.SchemaVersion = 2 },
		"commitment":     func(p *Plan) { p.Binding.BindingID = "secret-input" },
		"tokenizer":      func(p *Plan) { p.Binding.TokenizerFingerprint = "invalid" },
		"method":         func(p *Plan) { p.Binding.MethodRevision = "unknown" },
		"plan-method":    func(p *Plan) { p.MethodRevision = "unknown" },
		"plan-rng":       func(p *Plan) { p.RNGAlgorithm = "none" },
		"rng-version":    func(p *Plan) { p.Binding.RNGRevision = "2" },
		"layout":         func(p *Plan) { p.Binding.TiedEmbeddings = false },
		"shape":          func(p *Plan) { p.Outputs[0].Shape = []int64{2, 4} },
		"dtype":          func(p *Plan) { p.Outputs[0].DType = "I32" },
		"name":           func(p *Plan) { p.Outputs[0].Name = "arbitrary.weight" },
		"shard":          func(p *Plan) { p.Outputs[0].Shard = "arbitrary.safetensors" },
		"special-role":   func(p *Plan) { p.Binding.MappedSpecialIDs["private"] = []int{0} },
		"special-bounds": func(p *Plan) { p.Binding.MappedSpecialIDs["bos"] = []int{4} },
		"null-special":   func(p *Plan) { p.Binding.MappedSpecialIDs["bos"] = nil },
		"unsorted":       func(p *Plan) { p.Binding.MappedSpecialIDs["eos"] = []int{3, 1} },
		"empty-eos":      func(p *Plan) { p.Binding.MappedSpecialIDs["eos"] = []int{} },
		"vocab-limit":    func(p *Plan) { p.Binding.VocabularySize = MaxDiagnosticVocab + 1 },
		"hidden-limit":   func(p *Plan) { p.Binding.HiddenSize = MaxDiagnosticHidden + 1 },
		"element-limit": func(p *Plan) {
			p.Binding.VocabularySize = MaxDiagnosticVocab
			p.Binding.HiddenSize = MaxDiagnosticHidden
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			p := tokenPlan()
			mutate(&p)
			if _, err := normalizePlan(p); err == nil {
				t.Fatal("unsupported token plan admitted")
			}
		})
	}
}

func TestPublicTokenBindingSnapshotAndRecovery(t *testing.T) {
	parent := privateParent(t)
	p := tokenPlan()
	stage, err := Create(parent, ".stage-token", p)
	if err != nil {
		t.Fatal(err)
	}
	p.Binding.MappedSpecialIDs["eos"][0] = 0
	p.Binding.BindingID = strings.Repeat("4", 64)
	pending, err := stage.PendingShards()
	if err != nil || len(pending) != 1 {
		t.Fatal("pending shard count")
	}
	if err := stage.WriteShard("model.safetensors", map[string]io.Reader{"embedding.weight": bytes.NewReader(make([]byte, 32))}); err != nil {
		t.Fatal(err)
	}
	pending, err = stage.PendingShards()
	if err != nil || len(pending) != 0 {
		t.Fatal("completed shard listed pending")
	}
	digest, err := stage.Publish("visible")
	if err != nil {
		t.Fatal(err)
	}
	stage.Close()
	if _, err := stage.PendingShards(); err == nil {
		t.Fatal("closed stage returned pending shards")
	}
	manifest, got, err := Verify(filepath.Join(parent, "visible"))
	if err != nil || got != digest || manifest.Kind != TokenKind || manifest.Binding.BindingID != strings.Repeat("2", 64) || manifest.Binding.MappedSpecialIDs["eos"][0] != 1 {
		t.Fatal("binding mutated after stage creation", err)
	}
	if resumed, err := Resume(parent, ".stage-token", p); err == nil {
		resumed.Close()
		t.Fatal("changed token binding resumed")
	}
	root, err := os.OpenRoot(filepath.Join(parent, "visible"))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	for _, kind := range []string{"unknown", "missing-binding", "p1-with-binding", "private-field"} {
		t.Run(kind, func(t *testing.T) {
			changed := manifest
			switch kind {
			case "unknown":
				changed.Kind = "full-v2"
			case "missing-binding":
				changed.Binding = nil
			case "p1-with-binding":
				changed.Kind = Kind
			}
			if err := atomicJSON(root, "manifest.json", changed); err != nil {
				t.Fatal(err)
			}
			if kind == "private-field" {
				path := filepath.Join(parent, "visible", "manifest.json")
				data, _ := os.ReadFile(path)
				data = bytes.Replace(data, []byte(`"token_binding":{`), []byte(`"token_binding":{"permutation":[0,1,2,3],`), 1)
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, _, err := Verify(filepath.Join(parent, "visible")); err == nil {
				t.Fatal("invalid binding/kind admitted")
			}
		})
	}
}
