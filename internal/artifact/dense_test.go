package artifact

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"os"
	"strings"
	"testing"
)

func densePlan() (Plan, map[string][]byte) {
	c := &DenseContract{SchemaVersion: 1, Profile: "diagnostic", MethodRevision: DenseMethod, RNGAlgorithm: DenseRNG, RNGRevision: "1", LoaderRevision: "numpy-qwen25-v1", BindingID: strings.Repeat("2", 64), TokenizerFingerprint: strings.Repeat("3", 64), Vocabulary: 3, LogicalHidden: 4, PhysicalHidden: 8, Intermediate: 5, Layers: 1, Heads: 2, KVHeads: 1, HeadDim: 2, Context: 16, NormEpsilon: 1e-5, RopeTheta: 10000, RopeLayout: "split-half", CacheLayout: "layer-kv-head-position-dim-v1", SourceTied: true, MappedSpecialIDs: map[string][]int{"bos": {0}, "eos": {1}, "pad": {0}}}
	p := Plan{SchemaVersion: 1, SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("1", 64), MethodRevision: DenseMethod, RNGAlgorithm: DenseRNG, RNGRevision: "1", Dense: c}
	payloads := map[string][]byte{}
	for name, shape := range DenseShapes(*c) {
		size := int64(4)
		for _, d := range shape {
			size *= d
		}
		data := make([]byte, size)
		sum := sha256.Sum256(data)
		p.Outputs = append(p.Outputs, OutputTensor{Name: name, Shard: "model.safetensors", DType: "F32", Shape: shape, PayloadHash: hex.EncodeToString(sum[:])})
		payloads[name] = data
	}
	return p, payloads
}

func TestDenseContractRejectsIncompleteAndMixedGraphs(t *testing.T) {
	for _, kind := range []string{"missing", "shape", "profile", "rng", "mixed", "rotary", "special"} {
		t.Run(kind, func(t *testing.T) {
			p, _ := densePlan()
			switch kind {
			case "missing":
				p.Outputs = p.Outputs[1:]
			case "shape":
				p.Outputs[0].Shape[0]++
			case "profile":
				p.Dense.Profile = "production"
			case "rng":
				p.RNGRevision = "2"
			case "mixed":
				p.Binding = &TokenBinding{}
			case "rotary":
				p.Dense.RopeLayout = "interleaved"
			case "special":
				p.Dense.MappedSpecialIDs["eos"] = []int{3}
			}
			if _, err := normalizePlan(p); err == nil {
				t.Fatal("invalid dense contract admitted")
			}
		})
	}
}

func TestDensePublicationAndFiniteGate(t *testing.T) {
	for _, nonfinite := range []bool{false, true} {
		t.Run(map[bool]string{false: "finite", true: "nonfinite"}[nonfinite], func(t *testing.T) {
			p, data := densePlan()
			if nonfinite {
				name := p.Outputs[0].Name
				binary.LittleEndian.PutUint32(data[name], math.Float32bits(float32(math.NaN())))
				sum := sha256.Sum256(data[name])
				p.Outputs[0].PayloadHash = hex.EncodeToString(sum[:])
			}
			parent := t.TempDir()
			if err := os.Chmod(parent, 0700); err != nil {
				t.Fatal(err)
			}
			stage, err := Create(parent, ".stage-dense", p)
			if err != nil {
				t.Fatal(err)
			}
			defer stage.Close()
			readers := map[string]io.Reader{}
			for name, payload := range data {
				readers[name] = bytes.NewReader(payload)
			}
			if err := stage.WriteShard("model.safetensors", readers); err != nil {
				t.Fatal(err)
			}
			id, err := stage.Publish("published")
			if nonfinite {
				if err == nil {
					t.Fatal("published nonfinite dense payload")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			manifest, got, err := Verify(parent + "/published")
			if err != nil || got != id || manifest.Kind != DenseKind || manifest.Dense == nil || manifest.Binding != nil {
				t.Fatal("dense verification failed", err)
			}
		})
	}
}
