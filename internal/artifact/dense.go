package artifact

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

const (
	DenseKind   = "p5-qwen25-dense-diagnostic-only"
	DenseMethod = "qwen25-isometry-noise-off-v1"
	DenseRNG    = "crypto-rand-sha256-pcg64"
)

// DenseContract is public loader metadata. Private coordinate maps, source
// fingerprints, RNG seed, tokenizer files and original IDs are never included.
type DenseContract struct {
	SchemaVersion        int              `json:"schema_version"`
	Profile              string           `json:"profile"`
	MethodRevision       string           `json:"method_revision"`
	RNGAlgorithm         string           `json:"rng_algorithm"`
	RNGRevision          string           `json:"rng_revision"`
	LoaderRevision       string           `json:"loader_revision"`
	BindingID            string           `json:"binding_id"`
	TokenizerFingerprint string           `json:"tokenizer_sha256"`
	Vocabulary           int              `json:"vocabulary_size"`
	LogicalHidden        int              `json:"logical_hidden_size"`
	PhysicalHidden       int              `json:"physical_hidden_size"`
	Intermediate         int              `json:"intermediate_size"`
	Layers               int              `json:"num_hidden_layers"`
	Heads                int              `json:"num_attention_heads"`
	KVHeads              int              `json:"num_key_value_heads"`
	HeadDim              int              `json:"head_dim"`
	Context              int              `json:"max_position_embeddings"`
	NormEpsilon          float64          `json:"logical_norm_epsilon"`
	RopeTheta            float64          `json:"rope_theta"`
	RopeLayout           string           `json:"rope_layout"`
	CacheLayout          string           `json:"cache_layout"`
	SourceTied           bool             `json:"source_tied_embeddings"`
	MappedSpecialIDs     map[string][]int `json:"mapped_special_token_ids"`
}

func (c DenseContract) Validate() error {
	if c.SchemaVersion != 1 || c.Profile != "diagnostic" || c.MethodRevision != DenseMethod || c.RNGAlgorithm != DenseRNG || c.RNGRevision != "1" || c.LoaderRevision != "numpy-qwen25-v1" || !validHash(c.BindingID) || !validHash(c.TokenizerFingerprint) {
		return errors.New("unsupported dense diagnostic revision or binding")
	}
	if c.Vocabulary < 2 || c.Vocabulary > 262144 || c.LogicalHidden < 2 || c.LogicalHidden > 8192 || c.PhysicalHidden <= c.LogicalHidden || c.PhysicalHidden > c.LogicalHidden+128 || (c.PhysicalHidden-c.LogicalHidden)%4 != 0 || c.Intermediate < 2 || c.Intermediate > 32768 || c.Layers < 1 || c.Layers > 64 || c.Heads < 1 || c.Heads > 64 || c.KVHeads < 1 || c.KVHeads > c.Heads || c.Heads%c.KVHeads != 0 || c.HeadDim < 2 || c.HeadDim > 256 || c.HeadDim%2 != 0 || c.Heads*c.HeadDim != c.LogicalHidden || c.Context < 1 || c.Context > 131072 {
		return errors.New("unsupported dense diagnostic geometry")
	}
	if math.IsNaN(c.NormEpsilon) || math.IsInf(c.NormEpsilon, 0) || c.NormEpsilon <= 0 || c.NormEpsilon > 1 || math.IsNaN(c.RopeTheta) || math.IsInf(c.RopeTheta, 0) || c.RopeTheta < 1 || c.RopeTheta > 1e9 || c.RopeLayout != "split-half" || c.CacheLayout != "layer-kv-head-position-dim-v1" {
		return errors.New("unsupported dense norm, rotary or cache semantics")
	}
	return ValidateSpecialIDs(c.MappedSpecialIDs, c.Vocabulary)
}

// DenseShapes defines the complete custom graph, including explicit untied head
// after norm folding even when the source embedding and head were tied.
func DenseShapes(c DenseContract) map[string][]int64 {
	d, v, f := int64(c.PhysicalHidden), int64(c.Vocabulary), int64(c.Intermediate)
	q, kv := int64(c.Heads*c.HeadDim), int64(c.KVHeads*c.HeadDim)
	shapes := map[string][]int64{"embedding.weight": {v, d}, "head.weight": {v, d}}
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("layers.%d.", i)
		for role, shape := range map[string][]int64{"q.weight": {q, d}, "q.bias": {q}, "k.weight": {kv, d}, "k.bias": {kv}, "v.weight": {kv, d}, "v.bias": {kv}, "o.weight": {d, q}, "gate.weight": {f, d}, "up.weight": {f, d}, "down.weight": {d, f}} {
			shapes[prefix+role] = shape
		}
	}
	return shapes
}

func normalizeDense(c *DenseContract, outputs []OutputTensor) (*DenseContract, error) {
	if c == nil {
		return nil, nil
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	shapes := DenseShapes(*c)
	if len(shapes) != len(outputs) {
		return nil, errors.New("dense output tensor completeness mismatch")
	}
	for _, t := range outputs {
		if t.DType != "F32" || !slices.Equal(shapes[t.Name], t.Shape) {
			return nil, errors.New("dense output role, shape or dtype mismatch")
		}
	}
	copy := *c
	copy.MappedSpecialIDs = make(map[string][]int)
	for role, ids := range c.MappedSpecialIDs {
		copy.MappedSpecialIDs[role] = slices.Clone(ids)
	}
	return &copy, nil
}
