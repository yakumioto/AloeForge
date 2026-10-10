// Package dense implements a trusted Qwen2.5 diagnostic compiler. It does not
// admit the unresolved paper/production profiles or claim privacy protection.
package dense

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sort"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/strictjson"
)

type Node struct {
	Output    string  `json:"output"`
	Source    string  `json:"source"`
	Norm      string  `json:"norm,omitempty"`
	Operation string  `json:"operation"`
	Shape     []int64 `json:"shape"`
	Shard     string  `json:"shard"`
	File      string  `json:"file"`
}

type TransformPlan struct {
	SchemaVersion int                    `json:"schema_version"`
	Contract      artifact.DenseContract `json:"contract"`
	Inventory     model.ModelInventory   `json:"inventory"`
	Nodes         []Node                 `json:"nodes"`
}

// BuildPlan is a registry admission point. qwen2 identifies the exact Qwen2
// dense semantics below, not proof of the checkpoint's marketing model name.
func BuildPlan(path string, expansion int) (TransformPlan, error) {
	inv, err := model.Inspect(path)
	if err != nil {
		return TransformPlan{}, err
	}
	adapter, ok := architectureRegistry[inv.Architecture]
	if !ok {
		return TransformPlan{}, errors.New("no dense adapter for architecture")
	}
	return adapter(path, expansion, inv)
}

var architectureRegistry = map[string]func(string, int, model.ModelInventory) (TransformPlan, error){"qwen2": buildQwen25Plan}

func buildQwen25Plan(path string, expansion int, inv model.ModelInventory) (TransformPlan, error) {
	var plan TransformPlan
	root, err := os.OpenRoot(path)
	if err != nil {
		return plan, errors.New("cannot open dense source")
	}
	defer root.Close()
	raw, err := model.ReadMetadata(root, "config.json")
	if err != nil {
		return plan, err
	}
	var cfg map[string]any
	if strictjson.Decode(raw, &cfg, false) != nil {
		return plan, errors.New("invalid dense config")
	}
	allowed := map[string]bool{}
	for _, k := range []string{"architectures", "model_type", "hidden_size", "vocab_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "rms_norm_eps", "rope_theta", "hidden_act", "tie_word_embeddings", "use_sliding_window", "sliding_window", "max_window_layers", "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id", "torch_dtype", "dtype", "transformers_version", "initializer_range", "use_cache", "_name_or_path", "quantization_config", "auto_map", "rope_scaling"} {
		allowed[k] = true
	}
	for k := range cfg {
		if !allowed[k] {
			return plan, errors.New("unsupported dense config field")
		}
	}
	integer := func(k string) int {
		n, ok := cfg[k].(json.Number)
		if !ok {
			return 0
		}
		x, e := n.Int64()
		if e != nil || x < 0 || x > 1<<30 {
			return 0
		}
		return int(x)
	}
	number := func(k string) float64 {
		n, ok := cfg[k].(json.Number)
		if !ok {
			return 0
		}
		x, _ := n.Float64()
		return x
	}
	tied, ok := cfg["tie_word_embeddings"].(bool)
	if !ok {
		return plan, errors.New("explicit tied embedding semantics required")
	}
	sliding, ok := cfg["use_sliding_window"].(bool)
	if !ok || sliding || cfg["hidden_act"] != "silu" || number("attention_dropout") != 0 {
		return plan, errors.New("unsupported dense attention or activation")
	}
	if cfg["attention_dropout"] != nil {
		if _, ok := cfg["attention_dropout"].(json.Number); !ok {
			return plan, errors.New("invalid attention dropout")
		}
	}
	c := artifact.DenseContract{SchemaVersion: 1, Profile: "diagnostic", MethodRevision: artifact.DenseMethod, RNGAlgorithm: artifact.DenseRNG, RNGRevision: "1", LoaderRevision: "numpy-qwen25-v1", BindingID: fmt.Sprintf("%064d", 0), Vocabulary: integer("vocab_size"), LogicalHidden: integer("hidden_size"), Intermediate: integer("intermediate_size"), Layers: integer("num_hidden_layers"), Heads: integer("num_attention_heads"), KVHeads: integer("num_key_value_heads"), Context: integer("max_position_embeddings"), NormEpsilon: number("rms_norm_eps"), RopeTheta: number("rope_theta"), RopeLayout: "split-half", CacheLayout: "layer-kv-head-position-dim-v1", SourceTied: tied}
	c.PhysicalHidden = c.LogicalHidden + expansion
	if c.Heads > 0 {
		c.HeadDim = c.LogicalHidden / c.Heads
	}
	c.MappedSpecialIDs = map[string][]int{}
	for _, role := range []string{"bos", "eos", "pad"} {
		ids := []int{}
		x := cfg[role+"_token_id"]
		if x != nil {
			values, ok := x.([]any)
			if !ok {
				values = []any{x}
			}
			for _, v := range values {
				n, ok := v.(json.Number)
				if !ok {
					return plan, errors.New("invalid dense special IDs")
				}
				id, e := n.Int64()
				if e != nil || id < 0 || id >= int64(c.Vocabulary) {
					return plan, errors.New("invalid dense special IDs")
				}
				ids = append(ids, int(id))
			}
		}
		c.MappedSpecialIDs[role] = ids
	}
	tokenizer, err := model.ReadMetadata(root, "tokenizer.json")
	if err != nil {
		return plan, err
	}
	var tok map[string]any
	if strictjson.Decode(tokenizer, &tok, false) != nil || tok == nil {
		return plan, errors.New("invalid tokenizer JSON")
	}
	sum := sha256.Sum256(tokenizer)
	c.TokenizerFingerprint = hex.EncodeToString(sum[:])
	if err := c.Validate(); err != nil {
		return plan, err
	}
	shapes := map[string][]int64{"model.embed_tokens.weight": {int64(c.Vocabulary), int64(c.LogicalHidden)}, "model.norm.weight": {int64(c.LogicalHidden)}}
	if !tied {
		shapes["lm_head.weight"] = []int64{int64(c.Vocabulary), int64(c.LogicalHidden)}
	}
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		shapes[prefix+"input_layernorm.weight"] = []int64{int64(c.LogicalHidden)}
		shapes[prefix+"post_attention_layernorm.weight"] = []int64{int64(c.LogicalHidden)}
		for role, shape := range map[string][]int64{"self_attn.q_proj.weight": {int64(c.Heads * c.HeadDim), int64(c.LogicalHidden)}, "self_attn.k_proj.weight": {int64(c.KVHeads * c.HeadDim), int64(c.LogicalHidden)}, "self_attn.v_proj.weight": {int64(c.KVHeads * c.HeadDim), int64(c.LogicalHidden)}, "self_attn.q_proj.bias": {int64(c.Heads * c.HeadDim)}, "self_attn.k_proj.bias": {int64(c.KVHeads * c.HeadDim)}, "self_attn.v_proj.bias": {int64(c.KVHeads * c.HeadDim)}, "self_attn.o_proj.weight": {int64(c.LogicalHidden), int64(c.LogicalHidden)}, "mlp.gate_proj.weight": {int64(c.Intermediate), int64(c.LogicalHidden)}, "mlp.up_proj.weight": {int64(c.Intermediate), int64(c.LogicalHidden)}, "mlp.down_proj.weight": {int64(c.LogicalHidden), int64(c.Intermediate)}} {
			shapes[prefix+role] = shape
		}
	}
	if len(inv.Tensors) != len(shapes) {
		return plan, errors.New("dense source tensor completeness mismatch (tied aliases must be omitted)")
	}
	for _, t := range inv.Tensors {
		if !slices.Equal(shapes[t.Name], t.Shape) || !slices.Contains([]string{"F32", "F16", "BF16"}, t.DType) {
			return plan, errors.New("unsupported dense tensor role, shape or dtype")
		}
	}
	outputs := artifact.DenseShapes(c)
	var outputBytes int64
	for _, shape := range outputs {
		elements := int64(1)
		for _, size := range shape {
			elements *= size
		}
		outputBytes += elements * 4
	}
	if outputBytes > 4<<30 {
		return plan, errors.New("dense output exceeds diagnostic 4 GiB budget")
	}
	names := make([]string, 0, len(outputs))
	for name := range outputs {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		n := Node{Output: name, Shape: outputs[name], File: fmt.Sprintf("%05d.tensor", i), Shard: "global.safetensors"}
		switch name {
		case "embedding.weight":
			n.Source = "model.embed_tokens.weight"
			n.Operation = "embedding-expand-permute-v1"
		case "head.weight":
			n.Source = "lm_head.weight"
			if tied {
				n.Source = "model.embed_tokens.weight"
			}
			n.Norm = "model.norm.weight"
			n.Operation = "head-fold-expand-permute-v1"
		default:
			var layer int
			var role, kind string
			if _, err := fmt.Sscanf(name, "layers.%d.%s", &layer, &role); err != nil {
				return plan, errors.New("invalid output role")
			}
			// Split the fixed role, not source-controlled text.
			for _, r := range []string{"q", "k", "v", "o", "gate", "up", "down"} {
				for _, k := range []string{"weight", "bias"} {
					if role == r+"."+k {
						role = r
						kind = k
					}
				}
			}
			n.Shard = fmt.Sprintf("layer-%03d.safetensors", layer)
			prefix := fmt.Sprintf("model.layers.%d.", layer)
			if slices.Contains([]string{"q", "k", "v", "o"}, role) {
				n.Source = prefix + "self_attn." + role + "_proj." + kind
			} else {
				n.Source = prefix + "mlp." + role + "_proj." + kind
			}
			switch role {
			case "q", "k", "v":
				n.Operation = "attention-input-fold-v1"
				if kind == "bias" {
					n.Operation = "attention-affine-bias-v1"
				}
			case "o":
				n.Operation = "attention-output-expand-v1"
			case "gate":
				n.Operation = "ffn-gate-fold-permute-v1"
			case "up":
				n.Operation = "ffn-up-fold-scale-permute-v1"
			case "down":
				n.Operation = "ffn-down-unscale-expand-v1"
			}
			if kind == "weight" && slices.Contains([]string{"q", "k", "v", "gate", "up"}, role) {
				n.Norm = prefix + "input_layernorm.weight"
				if role == "gate" || role == "up" {
					n.Norm = prefix + "post_attention_layernorm.weight"
				}
			}
		}
		plan.Nodes = append(plan.Nodes, n)
	}
	plan.SchemaVersion = 1
	plan.Contract = c
	plan.Inventory = inv
	return plan, nil
}
