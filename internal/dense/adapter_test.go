package dense

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yakumioto/AloeForge/internal/model"
)

func fixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	config := `{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":4,"vocab_size":3,"intermediate_size":5,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,"rms_norm_eps":0.00001,"rope_theta":10000,"hidden_act":"silu","tie_word_embeddings":true,"use_sliding_window":false,"bos_token_id":0,"eos_token_id":[1,2],"pad_token_id":0}`
	if err := os.WriteFile(filepath.Join(root, "config.json"), []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "tokenizer.json"), []byte(`{"version":"1.0"}`), 0600); err != nil {
		t.Fatal(err)
	}
	shapes := map[string][]int{"model.embed_tokens.weight": {3, 4}, "model.norm.weight": {4}, "model.layers.0.input_layernorm.weight": {4}, "model.layers.0.post_attention_layernorm.weight": {4}}
	for name, shape := range map[string][]int{"self_attn.q_proj.weight": {4, 4}, "self_attn.q_proj.bias": {4}, "self_attn.k_proj.weight": {2, 4}, "self_attn.k_proj.bias": {2}, "self_attn.v_proj.weight": {2, 4}, "self_attn.v_proj.bias": {2}, "self_attn.o_proj.weight": {4, 4}, "mlp.gate_proj.weight": {5, 4}, "mlp.up_proj.weight": {5, 4}, "mlp.down_proj.weight": {4, 5}} {
		shapes["model.layers.0."+name] = shape
	}
	header := map[string]any{}
	offset := 0
	for name, shape := range shapes {
		size := 4
		for _, d := range shape {
			size *= d
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{offset, offset + size}}
		offset += size
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(raw)%8 != 0 {
		raw = append(raw, ' ')
	}
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(raw)))
	data = append(data, raw...)
	data = append(data, make([]byte, offset)...)
	if err := os.WriteFile(filepath.Join(root, "model.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestPlanAndDurableSecret(t *testing.T) {
	root := fixture(t)
	p, err := BuildPlan(root, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Nodes) != 12 || p.Contract.PhysicalHidden != 8 || !p.Contract.SourceTied {
		t.Fatal("incomplete dense plan")
	}
	for _, n := range p.Nodes {
		if n.Output == "head.weight" && (n.Source != "model.embed_tokens.weight" || n.Norm != "model.norm.weight") {
			t.Fatal("tied source/final norm folding lost")
		}
	}
	b, err := model.BindSource(root)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newSecret(p, b)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := SaveSecret(dir, s); err != nil {
		t.Fatal(err)
	}
	if err := SaveSecret(dir, s); err == nil {
		t.Fatal("overwrote dense secret")
	}
	stored, err := LoadSecret(dir)
	if err != nil || stored.disk.BindingID != s.disk.BindingID {
		t.Fatal("secret round trip failed", err)
	}
	if _, err := json.Marshal(s); err == nil {
		t.Fatal("secret exported through normal JSON")
	}
	mapped, err := stored.MapIDs([]int{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	restored, err := stored.RestoreIDs(mapped)
	if err != nil || digest(restored) != digest([]int{0, 1, 2}) {
		t.Fatal("client mapping direction", err)
	}
	physical := make([]float64, 3)
	for original, mapped := range stored.disk.Permutation {
		physical[mapped] = float64(original) + 0.25
	}
	logits, err := stored.RestoreLogits(physical)
	if err != nil || digest(logits) != digest([]float64{0.25, 1.25, 2.25}) {
		t.Fatal("client logit coordinates", err)
	}
	if _, err := stored.MapIDs([]int{3}); err == nil {
		t.Fatal("invalid client ID admitted")
	}
	if _, err := stored.RestoreIDs(make([]int, 4097)); err == nil {
		t.Fatal("oversized client IDs admitted")
	}
	if _, err := stored.RestoreLogits([]float64{0}); err == nil {
		t.Fatal("wrong logit vocabulary admitted")
	}
	bad := stored.disk
	bad.Permutation[0] = bad.Permutation[1]
	bad.BindingID = commitment(bad)
	if _, err := validateSecret(bad); err == nil {
		t.Fatal("accepted committed corrupt permutation")
	}
}

func TestAdapterRejectsSemanticVariants(t *testing.T) {
	for _, variant := range []string{"sliding", "activation", "partial-rope", "head-count", "special", "expansion", "alias"} {
		t.Run(variant, func(t *testing.T) {
			root := fixture(t)
			name := filepath.Join(root, "config.json")
			raw, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			text := string(raw)
			expansion := 4
			switch variant {
			case "sliding":
				text = strings.Replace(text, `"use_sliding_window":false`, `"use_sliding_window":true`, 1)
			case "activation":
				text = strings.Replace(text, `"silu"`, `"gelu"`, 1)
			case "partial-rope":
				text = strings.TrimSuffix(text, "}") + `,"partial_rotary_factor":0.5}`
			case "head-count":
				text = strings.Replace(text, `"num_attention_heads":2`, `"num_attention_heads":3`, 1)
			case "special":
				text = strings.Replace(text, `"eos_token_id":[1,2]`, `"eos_token_id":[1,1]`, 1)
			case "expansion":
				expansion = 2
			case "alias":
				text = strings.Replace(text, `"tie_word_embeddings":true`, `"tie_word_embeddings":false`, 1)
			}
			if err := os.WriteFile(name, []byte(text), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := BuildPlan(root, expansion); err == nil {
				t.Fatal("accepted unsupported graph")
			}
		})
	}
}
