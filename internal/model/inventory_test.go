package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

const config = `{"model_type":"llama","architectures":["LlamaForCausalLM"]}`

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

func writeShard(t *testing.T, root, name, tensor string) {
	t.Helper()
	header := `{"` + tensor + `":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 0, 0, 0, 0)
	write(t, filepath.Join(root, name), data)
}

func sharded(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write(t, filepath.Join(root, "config.json"), []byte(config))
	writeShard(t, root, "shard-1.safetensors", "a")
	writeShard(t, root, "shard-2.safetensors", "b")
	write(t, filepath.Join(root, "model.safetensors.index.json"), []byte(`{"metadata":{"total_size":8},"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}`))
	return root
}

func TestInventoryAndStableHash(t *testing.T) {
	root := sharded(t)
	first, err := Inspect(root)
	if err != nil {
		t.Fatal(err)
	}
	if first.SchemaVersion != 1 || first.Architecture != "llama" || len(first.Shards) != 2 || len(first.Tensors) != 2 || first.Tensors[0].Name != "a" {
		t.Fatal("wrong inventory")
	}
	write(t, filepath.Join(root, "config.json"), []byte(`{ "architectures" : ["LlamaForCausalLM"], "model_type":"llama" }`))
	write(t, filepath.Join(root, "model.safetensors.index.json"), []byte(`{"weight_map":{"b":"shard-2.safetensors","a":"shard-1.safetensors"},"metadata":{"total_size":8}}`))
	second, err := Inspect(root)
	if err != nil || first.Hash() != second.Hash() {
		t.Fatal("inventory hash changed with JSON field order", err)
	}
	write(t, filepath.Join(root, "config.json"), []byte(`{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":16}`))
	third, err := Inspect(root)
	if err != nil || third.Hash() == first.Hash() {
		t.Fatal("config changes did not bind inventory")
	}
}

func TestRejectMalformedInventory(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, string)
	}{
		{"unknown architecture", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"unknown","architectures":["UnknownForCausalLM"]}`))
		}},
		{"ambiguous architecture", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"llama","architectures":["LlamaForCausalLM","Other"]}`))
		}},
		{"mismatched architecture", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"qwen2","architectures":["LlamaForCausalLM"]}`))
		}},
		{"duplicate config", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"llama","model_type":"qwen2","architectures":["LlamaForCausalLM"]}`))
		}},
		{"nested duplicate", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"llama","architectures":["LlamaForCausalLM"],"extra":{"x":1,"x":2}}`))
		}},
		{"quantization", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"llama","architectures":["LlamaForCausalLM"],"quantization_config":{}}`))
		}},
		{"rotary variant", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "config.json"), []byte(`{"model_type":"llama","architectures":["LlamaForCausalLM"],"rope_scaling":{}}`))
		}},
		{"missing index", func(t *testing.T, r string) {
			if err := os.Remove(filepath.Join(r, "model.safetensors.index.json")); err != nil {
				t.Fatal(err)
			}
		}},
		{"missing shard", func(t *testing.T, r string) {
			if err := os.Remove(filepath.Join(r, "shard-2.safetensors")); err != nil {
				t.Fatal(err)
			}
		}},
		{"unreferenced shard", func(t *testing.T, r string) { writeShard(t, r, "extra.safetensors", "extra") }},
		{"duplicated tensor", func(t *testing.T, r string) { writeShard(t, r, "shard-2.safetensors", "a") }},
		{"index total mismatch", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"metadata":{"total_size":9},"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}`))
		}},
		{"index total type", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"metadata":{"total_size":null},"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}`))
		}},
		{"index contradiction", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"shard-2.safetensors","b":"shard-1.safetensors"}}`))
		}},
		{"index missing tensor", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors","c":"shard-1.safetensors"}}`))
		}},
		{"index duplicate", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"shard-1.safetensors","a":"shard-2.safetensors"}}`))
		}},
		{"traversal", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"../shard-1.safetensors","b":"shard-2.safetensors"}}`))
		}},
		{"absolute", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"/shard-1.safetensors","b":"shard-2.safetensors"}}`))
		}},
		{"backslash", func(t *testing.T, r string) {
			write(t, filepath.Join(r, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"..\\shard-1.safetensors","b":"shard-2.safetensors"}}`))
		}},
		{"oversized metadata", func(t *testing.T, r string) {
			f, err := os.Create(filepath.Join(r, "config.json"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if err := f.Truncate(MaxMetadataBytes + 1); err != nil {
				t.Fatal(err)
			}
		}},
		{"nonregular shard", func(t *testing.T, r string) {
			p := filepath.Join(r, "shard-1.safetensors")
			if err := os.Remove(p); err != nil {
				t.Fatal(err)
			}
			if err := unix.Mkfifo(p, 0600); err != nil {
				t.Fatal(err)
			}
		}},
		{"nonregular config", func(t *testing.T, r string) {
			p := filepath.Join(r, "config.json")
			if err := os.Remove(p); err != nil {
				t.Fatal(err)
			}
			if err := unix.Mkfifo(p, 0600); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := sharded(t)
			test.mutate(t, root)
			if _, err := Inspect(root); err == nil {
				t.Fatal("malformed inventory accepted")
			}
		})
	}
}

func TestSymlinkConfinement(t *testing.T) {
	for _, name := range []string{"config.json", "model.safetensors.index.json", "shard-1.safetensors"} {
		t.Run(name, func(t *testing.T) {
			root := sharded(t)
			outside := t.TempDir()
			data, err := os.ReadFile(filepath.Join(root, name))
			if err != nil {
				t.Fatal(err)
			}
			write(t, filepath.Join(outside, name), data)
			if err := os.Remove(filepath.Join(root, name)); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(filepath.Join(outside, name), filepath.Join(root, name)); err != nil {
				t.Fatal(err)
			}
			if _, err := Inspect(root); err == nil {
				t.Fatal("escaping symlink accepted")
			}
		})
	}
	root := t.TempDir()
	write(t, filepath.Join(root, "config.json"), []byte(config))
	writeShard(t, root, "payload.dat", "a")
	if err := os.Symlink("payload.dat", filepath.Join(root, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := Inspect(root); err != nil {
		t.Fatal("confined relative symlink rejected", err)
	}
}

func TestSingleFileAndQwenIdentity(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "config.json"), []byte(`{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}`))
	writeShard(t, root, "model.safetensors", "a")
	result, err := Inspect(root)
	if err != nil || result.Architecture != "qwen2" {
		t.Fatal(err)
	}
	data, _ := json.Marshal(result)
	if !json.Valid(data) {
		t.Fatal("inventory cannot serialize")
	}
}

func TestSourceBindingDetectsPayloadChanges(t *testing.T) {
	root := sharded(t)
	first, err := BindSource(root)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(filepath.Join(root, "shard-1.safetensors"), os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte{1}, stat.Size()-1); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := BindSource(root)
	if err != nil {
		t.Fatal(err)
	}
	if first.InventoryHash != second.InventoryHash || first.ContentHash == second.ContentHash {
		t.Fatal("payload change not bound independently of header")
	}
}

func TestShardCountBound(t *testing.T) {
	root := sharded(t)
	for i := 0; i < MaxShards; i++ {
		writeShard(t, root, fmt.Sprintf("extra-%03d.safetensors", i), fmt.Sprintf("extra_%03d", i))
	}
	if _, err := Inspect(root); err == nil {
		t.Fatal("shard limit bypassed")
	}
}
