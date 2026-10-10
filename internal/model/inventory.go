// Package model discovers and validates local HF safetensors inventories.
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/yakumioto/AloeForge/internal/safetensors"
	"github.com/yakumioto/AloeForge/internal/strictjson"
	"golang.org/x/sys/unix"
)

const (
	MaxMetadataBytes    = 8 << 20
	MaxShards           = 128
	MaxInventoryTensors = 100000
)

type Tensor struct {
	safetensors.Tensor
	Shard string `json:"shard"`
}

type Shard struct {
	Name         string `json:"name"`
	FileBytes    int64  `json:"file_bytes"`
	DataStart    int64  `json:"data_start"`
	PayloadBytes int64  `json:"payload_bytes"`
}

// ModelInventory is structural evidence, not an architecture adapter capability.
type ModelInventory struct {
	SchemaVersion int      `json:"schema_version"`
	Architecture  string   `json:"architecture"`
	ConfigHash    string   `json:"config_sha256"`
	Shards        []Shard  `json:"shards"`
	Tensors       []Tensor `json:"tensors"`
}

func (inventory ModelInventory) Hash() string {
	data, _ := json.Marshal(inventory)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// ShardName restricts shards to portable basenames, never paths.
func ShardName(name string) bool {
	if len(name) == 0 || len(name) > 200 || !strings.HasSuffix(name, ".safetensors") || name[0] == '.' {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func ReadMetadata(root *os.Root, name string) ([]byte, error) {
	file, err := root.OpenFile(name, os.O_RDONLY|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, errors.New("cannot open confined metadata file")
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Size() > MaxMetadataBytes {
		return nil, errors.New("metadata file type or length invalid")
	}
	data, err := io.ReadAll(io.LimitReader(file, MaxMetadataBytes+1))
	if err != nil || len(data) > MaxMetadataBytes {
		return nil, errors.New("metadata read exceeds bounds")
	}
	return data, nil
}

func Inspect(path string) (ModelInventory, error) {
	var inventory ModelInventory
	root, err := os.OpenRoot(path)
	if err != nil {
		return inventory, errors.New("cannot open model root")
	}
	defer root.Close()
	config, err := ReadMetadata(root, "config.json")
	if err != nil {
		return inventory, err
	}
	var configObject map[string]any
	if err := strictjson.Decode(config, &configObject, false); err != nil || configObject == nil {
		return inventory, errors.New("invalid model config")
	}
	modelType, _ := configObject["model_type"].(string)
	architectures, _ := configObject["architectures"].([]any)
	expected := map[string]string{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM"}
	if expected[modelType] == "" || len(architectures) != 1 || architectures[0] != expected[modelType] {
		return inventory, errors.New("unsupported or ambiguous architecture identity")
	}
	for _, field := range []string{"quantization_config", "auto_map", "rope_scaling"} {
		if configObject[field] != nil {
			return inventory, errors.New("unsupported model configuration variant")
		}
	}
	canonical, err := json.Marshal(configObject)
	if err != nil {
		return inventory, errors.New("cannot canonicalize config")
	}
	configSum := sha256.Sum256(canonical)
	inventory = ModelInventory{SchemaVersion: 1, Architecture: modelType, ConfigHash: hex.EncodeToString(configSum[:])}
	directory, err := root.Open(".")
	if err != nil {
		return inventory, errors.New("cannot list model root")
	}
	entries, err := directory.ReadDir(4097)
	directory.Close()
	if err != nil && err != io.EOF || len(entries) > 4096 {
		return inventory, errors.New("model directory listing exceeds bounds")
	}
	discovered := make(map[string]bool)
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".safetensors") {
			if !ShardName(entry.Name()) {
				return inventory, errors.New("invalid discovered shard name")
			}
			discovered[entry.Name()] = true
		}
	}
	if len(discovered) == 0 || len(discovered) > MaxShards {
		return inventory, errors.New("invalid shard count")
	}
	var weightMap map[string]string
	var declaredSize *int64
	_, indexErr := root.Stat("model.safetensors.index.json")
	if indexErr == nil {
		index, err := ReadMetadata(root, "model.safetensors.index.json")
		if err != nil {
			return inventory, err
		}
		var document struct {
			Metadata  map[string]any    `json:"metadata"`
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := strictjson.Decode(index, &document, true); err != nil {
			return inventory, err
		}
		weightMap = document.WeightMap
		if rawSize, ok := document.Metadata["total_size"]; ok {
			number, ok := rawSize.(json.Number)
			if !ok {
				return inventory, errors.New("invalid index total size")
			}
			size, err := number.Int64()
			if err != nil || size < 0 {
				return inventory, errors.New("invalid index total size")
			}
			declaredSize = &size
		}
		if len(weightMap) == 0 || len(weightMap) > MaxInventoryTensors {
			return inventory, errors.New("invalid index tensor count")
		}
		referenced := make(map[string]bool)
		for name, shard := range weightMap {
			if !safetensors.ValidName(name) || !ShardName(shard) || !discovered[shard] {
				return inventory, errors.New("index name, path or shard completeness invalid")
			}
			referenced[shard] = true
		}
		if len(referenced) != len(discovered) {
			return inventory, errors.New("unreferenced shard or contradictory discovery")
		}
	} else if !os.IsNotExist(indexErr) {
		return inventory, errors.New("cannot inspect model index")
	} else if len(discovered) != 1 || !discovered["model.safetensors"] {
		return inventory, errors.New("sharded model requires an index")
	}
	names := make([]string, 0, len(discovered))
	for name := range discovered {
		names = append(names, name)
	}
	sort.Strings(names)
	seen := make(map[string]bool)
	totalBytes := int64(0)
	for _, name := range names {
		file, err := root.OpenFile(name, os.O_RDONLY|unix.O_NONBLOCK, 0)
		if err != nil {
			return inventory, errors.New("cannot open confined shard")
		}
		stat, err := file.Stat()
		if err != nil || !stat.Mode().IsRegular() {
			file.Close()
			return inventory, errors.New("shard must be a regular file")
		}
		header, err := safetensors.Read(file, stat.Size())
		file.Close()
		if err != nil {
			return inventory, err
		}
		if len(inventory.Tensors)+len(header.Tensors) > MaxInventoryTensors {
			return inventory, errors.New("inventory tensor count exceeds bounds")
		}
		inventory.Shards = append(inventory.Shards, Shard{Name: name, FileBytes: stat.Size(), DataStart: header.DataStart, PayloadBytes: header.PayloadBytes})
		for _, tensor := range header.Tensors {
			if seen[tensor.Name] || weightMap != nil && weightMap[tensor.Name] != name {
				return inventory, errors.New("duplicated tensor or contradictory index entry")
			}
			seen[tensor.Name] = true
			inventory.Tensors = append(inventory.Tensors, Tensor{Tensor: tensor, Shard: name})
		}
		if totalBytes > math.MaxInt64-header.PayloadBytes {
			return inventory, errors.New("aggregate payload size overflow")
		}
		totalBytes += header.PayloadBytes
	}
	if weightMap != nil && len(seen) != len(weightMap) {
		return inventory, errors.New("index lists missing tensors")
	}
	if declaredSize != nil && *declaredSize != totalBytes {
		return inventory, errors.New("index total size contradicts payload inventory")
	}
	sort.Slice(inventory.Tensors, func(i, j int) bool { return inventory.Tensors[i].Name < inventory.Tensors[j].Name })
	return inventory, nil
}
