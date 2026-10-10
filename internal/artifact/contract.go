// Package artifact implements diagnostic-only output staging and publication.
package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"sort"

	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/safetensors"
)

const Kind = "p1-diagnostic-only"

// OutputTensor is output-driven: no source name, shape, offset or private value.
type OutputTensor struct {
	Name        string  `json:"name"`
	Shard       string  `json:"shard"`
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	PayloadHash string  `json:"payload_sha256"`
}

// Plan is PRIVATE control metadata and is never included in published artifacts.
type Plan struct {
	SchemaVersion       int            `json:"schema_version"`
	SourceInventoryHash string         `json:"source_inventory_sha256"`
	SourceContentHash   string         `json:"source_content_sha256"`
	MethodRevision      string         `json:"method_revision"`
	RNGAlgorithm        string         `json:"rng_algorithm"`
	RNGRevision         string         `json:"rng_revision"`
	Outputs             []OutputTensor `json:"outputs"`
	Binding             *TokenBinding  `json:"token_binding,omitempty"`
}

type FileDigest struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// Manifest contains public output identity only; it is not a runtime contract.
type Manifest struct {
	SchemaVersion  int            `json:"schema_version"`
	Kind           string         `json:"kind"`
	DescriptorHash string         `json:"descriptor_sha256"`
	Outputs        []OutputTensor `json:"outputs"`
	Files          []FileDigest   `json:"files"`
	Binding        *TokenBinding  `json:"token_binding,omitempty"`
}

type checkpoint struct {
	SchemaVersion       int               `json:"schema_version"`
	PlanHash            string            `json:"plan_sha256"`
	SourceInventoryHash string            `json:"source_inventory_sha256"`
	SourceContentHash   string            `json:"source_content_sha256"`
	Completed           map[string]string `json:"completed"`
}

func hash(value any) string {
	data, _ := json.Marshal(value)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func validHash(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func identifier(value string) bool {
	if len(value) == 0 || len(value) > 80 {
		return false
	}
	for _, c := range value {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return value != "." && value != ".."
}

func normalizeOutputs(outputs []OutputTensor) ([]OutputTensor, error) {
	if len(outputs) == 0 || len(outputs) > model.MaxInventoryTensors {
		return nil, errors.New("invalid output descriptor count")
	}
	result := make([]OutputTensor, len(outputs))
	names := make(map[string]bool)
	shardBytes := make(map[string]int64)
	shardCounts := make(map[string]int)
	for i, tensor := range outputs {
		if !safetensors.ValidName(tensor.Name) || !model.ShardName(tensor.Shard) || !validHash(tensor.PayloadHash) || names[tensor.Name] {
			return nil, errors.New("invalid or duplicated output descriptor")
		}
		size, err := safetensors.ByteSize(tensor.DType, tensor.Shape)
		if err != nil || shardBytes[tensor.Shard] > math.MaxInt64-safetensors.MaxHeaderBytes-8-size {
			return nil, errors.New("output size exceeds bounds")
		}
		names[tensor.Name] = true
		shardBytes[tensor.Shard] += size
		shardCounts[tensor.Shard]++
		if shardCounts[tensor.Shard] > safetensors.MaxTensors || len(shardBytes) > model.MaxShards {
			return nil, errors.New("output shard count exceeds bounds")
		}
		result[i] = tensor
		result[i].Shape = slices.Clone(tensor.Shape)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Shard == result[j].Shard {
			return result[i].Name < result[j].Name
		}
		return result[i].Shard < result[j].Shard
	})
	data, err := json.Marshal(result)
	if err != nil || len(data) > model.MaxMetadataBytes-(64<<10) {
		return nil, errors.New("output descriptor metadata exceeds bounds")
	}
	for _, tensors := range groups(result) {
		if _, err := makeHeader(tensors); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func normalizePlan(plan Plan) (Plan, error) {
	if plan.SchemaVersion != 1 || !validHash(plan.SourceInventoryHash) || !validHash(plan.SourceContentHash) || !identifier(plan.MethodRevision) || !identifier(plan.RNGAlgorithm) || !identifier(plan.RNGRevision) {
		return Plan{}, errors.New("invalid plan revision or source binding")
	}
	outputs, err := normalizeOutputs(plan.Outputs)
	if err != nil {
		return Plan{}, err
	}
	plan.Outputs = outputs
	plan.Binding, err = normalizeTokenBinding(plan.Binding, outputs)
	if err != nil {
		return Plan{}, err
	}
	if plan.Binding != nil && (plan.MethodRevision != plan.Binding.MethodRevision || plan.RNGAlgorithm != plan.Binding.RNGAlgorithm || plan.RNGRevision != plan.Binding.RNGRevision) {
		return Plan{}, errors.New("plan and token binding revisions differ")
	}
	return plan, nil
}

// DescriptorHash is order-independent and binds output name/dtype/shape/content.
func DescriptorHash(outputs []OutputTensor) (string, error) {
	normalized, err := normalizeOutputs(outputs)
	if err != nil {
		return "", err
	}
	return hash(normalized), nil
}

func groups(outputs []OutputTensor) map[string][]OutputTensor {
	result := make(map[string][]OutputTensor)
	for _, tensor := range outputs {
		result[tensor.Shard] = append(result[tensor.Shard], tensor)
	}
	return result
}

func shardNames(outputs []OutputTensor) []string {
	result := make([]string, 0)
	for name := range groups(outputs) {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func makeHeader(tensors []OutputTensor) ([]byte, error) {
	entries := make(map[string]any)
	offset := int64(0)
	for _, tensor := range tensors {
		size, err := safetensors.ByteSize(tensor.DType, tensor.Shape)
		if err != nil {
			return nil, err
		}
		entries[tensor.Name] = struct {
			DType   string   `json:"dtype"`
			Shape   []int64  `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}{tensor.DType, tensor.Shape, [2]int64{offset, offset + size}}
		offset += size
	}
	data, err := json.Marshal(entries)
	if err != nil {
		return nil, errors.New("cannot encode output header")
	}
	for len(data)%8 != 0 {
		data = append(data, ' ')
	}
	if len(data) > safetensors.MaxHeaderBytes {
		return nil, errors.New("output header exceeds bounds")
	}
	return data, nil
}
