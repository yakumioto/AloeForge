// Package safetensors validates metadata without reading tensor payloads.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/yakumioto/AloeForge/internal/strictjson"
)

const (
	MaxHeaderBytes = 8 << 20
	MaxTensors     = 32768
	MaxRank        = 16
)

type Tensor struct {
	Name    string   `json:"name"`
	DType   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

type Header struct {
	DataStart    int64    `json:"data_start"`
	PayloadBytes int64    `json:"payload_bytes"`
	Tensors      []Tensor `json:"tensors"`
}

// ValidName bounds diagnostic metadata and rejects reserved/control names.
func ValidName(name string) bool {
	return name != "" && name != "__metadata__" && len(name) <= 512 && utf8.ValidString(name) && strings.IndexFunc(name, unicode.IsControl) < 0
}

// ByteSize rejects unsupported dtypes, invalid dimensions and integer overflow.
func ByteSize(dtype string, shape []int64) (int64, error) {
	widths := map[string]int64{"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
	width, ok := widths[dtype]
	if !ok || shape == nil || len(shape) > MaxRank {
		return 0, errors.New("unsupported dtype or shape")
	}
	count := int64(1)
	zero := false
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, errors.New("negative tensor dimension")
		}
		if dimension == 0 {
			zero = true
		}
	}
	if zero {
		return 0, nil
	}
	for _, dimension := range shape {
		if dimension > math.MaxInt64/count {
			return 0, errors.New("tensor size overflow")
		}
		count *= dimension
	}
	if count > math.MaxInt64/width {
		return 0, errors.New("tensor byte size overflow")
	}
	return count * width, nil
}

// Read validates complete, nonoverlapping payload coverage using only the header.
func Read(reader io.ReaderAt, fileBytes int64) (Header, error) {
	var result Header
	if fileBytes < 10 {
		return result, errors.New("truncated safetensors file")
	}
	var prefix [8]byte
	if _, err := reader.ReadAt(prefix[:], 0); err != nil {
		return result, errors.New("cannot read safetensors length")
	}
	length := binary.LittleEndian.Uint64(prefix[:])
	if length < 2 || length > MaxHeaderBytes || length > uint64(fileBytes-8) {
		return result, errors.New("safetensors header length exceeds bounds")
	}
	data := make([]byte, int(length))
	if _, err := reader.ReadAt(data, 8); err != nil {
		return result, errors.New("truncated safetensors header")
	}
	if data[0] != '{' {
		return result, errors.New("safetensors header must be an object")
	}
	var entries map[string]json.RawMessage
	if err := strictjson.Decode(data, &entries, false); err != nil {
		return result, err
	}
	if len(entries) == 0 || len(entries) > MaxTensors+1 {
		return result, errors.New("tensor count exceeds bounds")
	}
	result.DataStart = 8 + int64(length)
	result.PayloadBytes = fileBytes - result.DataStart
	for name, raw := range entries {
		if name == "__metadata__" {
			var metadata map[string]any
			if err := strictjson.Decode(raw, &metadata, false); err != nil || metadata == nil {
				return result, errors.New("invalid safetensors metadata")
			}
			for _, value := range metadata {
				if _, ok := value.(string); !ok {
					return result, errors.New("invalid safetensors metadata value")
				}
			}
			continue
		}
		if !ValidName(name) {
			return result, errors.New("invalid tensor name")
		}
		var descriptor struct {
			DType   string  `json:"dtype"`
			Shape   []int64 `json:"shape"`
			Offsets []int64 `json:"data_offsets"`
		}
		if err := strictjson.Decode(raw, &descriptor, true); err != nil {
			return result, err
		}
		size, err := ByteSize(descriptor.DType, descriptor.Shape)
		if err != nil {
			return result, err
		}
		if len(descriptor.Offsets) != 2 {
			return result, errors.New("tensor offsets must have length two")
		}
		start, end := descriptor.Offsets[0], descriptor.Offsets[1]
		if start < 0 || end < start || end > result.PayloadBytes || end-start != size {
			return result, errors.New("tensor offsets, shape or payload bounds disagree")
		}
		result.Tensors = append(result.Tensors, Tensor{Name: name, DType: descriptor.DType, Shape: descriptor.Shape, Offsets: [2]int64{start, end}})
	}
	if len(result.Tensors) == 0 || len(result.Tensors) > MaxTensors {
		return result, errors.New("invalid tensor count")
	}
	byOffset := append([]Tensor(nil), result.Tensors...)
	sort.Slice(byOffset, func(i, j int) bool {
		if byOffset[i].Offsets[0] == byOffset[j].Offsets[0] {
			return byOffset[i].Offsets[1] < byOffset[j].Offsets[1]
		}
		return byOffset[i].Offsets[0] < byOffset[j].Offsets[0]
	})
	next := int64(0)
	for _, tensor := range byOffset {
		if tensor.Offsets[0] != next {
			return result, errors.New("tensor payload has overlaps or holes")
		}
		next = tensor.Offsets[1]
	}
	if next != result.PayloadBytes {
		return result, errors.New("unclaimed trailing tensor payload")
	}
	sort.Slice(result.Tensors, func(i, j int) bool { return result.Tensors[i].Name < result.Tensors[j].Name })
	return result, nil
}
