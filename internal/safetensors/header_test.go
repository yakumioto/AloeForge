package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func fixture(header string, payload int) []byte {
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	return append(data, make([]byte, payload)...)
}

func TestValidHeader(t *testing.T) {
	data := fixture(`{"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 8)
	header, err := Read(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if header.PayloadBytes != 8 || len(header.Tensors) != 2 || header.Tensors[0].Name != "a" {
		t.Fatal("inventory is not canonical")
	}
	for _, header := range []string{`{"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}`, `{"a":{"dtype":"F32","shape":[0,7],"data_offsets":[0,0]}}`} {
		payload := 4
		if bytes.Contains([]byte(header), []byte("[0,7]")) {
			payload = 0
		}
		data := fixture(header, payload)
		if _, err := Read(bytes.NewReader(data), int64(len(data))); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRejectMalformedHeaders(t *testing.T) {
	tests := []struct {
		name, header string
		payload      int
	}{
		{"duplicate", `{"a":{},"a":{}}`, 0},
		{"overlap", `{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}`, 4},
		{"hole", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, 2},
		{"trailing", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, 2},
		{"out of bounds", `{"a":{"dtype":"U8","shape":[5],"data_offsets":[0,5]}}`, 4},
		{"reversed", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}`, 1},
		{"negative offset", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}}`, 1},
		{"offset count", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1,1]}}`, 1},
		{"fractional offset", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1.0]}}`, 1},
		{"dtype", `{"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}`, 1},
		{"negative shape", `{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}}`, 0},
		{"null shape", `{"a":{"dtype":"U8","shape":null,"data_offsets":[0,0]}}`, 0},
		{"overflow", `{"a":{"dtype":"F64","shape":[9223372036854775807,2],"data_offsets":[0,0]}}`, 0},
		{"unknown field", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"private":"never-log"}}`, 1},
		{"metadata type", `{"__metadata__":{"x":2},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, 1},
		{"no tensors", `{"__metadata__":{"x":"y"}}`, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data := fixture(test.header, test.payload)
			if _, err := Read(bytes.NewReader(data), int64(len(data))); err == nil {
				t.Fatal("malformed header accepted")
			}
		})
	}
	for _, length := range []uint64{0, 1, MaxHeaderBytes + 1, math.MaxUint64} {
		data := make([]byte, 16)
		binary.LittleEndian.PutUint64(data, length)
		if _, err := Read(bytes.NewReader(data), 16); err == nil {
			t.Fatal("oversized length accepted")
		}
	}
	if _, err := ByteSize("F32", make([]int64, MaxRank+1)); err == nil {
		t.Fatal("rank limit bypassed")
	}
}

// A reader that fails for payload accesses proves inspection is header-only.
type headerOnly struct {
	*bytes.Reader
	end int64
}

func (reader headerOnly) ReadAt(buffer []byte, offset int64) (int, error) {
	if offset+int64(len(buffer)) > reader.end {
		panic("payload read during metadata inspection")
	}
	return reader.Reader.ReadAt(buffer, offset)
}

func TestDoesNotReadLargePayload(t *testing.T) {
	header := `{"a":{"dtype":"U8","shape":[1073741824],"data_offsets":[0,1073741824]}}`
	data := fixture(header, 0)
	if _, err := Read(headerOnly{bytes.NewReader(data), int64(len(data))}, int64(len(data))+(1<<30)); err != nil {
		t.Fatal(err)
	}
}

func FuzzHeader(f *testing.F) {
	f.Add(fixture(`{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, 1))
	f.Add([]byte{})
	f.Add([]byte{255, 255, 255, 255, 255, 255, 255, 255})
	f.Fuzz(func(t *testing.T, data []byte) {
		// Untrusted byte sequences must fail with diagnostics, never panic.
		Read(bytes.NewReader(data), int64(len(data)))
	})
}
