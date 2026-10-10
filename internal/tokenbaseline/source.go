package tokenbaseline

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"
	"slices"
	"unicode"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/strictjson"
	"golang.org/x/sys/unix"
)

type fixtureSpec struct {
	SchemaVersion  int              `json:"schema_version"`
	Kind           string           `json:"kind"`
	VocabularySize int              `json:"vocabulary_size"`
	HiddenSize     int              `json:"hidden_size"`
	TiedEmbeddings *bool            `json:"tied_embeddings"`
	SpecialIDs     map[string][]int `json:"special_token_ids"`
}

type sourceMatrix struct {
	file  *os.File
	start int64
}

type source struct {
	context  Context
	matrices map[string]sourceMatrix
}

func loadSource(path string) (*source, error) {
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, errors.New("cannot open token baseline source")
	}
	defer root.Close()
	data, err := model.ReadMetadata(root, "token-baseline.json")
	if err != nil || len(data) > 16384 {
		return nil, errors.New("invalid token baseline sidecar")
	}
	var spec fixtureSpec
	if err := strictjson.Decode(data, &spec, true); err != nil || spec.SchemaVersion != 1 || spec.Kind != "synthetic-embedding-head-v1" || spec.TiedEmbeddings == nil {
		return nil, errors.New("unsupported token baseline fixture")
	}
	if spec.VocabularySize < 2 || spec.VocabularySize > artifact.MaxDiagnosticVocab || spec.HiddenSize < 1 || spec.HiddenSize > artifact.MaxDiagnosticHidden || spec.VocabularySize > artifact.MaxDiagnosticElements/spec.HiddenSize || artifact.ValidateSpecialIDs(spec.SpecialIDs, spec.VocabularySize) != nil {
		return nil, errors.New("unsupported token baseline dimensions or IDs")
	}
	data, err = model.ReadMetadata(root, "tokenizer.json")
	if err != nil {
		return nil, errors.New("invalid synthetic tokenizer")
	}
	var tokenizer struct {
		SchemaVersion int      `json:"schema_version"`
		Kind          string   `json:"kind"`
		Tokens        []string `json:"tokens"`
	}
	if err := strictjson.Decode(data, &tokenizer, true); err != nil || tokenizer.SchemaVersion != 1 || tokenizer.Kind != "synthetic-vocabulary-v1" || len(tokenizer.Tokens) != spec.VocabularySize {
		return nil, errors.New("unsupported synthetic tokenizer")
	}
	seen := make(map[string]bool)
	for _, token := range tokenizer.Tokens {
		if len(token) == 0 || len(token) > 64 || seen[token] {
			return nil, errors.New("invalid synthetic vocabulary")
		}
		for _, r := range token {
			if unicode.IsControl(r) {
				return nil, errors.New("invalid synthetic vocabulary")
			}
		}
		seen[token] = true
	}
	sum := sha256.Sum256(data)
	inventory, err := model.Inspect(path)
	if err != nil {
		return nil, err
	}
	expected := 2
	if *spec.TiedEmbeddings {
		expected = 1
	}
	if len(inventory.Tensors) != expected {
		return nil, errors.New("unsupported token baseline tensor inventory")
	}
	for i, tensor := range inventory.Tensors {
		name := "embedding.weight"
		if i == 1 {
			name = "lm_head.weight"
		}
		if tensor.Name != name || tensor.DType != "F32" || !slices.Equal(tensor.Shape, []int64{int64(spec.VocabularySize), int64(spec.HiddenSize)}) {
			return nil, errors.New("unsupported token baseline matrix layout")
		}
	}
	// Only structurally admitted bounded matrices reach the whole-byte binder.
	binding, err := model.BindSource(path)
	if err != nil {
		return nil, err
	}
	if inventory.Hash() != binding.InventoryHash {
		return nil, errors.New("source inventory changed")
	}
	c := Context{Source: binding, TokenizerFingerprint: hex.EncodeToString(sum[:]), VocabularySize: spec.VocabularySize, HiddenSize: spec.HiddenSize, TiedEmbeddings: *spec.TiedEmbeddings, SpecialIDs: spec.SpecialIDs, MethodRevision: artifact.TokenMethod, RNGAlgorithm: artifact.TokenRNG, RNGRevision: artifact.TokenRNGRevision}
	if err := validateContext(c); err != nil {
		return nil, err
	}
	s := &source{context: c, matrices: map[string]sourceMatrix{}}
	fail := func() (*source, error) {
		s.close()
		return nil, errors.New("unsupported or changed token baseline matrix")
	}
	for _, tensor := range inventory.Tensors {
		name := tensor.Name
		var shard model.Shard
		for _, candidate := range inventory.Shards {
			if candidate.Name == tensor.Shard {
				shard = candidate
			}
		}
		file, err := root.OpenFile(tensor.Shard, os.O_RDONLY|unix.O_NONBLOCK, 0)
		if err != nil {
			return fail()
		}
		stat, err := file.Stat()
		if err != nil || !stat.Mode().IsRegular() || stat.Size() != shard.FileBytes {
			file.Close()
			return fail()
		}
		s.matrices[name] = sourceMatrix{file: file, start: shard.DataStart + tensor.Offsets[0]}
	}
	return s, nil
}

func (s *source) close() {
	for _, matrix := range s.matrices {
		matrix.file.Close()
	}
}

// rowReader keeps one bounded row. It emits physical rows by inverse lookup and
// validates F32 finiteness without loading the complete matrix into Go memory.
type rowReader struct {
	matrix   sourceMatrix
	inverse  []int
	row      []byte
	physical int
	offset   int
}

func (s *source) reader(name string, secret *Secret) io.Reader {
	return &rowReader{matrix: s.matrices[name], inverse: secret.inverse, row: make([]byte, s.context.HiddenSize*4)}
}

func (r *rowReader) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	if r.physical == len(r.inverse) {
		return 0, io.EOF
	}
	if r.offset == 0 {
		start := r.matrix.start + int64(r.inverse[r.physical])*int64(len(r.row))
		if n, err := r.matrix.file.ReadAt(r.row, start); err != nil || n != len(r.row) {
			return 0, errors.New("source matrix read failed")
		}
		for i := 0; i < len(r.row); i += 4 {
			value := float64(math.Float32frombits(binary.LittleEndian.Uint32(r.row[i : i+4])))
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return 0, errors.New("nonfinite source matrix")
			}
		}
	}
	n := copy(dst, r.row[r.offset:])
	r.offset += n
	if r.offset == len(r.row) {
		r.physical++
		r.offset = 0
	}
	return n, nil
}
