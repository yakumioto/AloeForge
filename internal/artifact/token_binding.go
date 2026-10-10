package artifact

import (
	"encoding/binary"
	"errors"
	"io"
	"maps"
	"math"
	"os"
	"slices"
)

const (
	TokenKind             = "p2-token-permutation-diagnostic-only"
	TokenProfile          = "token-permutation-diagnostic"
	TokenMethod           = "token-permutation-v1"
	TokenRNG              = "crypto-rand-fisher-yates"
	TokenRNGRevision      = "1"
	MaxDiagnosticVocab    = 65536
	MaxDiagnosticHidden   = 256
	MaxDiagnosticElements = 1 << 20
)

// TokenBinding has public identity only. It contains no permutation, nonce,
// original IDs or source fingerprints. It is not a P3 runtime/wire contract.
type TokenBinding struct {
	SchemaVersion        int              `json:"schema_version"`
	Profile              string           `json:"profile"`
	BindingID            string           `json:"binding_id"`
	TokenizerFingerprint string           `json:"tokenizer_sha256"`
	MethodRevision       string           `json:"method_revision"`
	RNGAlgorithm         string           `json:"rng_algorithm"`
	RNGRevision          string           `json:"rng_revision"`
	VocabularySize       int              `json:"vocabulary_size"`
	HiddenSize           int              `json:"hidden_size"`
	TiedEmbeddings       bool             `json:"tied_embeddings"`
	MappedSpecialIDs     map[string][]int `json:"mapped_special_token_ids"`
}

// P2 admits only finite F32 diagnostic matrices. This extra gate also applies
// when a caller uses the lower-level artifact writer rather than the compiler.
func verifyFiniteTokenShard(root *os.Root, name string, tensors []OutputTensor) error {
	header, err := makeHeader(tensors)
	if err != nil {
		return err
	}
	file, err := openRegular(root, name)
	if err != nil {
		return err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return errors.New("cannot stat token diagnostic shard")
	}
	reader := io.NewSectionReader(file, int64(8+len(header)), stat.Size()-int64(8+len(header)))
	var buffer [32 << 10]byte
	for {
		n, err := io.ReadFull(reader, buffer[:])
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF || n%4 != 0 {
			return errors.New("token diagnostic payload read failed")
		}
		for i := 0; i < n; i += 4 {
			value := float64(math.Float32frombits(binary.LittleEndian.Uint32(buffer[i : i+4])))
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return errors.New("nonfinite token diagnostic payload")
			}
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil
		}
	}
}

// ValidateSpecialIDs accepts exactly three roles with sorted unique in-range
// values. Shared BOS/EOS/PAD IDs across roles are deliberate and allowed.
func ValidateSpecialIDs(ids map[string][]int, vocab int) error {
	if len(ids) != 3 {
		return errors.New("invalid special token roles")
	}
	for _, role := range []string{"bos", "eos", "pad"} {
		values, ok := ids[role]
		if !ok || values == nil || len(values) > 16 || role != "eos" && len(values) > 1 || role == "eos" && len(values) == 0 {
			return errors.New("invalid special token count")
		}
		for i, value := range values {
			if value < 0 || value >= vocab || i > 0 && value <= values[i-1] {
				return errors.New("invalid special token IDs")
			}
		}
	}
	return nil
}

func normalizeTokenBinding(binding *TokenBinding, outputs []OutputTensor) (*TokenBinding, error) {
	if binding == nil {
		return nil, nil
	}
	b := *binding
	if b.SchemaVersion != 1 || b.Profile != TokenProfile || !validHash(b.BindingID) || !validHash(b.TokenizerFingerprint) || b.MethodRevision != TokenMethod || b.RNGAlgorithm != TokenRNG || b.RNGRevision != TokenRNGRevision || b.VocabularySize < 2 || b.VocabularySize > MaxDiagnosticVocab || b.HiddenSize < 1 || b.HiddenSize > MaxDiagnosticHidden || b.VocabularySize > MaxDiagnosticElements/b.HiddenSize {
		return nil, errors.New("unsupported token diagnostic binding")
	}
	if err := ValidateSpecialIDs(b.MappedSpecialIDs, b.VocabularySize); err != nil {
		return nil, err
	}
	b.MappedSpecialIDs = maps.Clone(b.MappedSpecialIDs)
	for role, ids := range b.MappedSpecialIDs {
		b.MappedSpecialIDs[role] = slices.Clone(ids)
	}
	expected := 2
	if b.TiedEmbeddings {
		expected = 1
	}
	if len(outputs) != expected {
		return nil, errors.New("token diagnostic tensor completeness mismatch")
	}
	for i, output := range outputs {
		name := "embedding.weight"
		if i == 1 {
			name = "lm_head.weight"
		}
		if output.Name != name || output.Shard != "model.safetensors" || output.DType != "F32" || !slices.Equal(output.Shape, []int64{int64(b.VocabularySize), int64(b.HiddenSize)}) {
			return nil, errors.New("unsupported token diagnostic output layout")
		}
	}
	return &b, nil
}
