// Package tokenbaseline implements a diagnostic synthetic vocabulary baseline.
// It is not an encryption scheme, transformer adapter or remote runtime.
package tokenbaseline

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"math/big"
	"slices"
	"strings"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
)

const bindingDomain = "AloeForge/token-binding/v1\n"

// Context is private compiler/client metadata, never a public artifact field.
type Context struct {
	Source               model.SourceBinding `json:"source"`
	TokenizerFingerprint string              `json:"tokenizer_sha256"`
	VocabularySize       int                 `json:"vocabulary_size"`
	HiddenSize           int                 `json:"hidden_size"`
	TiedEmbeddings       bool                `json:"tied_embeddings"`
	SpecialIDs           map[string][]int    `json:"special_token_ids"`
	MethodRevision       string              `json:"method_revision"`
	RNGAlgorithm         string              `json:"rng_algorithm"`
	RNGRevision          string              `json:"rng_revision"`
}

type material struct {
	Context     Context `json:"context"`
	Nonce       string  `json:"nonce"`
	Permutation []int   `json:"permutation"`
}

type secretDisk struct {
	SchemaVersion int      `json:"schema_version"`
	BindingID     string   `json:"binding_id"`
	Material      material `json:"material"`
}

// Secret is immutable after validation. No secret material is returned by the
// public API or normal formatting/JSON; only the private store serializes it.
type Secret struct {
	disk    secretDisk
	inverse []int
}

func (*Secret) String() string   { return "[ClientSecret redacted]" }
func (*Secret) GoString() string { return "[ClientSecret redacted]" }
func (*Secret) MarshalJSON() ([]byte, error) {
	return nil, errors.New("Client Secret must use private storage")
}

func validHash(value string) bool {
	if len(value) != 64 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func cloneIDs(ids map[string][]int) map[string][]int {
	result := maps.Clone(ids)
	for role, values := range result {
		result[role] = slices.Clone(values)
	}
	return result
}

func validateContext(c Context) error {
	if !validHash(c.Source.InventoryHash) || !validHash(c.Source.ContentHash) || !validHash(c.TokenizerFingerprint) || c.VocabularySize < 2 || c.VocabularySize > artifact.MaxDiagnosticVocab || c.HiddenSize < 1 || c.HiddenSize > artifact.MaxDiagnosticHidden || c.VocabularySize > artifact.MaxDiagnosticElements/c.HiddenSize || c.MethodRevision != artifact.TokenMethod || c.RNGAlgorithm != artifact.TokenRNG || c.RNGRevision != artifact.TokenRNGRevision {
		return errors.New("unsupported private token context")
	}
	return artifact.ValidateSpecialIDs(c.SpecialIDs, c.VocabularySize)
}

func commitment(m material) string {
	encoded, _ := json.Marshal(m)
	sum := sha256.Sum256(append([]byte(bindingDomain), encoded...))
	return hex.EncodeToString(sum[:])
}

func validateSecret(disk secretDisk) (*Secret, error) {
	if disk.SchemaVersion != 1 || validateContext(disk.Material.Context) != nil || !validHash(disk.Material.Nonce) || !validHash(disk.BindingID) || len(disk.Material.Permutation) != disk.Material.Context.VocabularySize {
		return nil, errors.New("invalid Client Secret schema or material")
	}
	inverse := make([]int, len(disk.Material.Permutation))
	seen := make([]bool, len(inverse))
	for original, physical := range disk.Material.Permutation {
		if physical < 0 || physical >= len(inverse) || seen[physical] {
			return nil, errors.New("Client Secret permutation is not bijective")
		}
		seen[physical] = true
		inverse[physical] = original
	}
	if subtle.ConstantTimeCompare([]byte(commitment(disk.Material)), []byte(disk.BindingID)) != 1 {
		return nil, errors.New("Client Secret commitment mismatch")
	}
	disk.Material.Context.SpecialIDs = cloneIDs(disk.Material.Context.SpecialIDs)
	disk.Material.Permutation = slices.Clone(disk.Material.Permutation)
	return &Secret{disk: disk, inverse: inverse}, nil
}

// NewSecret uses unbiased crypto/rand Fisher-Yates. Saved secrets, not repeated
// seeds, determine recovery. It never silently regenerates a stored secret.
func NewSecret(context Context) (*Secret, error) { return generateSecret(context, rand.Reader) }

func generateSecret(context Context, entropy io.Reader) (*Secret, error) {
	if err := validateContext(context); err != nil {
		return nil, err
	}
	var nonce [32]byte
	if _, err := io.ReadFull(entropy, nonce[:]); err != nil {
		return nil, errors.New("Client Secret entropy failed")
	}
	p := make([]int, context.VocabularySize)
	for i := range p {
		p[i] = i
	}
	for i := len(p) - 1; i > 0; i-- {
		j, err := rand.Int(entropy, big.NewInt(int64(i+1)))
		if err != nil {
			return nil, errors.New("Client Secret entropy failed")
		}
		p[i], p[int(j.Int64())] = p[int(j.Int64())], p[i]
	}
	m := material{Context: context, Nonce: hex.EncodeToString(nonce[:]), Permutation: p}
	return validateSecret(secretDisk{SchemaVersion: 1, BindingID: commitment(m), Material: m})
}

func (secret *Secret) matches(context Context) bool {
	if secret == nil || validateContext(context) != nil {
		return false
	}
	a, _ := json.Marshal(secret.disk.Material.Context)
	b, _ := json.Marshal(context)
	return subtle.ConstantTimeCompare(a, b) == 1
}

func (secret *Secret) MapIDs(ids []int) ([]int, error)     { return secret.lookup(ids, false) }
func (secret *Secret) RestoreIDs(ids []int) ([]int, error) { return secret.lookup(ids, true) }

func (secret *Secret) lookup(ids []int, inverse bool) ([]int, error) {
	if secret == nil || len(ids) > 4096 {
		return nil, errors.New("invalid token mapping request")
	}
	mapping := secret.disk.Material.Permutation
	if inverse {
		mapping = secret.inverse
	}
	result := make([]int, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(mapping) {
			return nil, errors.New("token ID out of range")
		}
		result[i] = mapping[id]
	}
	return result, nil
}

// RestoreLogits returns original vocabulary coordinates before any comparison.
func (secret *Secret) RestoreLogits(physical []float64) ([]float64, error) {
	if secret == nil || len(physical) != len(secret.inverse) {
		return nil, errors.New("logit vocabulary mismatch")
	}
	result := make([]float64, len(physical))
	for original, mapped := range secret.disk.Material.Permutation {
		value := physical[mapped]
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, errors.New("nonfinite diagnostic logits")
		}
		result[original] = value
	}
	return result, nil
}

func (secret *Secret) publicBinding() *artifact.TokenBinding {
	c := secret.disk.Material.Context
	ids := cloneIDs(c.SpecialIDs)
	for role, values := range ids {
		for i, id := range values {
			values[i] = secret.disk.Material.Permutation[id]
		}
		slices.Sort(values)
		ids[role] = values
	}
	return &artifact.TokenBinding{SchemaVersion: 1, Profile: artifact.TokenProfile, BindingID: secret.disk.BindingID, TokenizerFingerprint: c.TokenizerFingerprint, MethodRevision: c.MethodRevision, RNGAlgorithm: c.RNGAlgorithm, RNGRevision: c.RNGRevision, VocabularySize: c.VocabularySize, HiddenSize: c.HiddenSize, TiedEmbeddings: c.TiedEmbeddings, MappedSpecialIDs: ids}
}
