package dense

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"math/big"

	"github.com/yakumioto/AloeForge/internal/model"
)

type secretDisk struct {
	SchemaVersion int    `json:"schema_version"`
	ContextHash   string `json:"context_sha256"`
	Seed          string `json:"seed"`
	Permutation   []int  `json:"permutation"`
	BindingID     string `json:"binding_id"`
}

type Secret struct {
	disk    secretDisk
	inverse []int
}

func (*Secret) String() string   { return "[DenseClientSecret redacted]" }
func (*Secret) GoString() string { return "[DenseClientSecret redacted]" }
func (*Secret) MarshalJSON() ([]byte, error) {
	return nil, errors.New("dense secret requires private storage")
}

func digest(value any) string {
	raw, _ := json.Marshal(value)
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
func commitment(d secretDisk) string {
	d.BindingID = ""
	raw, _ := json.Marshal(d)
	sum := sha256.Sum256(append([]byte("AloeForge/dense-secret/v1\n"), raw...))
	return hex.EncodeToString(sum[:])
}
func validHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func validateSecret(d secretDisk) (*Secret, error) {
	if d.SchemaVersion != 1 || !validHash(d.ContextHash) || !validHash(d.Seed) || !validHash(d.BindingID) || len(d.Permutation) < 2 || len(d.Permutation) > 262144 || commitment(d) != d.BindingID {
		return nil, errors.New("invalid dense Client Secret")
	}
	seen := make([]bool, len(d.Permutation))
	inverse := make([]int, len(d.Permutation))
	for original, id := range d.Permutation {
		if id < 0 || id >= len(seen) || seen[id] {
			return nil, errors.New("invalid dense token permutation")
		}
		seen[id] = true
		inverse[id] = original
	}
	return &Secret{disk: d, inverse: inverse}, nil
}

// Mapping APIs are trusted-client-only; no private maps are exposed or logged.
func (s *Secret) MapIDs(ids []int) ([]int, error)     { return s.lookup(ids, false) }
func (s *Secret) RestoreIDs(ids []int) ([]int, error) { return s.lookup(ids, true) }
func (s *Secret) lookup(ids []int, inverse bool) ([]int, error) {
	if s == nil || len(ids) > 4096 {
		return nil, errors.New("invalid dense token mapping request")
	}
	mapping := s.disk.Permutation
	if inverse {
		mapping = s.inverse
	}
	result := make([]int, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(mapping) {
			return nil, errors.New("dense token ID outside vocabulary")
		}
		result[i] = mapping[id]
	}
	return result, nil
}
func (s *Secret) RestoreLogits(physical []float64) ([]float64, error) {
	if s == nil || len(physical) != len(s.inverse) {
		return nil, errors.New("dense logit vocabulary mismatch")
	}
	result := make([]float64, len(physical))
	for original, mapped := range s.disk.Permutation {
		value := physical[mapped]
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, errors.New("nonfinite dense logits")
		}
		result[original] = value
	}
	return result, nil
}
func contextHash(p TransformPlan, b model.SourceBinding) string {
	return digest(struct {
		Plan   TransformPlan
		Source model.SourceBinding
	}{p, b})
}
func newSecret(p TransformPlan, b model.SourceBinding) (*Secret, error) {
	seed := make([]byte, 32)
	if _, err := rand.Read(seed); err != nil {
		return nil, errors.New("dense entropy failed")
	}
	d := secretDisk{SchemaVersion: 1, ContextHash: contextHash(p, b), Seed: hex.EncodeToString(seed), Permutation: make([]int, p.Contract.Vocabulary)}
	for i := range d.Permutation {
		d.Permutation[i] = i
	}
	for i := len(d.Permutation) - 1; i > 0; i-- {
		j, err := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		if err != nil {
			return nil, errors.New("dense entropy failed")
		}
		d.Permutation[i], d.Permutation[j.Int64()] = d.Permutation[j.Int64()], d.Permutation[i]
	}
	d.BindingID = commitment(d)
	return validateSecret(d)
}
