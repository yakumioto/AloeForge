package dense

import (
	"errors"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

// VerifyClient checks immutable artifact identity against the trusted compiler
// channel and the external private secret commitment. It loads no source model.
func VerifyClient(path, secretRoot, expectedID, tokenizer string) (Result, error) {
	if !validHash(expectedID) || !validHash(tokenizer) {
		return Result{}, errors.New("external dense identities required")
	}
	m, id, err := artifact.Verify(path)
	if err != nil {
		return Result{}, err
	}
	if m.Kind != artifact.DenseKind || m.Dense == nil || id != expectedID || m.Dense.TokenizerFingerprint != tokenizer {
		return Result{}, errors.New("dense artifact identity mismatch")
	}
	s, err := LoadSecret(secretRoot)
	if err != nil {
		return Result{}, err
	}
	if s.disk.BindingID != m.Dense.BindingID || len(s.disk.Permutation) != m.Dense.Vocabulary {
		return Result{}, errors.New("dense Client Secret binding mismatch")
	}
	return Result{Kind: m.Kind, ArtifactID: id, BindingID: m.Dense.BindingID, TokenizerFingerprint: tokenizer}, nil
}
