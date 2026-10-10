package tokenbaseline

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

type Options struct {
	SourceRoot   string
	OutputParent string
	SecretRoot   string
	StageName    string
	TargetName   string
	Resume       bool
}

// Result contains public diagnostic identity only. No token IDs or source paths.
type Result struct {
	Kind                 string `json:"kind"`
	ArtifactID           string `json:"artifact_id"`
	BindingID            string `json:"binding_id"`
	TokenizerFingerprint string `json:"tokenizer_sha256"`
	VocabularySize       int    `json:"vocabulary_size"`
}

func contained(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func separateRoots(paths ...string) error {
	roots := make([]string, len(paths))
	for i, path := range paths {
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			return errors.New("compiler directories must exist")
		}
		resolved, err = filepath.Abs(resolved)
		if err != nil {
			return errors.New("cannot resolve compiler directory")
		}
		stat, err := os.Stat(resolved)
		if err != nil || !stat.IsDir() {
			return errors.New("compiler roots must be directories")
		}
		for _, prior := range roots[:i] {
			if contained(prior, resolved) || contained(resolved, prior) {
				return errors.New("source, output and Client Secret directories must be disjoint")
			}
		}
		roots[i] = resolved
	}
	return nil
}

func validName(name string) bool {
	if len(name) < 1 || len(name) > 80 || name == "." || name == ".." {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func buildPlan(s *source, secret *Secret) (artifact.Plan, error) {
	c := s.context
	plan := artifact.Plan{SchemaVersion: 1, SourceInventoryHash: c.Source.InventoryHash, SourceContentHash: c.Source.ContentHash, MethodRevision: c.MethodRevision, RNGAlgorithm: c.RNGAlgorithm, RNGRevision: c.RNGRevision, Binding: secret.publicBinding()}
	for _, name := range []string{"embedding.weight", "lm_head.weight"} {
		if _, ok := s.matrices[name]; !ok {
			continue
		}
		digest := sha256.New()
		if _, err := io.CopyBuffer(digest, s.reader(name, secret), make([]byte, 32<<10)); err != nil {
			return artifact.Plan{}, errors.New("token baseline matrix planning failed")
		}
		plan.Outputs = append(plan.Outputs, artifact.OutputTensor{Name: name, Shard: "model.safetensors", DType: "F32", Shape: []int64{int64(c.VocabularySize), int64(c.HiddenSize)}, PayloadHash: hex.EncodeToString(digest.Sum(nil))})
	}
	return plan, nil
}

func result(secret *Secret, digest string) Result {
	b := secret.publicBinding()
	return Result{Kind: artifact.TokenKind, ArtifactID: digest, BindingID: b.BindingID, TokenizerFingerprint: b.TokenizerFingerprint, VocabularySize: b.VocabularySize}
}

// Compile has no remote/Python bypass: Go binds source, plans byte hashes, saves
// the external secret, verifies staged output, and publishes through P1 gates.
func Compile(options Options) (Result, error) {
	if !strings.HasPrefix(options.StageName, ".stage-") || !validName(strings.TrimPrefix(options.StageName, ".stage-")) || !validName(options.TargetName) || strings.HasPrefix(options.TargetName, ".") {
		return Result{}, errors.New("invalid diagnostic stage or target name")
	}
	if err := separateRoots(options.SourceRoot, options.OutputParent, options.SecretRoot); err != nil {
		return Result{}, err
	}
	// Validate private directory gates before creating permanent secret material.
	for _, path := range []string{options.OutputParent, options.SecretRoot} {
		root, err := privateRoot(path)
		if err != nil {
			return Result{}, errors.New("compiler output and secret directories must be private (0700)")
		}
		if !options.Resume {
			name := secretFile
			if path == options.OutputParent {
				name = options.StageName
			}
			_, statErr := root.Lstat(name)
			if statErr == nil || !os.IsNotExist(statErr) {
				root.Close()
				return Result{}, errors.New("fresh compilation requires absent stage and Client Secret; use explicit resume")
			}
		}
		root.Close()
	}
	s, err := loadSource(options.SourceRoot)
	if err != nil {
		return Result{}, err
	}
	defer s.close()
	var secret *Secret
	if options.Resume {
		secret, err = LoadSecret(options.SecretRoot)
		if err != nil {
			return Result{}, err
		}
		if !secret.matches(s.context) {
			return Result{}, errors.New("Client Secret source/tokenizer/layout binding mismatch")
		}
	} else {
		secret, err = NewSecret(s.context)
		if err != nil {
			return Result{}, err
		}
	}
	plan, err := buildPlan(s, secret)
	if err != nil {
		return Result{}, err
	}
	if !options.Resume {
		if err := SaveSecret(options.SecretRoot, secret); err != nil {
			return Result{}, err
		}
	}
	if err := syncStoredSecret(options.SecretRoot); err != nil {
		return Result{}, err
	}
	var stage *artifact.Stage
	if options.Resume {
		parent, openErr := os.OpenRoot(options.OutputParent)
		if openErr != nil {
			return Result{}, errors.New("cannot open token baseline output parent")
		}
		_, statErr := parent.Lstat(options.StageName)
		parent.Close()
		if os.IsNotExist(statErr) {
			stage, err = artifact.Create(options.OutputParent, options.StageName, plan)
		} else {
			stage, err = artifact.Resume(options.OutputParent, options.StageName, plan)
		}
	} else {
		stage, err = artifact.Create(options.OutputParent, options.StageName, plan)
	}
	if err != nil {
		return Result{}, err
	}
	defer stage.Close()
	pending, err := stage.PendingShards()
	if err != nil {
		return Result{}, err
	}
	for _, shard := range pending {
		readers := make(map[string]io.Reader)
		for _, output := range plan.Outputs {
			readers[output.Name] = s.reader(output.Name, secret)
		}
		if err := stage.WriteShard(shard, readers); err != nil {
			return Result{}, err
		}
	}
	// Recheck source bytes and private sidecars before publication. Inputs must
	// stay immutable; privileged concurrent filesystem writers are out of scope.
	fresh, err := loadSource(options.SourceRoot)
	if err != nil {
		return Result{}, err
	}
	unchanged := secret.matches(fresh.context)
	fresh.close()
	if !unchanged {
		return Result{}, errors.New("source or tokenizer changed before publication")
	}
	digest, err := stage.Publish(options.TargetName)
	if err != nil {
		return Result{}, err
	}
	return result(secret, digest), nil
}

// VerifyClient requires identities from the trusted compiler/client channel.
// A self-declared manifest/binding alone never authenticates an artifact.
func VerifyClient(artifactRoot, secretRoot, expectedArtifactID, expectedTokenizer string) (Result, error) {
	if !validHash(expectedArtifactID) || !validHash(expectedTokenizer) {
		return Result{}, errors.New("invalid expected client identities")
	}
	manifest, digest, err := artifact.Verify(artifactRoot)
	if err != nil {
		return Result{}, err
	}
	if manifest.Kind != artifact.TokenKind || manifest.Binding == nil || subtle.ConstantTimeCompare([]byte(digest), []byte(expectedArtifactID)) != 1 {
		return Result{}, errors.New("client artifact identity mismatch")
	}
	secret, err := LoadSecret(secretRoot)
	if err != nil {
		return Result{}, err
	}
	if secret.disk.Material.Context.TokenizerFingerprint != expectedTokenizer {
		return Result{}, errors.New("client tokenizer identity mismatch")
	}
	want, _ := json.Marshal(secret.publicBinding())
	actual, _ := json.Marshal(manifest.Binding)
	if subtle.ConstantTimeCompare(want, actual) != 1 {
		return Result{}, errors.New("client secret and artifact binding mismatch")
	}
	return result(secret, digest), nil
}
