package dense

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/safetensors"
	"github.com/yakumioto/AloeForge/internal/strictjson"
	"golang.org/x/sys/unix"
)

type Options struct {
	SourceRoot, OutputParent, SecretRoot, StageName, TargetName, Python string
	Expansion                                                           int
	Resume                                                              bool
}
type Result struct {
	Kind                 string `json:"kind"`
	ArtifactID           string `json:"artifact_id"`
	BindingID            string `json:"binding_id"`
	TokenizerFingerprint string `json:"tokenizer_sha256"`
}

func separateRoots(paths ...string) error {
	roots := []string{}
	for _, p := range paths {
		r, err := filepath.EvalSymlinks(p)
		if err != nil {
			return errors.New("compiler roots must exist")
		}
		r, err = filepath.Abs(r)
		if err != nil {
			return errors.New("cannot resolve compiler roots")
		}
		for _, prior := range roots {
			for _, pair := range [][2]string{{prior, r}, {r, prior}} {
				rel, err := filepath.Rel(pair[0], pair[1])
				if err != nil || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
					return errors.New("source, output and secret roots must be disjoint")
				}
			}
		}
		roots = append(roots, r)
	}
	return nil
}

type boundedBuffer struct{ bytes.Buffer }

func (b *boundedBuffer) Write(p []byte) (int, error) {
	if b.Len()+len(p) > model.MaxMetadataBytes {
		return 0, errors.New("worker control exceeds bounds")
	}
	return b.Buffer.Write(p)
}

// Compile gives the worker private source/secret access only offline. Go alone
// controls staging, recomputes descriptor/content hashes and atomically publishes.
func Compile(ctx context.Context, o Options) (Result, error) {
	fail := func() (Result, error) { return Result{}, errors.New("dense offline worker failed") }
	if err := separateRoots(o.SourceRoot, o.OutputParent, o.SecretRoot); err != nil {
		return Result{}, err
	}
	for _, path := range []string{o.OutputParent, o.SecretRoot} {
		r, e := privateRoot(path)
		if e != nil {
			return Result{}, e
		}
		r.Close()
	}
	if !strings.HasPrefix(o.StageName, ".stage-") || !safeName(strings.TrimPrefix(o.StageName, ".stage-")) || !safeName(o.TargetName) || strings.HasPrefix(o.TargetName, ".") {
		return Result{}, errors.New("invalid dense stage or target")
	}
	p, err := BuildPlan(o.SourceRoot, o.Expansion)
	if err != nil {
		return Result{}, err
	}
	b, err := model.BindSource(o.SourceRoot)
	if err != nil {
		return Result{}, err
	}
	if b.InventoryHash != p.Inventory.Hash() {
		return Result{}, errors.New("source changed during planning")
	}
	var s *Secret
	if o.Resume {
		s, err = LoadSecret(o.SecretRoot)
		if err != nil {
			return Result{}, err
		}
		if s.disk.ContextHash != contextHash(p, b) || len(s.disk.Permutation) != p.Contract.Vocabulary {
			return Result{}, errors.New("dense source/tokenizer/method/geometry secret mismatch")
		}
	} else {
		if _, e := os.Lstat(filepath.Join(o.OutputParent, o.StageName)); !os.IsNotExist(e) {
			return Result{}, errors.New("fresh dense compilation requires absent stage")
		}
		s, err = newSecret(p, b)
		if err != nil {
			return Result{}, err
		}
		if err = SaveSecret(o.SecretRoot, s); err != nil {
			return Result{}, err
		}
	}
	if err = syncStoredSecret(o.SecretRoot); err != nil {
		return Result{}, err
	}
	p.Contract.BindingID = s.disk.BindingID
	for role, ids := range p.Contract.MappedSpecialIDs {
		for i, id := range ids {
			ids[i] = s.disk.Permutation[id]
		}
		sort.Ints(ids)
		p.Contract.MappedSpecialIDs[role] = ids
	}
	raw, err := json.Marshal(p)
	if err != nil || len(raw) > model.MaxMetadataBytes {
		return fail()
	}
	scratch, err := os.MkdirTemp(o.OutputParent, ".dense-worker-")
	if err != nil {
		return fail()
	}
	defer os.RemoveAll(scratch)
	cmd := exec.CommandContext(ctx, o.Python, "-m", "aloeforge_worker.dense", "--source-root", o.SourceRoot, "--scratch-root", scratch, "--secret-root", o.SecretRoot)
	cmd.Stdin = bytes.NewReader(raw)
	output := &boundedBuffer{}
	cmd.Stdout = output
	cmd.Stderr = io.Discard
	if cmd.Run() != nil {
		return fail()
	}
	var descriptors []artifact.OutputTensor
	if strictjson.Decode(output.Bytes(), &descriptors, true) != nil || len(descriptors) != len(p.Nodes) {
		return fail()
	}
	root, err := os.OpenRoot(scratch)
	if err != nil {
		return fail()
	}
	defer root.Close()
	directory, err := root.Open(".")
	if err != nil {
		return fail()
	}
	entries, err := directory.ReadDir(len(p.Nodes) + 1)
	directory.Close()
	if err != nil && err != io.EOF || len(entries) != len(p.Nodes) {
		return fail()
	}
	for i, n := range p.Nodes {
		d := descriptors[i]
		if d.Name != n.Output || d.Shard != n.Shard || d.DType != "F32" || digest(d.Shape) != digest(n.Shape) {
			return fail()
		}
		f, e := openPayload(root, n.File)
		if e != nil {
			return fail()
		}
		stat, e := f.Stat()
		size, _ := safetensors.ByteSize("F32", n.Shape)
		h := sha256.New()
		count, e2 := io.CopyBuffer(h, f, make([]byte, 32<<10))
		f.Close()
		if e != nil || e2 != nil || stat.Size() != size || count != size || hex.EncodeToString(h.Sum(nil)) != d.PayloadHash {
			return fail()
		}
	}
	current, err := model.BindSource(o.SourceRoot)
	if err != nil || current != b {
		return Result{}, errors.New("source changed during transformation")
	}
	// Tokenizer is not a shard: separately recheck its fingerprint/config/roles.
	check, err := BuildPlan(o.SourceRoot, o.Expansion)
	if err != nil || s.disk.ContextHash != contextHash(check, current) {
		return Result{}, errors.New("source metadata changed during transformation")
	}
	plan := artifact.Plan{SchemaVersion: 1, SourceInventoryHash: b.InventoryHash, SourceContentHash: b.ContentHash, MethodRevision: artifact.DenseMethod, RNGAlgorithm: artifact.DenseRNG, RNGRevision: "1", Outputs: descriptors, Dense: &p.Contract}
	var stage *artifact.Stage
	if o.Resume {
		if _, e := os.Lstat(filepath.Join(o.OutputParent, o.StageName)); os.IsNotExist(e) {
			stage, err = artifact.Create(o.OutputParent, o.StageName, plan)
		} else {
			stage, err = artifact.Resume(o.OutputParent, o.StageName, plan)
		}
	} else {
		stage, err = artifact.Create(o.OutputParent, o.StageName, plan)
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
		readers := map[string]io.Reader{}
		files := []*os.File{}
		for _, n := range p.Nodes {
			if n.Shard == shard {
				f, e := openPayload(root, n.File)
				if e != nil {
					for _, f := range files {
						f.Close()
					}
					return fail()
				}
				files = append(files, f)
				readers[n.Output] = f
			}
		}
		err = stage.WriteShard(shard, readers)
		for _, f := range files {
			f.Close()
		}
		if err != nil {
			return Result{}, err
		}
	}
	id, err := stage.Publish(o.TargetName)
	if err != nil {
		return Result{}, err
	}
	return Result{Kind: artifact.DenseKind, ArtifactID: id, BindingID: s.disk.BindingID, TokenizerFingerprint: p.Contract.TokenizerFingerprint}, nil
}
func safeName(s string) bool {
	if len(s) < 1 || len(s) > 80 || s == "." || s == ".." {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}
func openPayload(root *os.Root, name string) (*os.File, error) {
	stat, err := root.Lstat(name)
	if err != nil || !stat.Mode().IsRegular() {
		return nil, errors.New("worker payload must be regular")
	}
	f, err := root.OpenFile(name, os.O_RDONLY|unix.O_NONBLOCK|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, errors.New("cannot open worker payload")
	}
	current, err := f.Stat()
	if err != nil || !os.SameFile(stat, current) || !current.Mode().IsRegular() {
		f.Close()
		return nil, errors.New("worker payload changed")
	}
	return f, nil
}
