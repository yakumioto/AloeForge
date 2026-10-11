package referencebackend

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/yakumioto/AloeForge/internal/artifact"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

func fixture(t *testing.T, mode string) (*Backend, p.Request) {
	t.Helper()
	c := &artifact.DenseContract{SchemaVersion: 1, Profile: "diagnostic", MethodRevision: artifact.DenseMethod, RNGAlgorithm: artifact.DenseRNG, RNGRevision: "1", LoaderRevision: "numpy-qwen25-v1", BindingID: strings.Repeat("2", 64), TokenizerFingerprint: strings.Repeat("3", 64), Vocabulary: 3, LogicalHidden: 4, PhysicalHidden: 8, Intermediate: 5, Layers: 1, Heads: 2, KVHeads: 1, HeadDim: 2, Context: 16, NormEpsilon: 1e-5, RopeTheta: 10000, RopeLayout: "split-half", CacheLayout: "layer-kv-head-position-dim-v1", MappedSpecialIDs: map[string][]int{"bos": {0}, "eos": {1}, "pad": {0}}}
	plan := artifact.Plan{SchemaVersion: 1, SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("1", 64), MethodRevision: artifact.DenseMethod, RNGAlgorithm: artifact.DenseRNG, RNGRevision: "1", Dense: c}
	readers := map[string]io.Reader{}
	for name, shape := range artifact.DenseShapes(*c) {
		size := 4
		for _, dim := range shape {
			size *= int(dim)
		}
		data := make([]byte, size)
		sum := sha256.Sum256(data)
		plan.Outputs = append(plan.Outputs, artifact.OutputTensor{Name: name, Shard: "model.safetensors", DType: "F32", Shape: shape, PayloadHash: hex.EncodeToString(sum[:])})
		readers[name] = bytes.NewReader(data)
	}
	parent := t.TempDir()
	if os.Chmod(parent, 0700) != nil {
		t.Fatal("private fixture directory")
	}
	stage, err := artifact.Create(parent, ".stage-reference", plan)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if err := stage.WriteShard("model.safetensors", readers); err != nil {
		t.Fatal(err)
	}
	id, err := stage.Publish("published")
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	wrapper := filepath.Join(parent, "fake-python")
	// This trusted test helper replaces Python and runs only the IPC contract.
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexec '"+strings.ReplaceAll(executable, "'", "'\\''")+"' -test.run=^TestReferenceHelper$ -- \"$@\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	r := p.Runtime{ProtocolVersion: 1, Profile: p.LocalDenseProfile, ModelID: mode, ArtifactID: id, Dense: c, Features: []string{"greedy", "sse", "stop_token_ids"}, MaxPromptTokens: 8, MaxOutputTokens: 8, ContextTokens: 16}
	b, err := New(Config{Runtime: r, ArtifactRoot: filepath.Join(parent, "published"), Python: wrapper})
	if err != nil {
		t.Fatal(err)
	}
	temp := 0.0
	q := p.Request{ProtocolVersion: 1, ModelID: r.ModelID, ArtifactID: id, BindingID: c.BindingID, RequestID: strings.Repeat("a", 32), PromptTokenIDs: []int{0}, StopTokenIDs: []int{}, Sampling: p.Sampling{Temperature: &temp, MaxTokens: 3}, Stream: true}
	return b, q
}

func TestReferenceHelper(t *testing.T) {
	if os.Args[len(os.Args)-1] != "aloeforge_runtime.reference_worker" {
		return
	}
	raw, err := readOpenForTest(os.Stdin)
	if err != nil {
		os.Exit(2)
	}
	var open struct {
		Runtime p.Runtime `json:"runtime"`
		Request p.Request `json:"request"`
	}
	if json.Unmarshal(raw, &open) != nil {
		os.Exit(2)
	}
	mode := open.Runtime.ModelID
	// Parent must strip ambient credentials/Python injection and suppress stderr.
	if os.Getenv("ALOE_PRIVATE_MARKER") != "" || os.Getenv("PYTHONPATH") != "" {
		os.Exit(3)
	}
	_, _ = io.WriteString(os.Stderr, "PRIVATE_BACKEND_DIAGNOSTIC")
	if mode == "hang-open" {
		for {
			time.Sleep(time.Hour)
		}
	}
	ready := map[string]any{"schema_version": 1, "event": "ready", "request_id": open.Request.RequestID, "artifact_id": open.Runtime.ArtifactID, "binding_id": open.Runtime.BindingID(), "tokenizer_sha256": open.Runtime.TokenizerFingerprint()}
	if mode == "bad-ready" {
		ready["binding_id"] = strings.Repeat("0", 64)
	}
	_ = writeFrame(os.Stdout, ready, maxFrame)
	if mode == "idle" || mode == "hang-next" {
		for {
			time.Sleep(time.Hour)
		}
	}
	for sequence := 1; ; sequence++ {
		if _, err := readFrame(os.Stdin); err != nil {
			os.Exit(0)
		}
		token := map[string]any{"schema_version": 1, "event": "token", "request_id": open.Request.RequestID, "sequence": sequence, "physical_token_id": 2}
		switch mode {
		case "bad-sequence":
			token["sequence"] = sequence + 1
		case "bad-id":
			token["physical_token_id"] = 3
		case "null-id":
			token["physical_token_id"] = nil
		case "unknown":
			token["private"] = "PRIVATE_BACKEND_DIAGNOSTIC"
		case "eos":
			token["physical_token_id"] = 1
		case "oversize":
			var prefix [4]byte
			binary.BigEndian.PutUint32(prefix[:], maxFrame+1)
			_, _ = os.Stdout.Write(prefix[:])
			os.Exit(0)
		case "truncated":
			_, _ = os.Stdout.Write([]byte{0, 0, 0, 4, '{'})
			os.Exit(0)
		case "duplicate":
			raw := []byte(`{"schema_version":1,"schema_version":1}`)
			var prefix [4]byte
			binary.BigEndian.PutUint32(prefix[:], uint32(len(raw)))
			_, _ = os.Stdout.Write(append(prefix[:], raw...))
			os.Exit(0)
		}
		if writeFrame(os.Stdout, token, maxFrame) != nil {
			os.Exit(0)
		}
	}
}

func readOpenForTest(r io.Reader) ([]byte, error) {
	var prefix [4]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		return nil, err
	}
	size := binary.BigEndian.Uint32(prefix[:])
	if size > p.MaxBody {
		return nil, errBackend
	}
	raw := make([]byte, size)
	_, err := io.ReadFull(r, raw)
	return raw, err
}

func assertReaped(t *testing.T, b *Backend, s *stream) {
	t.Helper()
	s.Close()
	if len(b.slots) != 0 || s.cmd.ProcessState == nil || syscall.Kill(s.cmd.Process.Pid, 0) != syscall.ESRCH {
		t.Fatal("child or capacity leaked")
	}
}

func TestProcessFramesStopAndReap(t *testing.T) {
	t.Setenv("ALOE_PRIVATE_MARKER", "PRIVATE_BACKEND_DIAGNOSTIC")
	t.Setenv("PYTHONPATH", "PRIVATE_BACKEND_DIAGNOSTIC")
	for _, mode := range []string{"normal", "eos", "bad-ready", "bad-sequence", "bad-id", "null-id", "unknown", "oversize", "truncated", "duplicate"} {
		t.Run(mode, func(t *testing.T) {
			b, q := fixture(t, mode)
			s, err := b.Open(context.Background(), q)
			if mode == "bad-ready" {
				if err == nil || len(b.slots) != 0 {
					t.Fatal("invalid ready accepted/leaked")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			child := s.(*stream)
			defer assertReaped(t, b, child)
			id, ok, err := s.Next(context.Background())
			if mode != "normal" && mode != "eos" {
				if err == nil || err.Error() != "reference_backend_error" {
					t.Fatal("invalid token accepted/diagnostic escaped")
				}
				return
			}
			if err != nil || !ok || id < 1 {
				t.Fatal("token missing")
			}
			if mode == "normal" {
				for range 2 {
					if _, ok, err := s.Next(context.Background()); !ok || err != nil {
						t.Fatal("bound stream missing")
					}
				}
			}
			before := child.sequence
			if _, ok, err := s.Next(context.Background()); ok || err != nil || child.sequence != before {
				t.Fatal("terminal prefetched")
			}
		})
	}
}

func TestProcessTimeoutCancellationCapacityAndConcurrentClose(t *testing.T) {
	for _, mode := range []string{"hang-open", "hang-next", "idle", "normal"} {
		t.Run(mode, func(t *testing.T) {
			b, q := fixture(t, mode)
			b.timeout = 100 * time.Millisecond
			start := time.Now()
			s, err := b.Open(context.Background(), q)
			if mode == "hang-open" {
				if err == nil || len(b.slots) != 0 || time.Since(start) > 2*time.Second {
					t.Fatal("open timeout failed")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			child := s.(*stream)
			defer assertReaped(t, b, child)
			if _, err := b.Open(context.Background(), q); err == nil {
				t.Fatal("capacity ignored")
			}
			if mode == "idle" {
				<-child.ctx.Done()
				child.Close()
				if len(b.slots) != 0 {
					t.Fatal("idle child retained slot")
				}
				return
			}
			ctx, cancel := context.WithCancel(context.Background())
			if mode == "hang-next" {
				timer := time.AfterFunc(20*time.Millisecond, cancel)
				defer timer.Stop()
				if _, _, err := s.Next(ctx); err == nil {
					t.Fatal("next cancellation ignored")
				}
			} else {
				var wait sync.WaitGroup
				for range 4 {
					wait.Go(s.Close)
				}
				wait.Wait()
			}
			cancel()
			if time.Since(start) > 2*time.Second {
				t.Fatal("child cleanup exceeded bound")
			}
		})
	}
}

func TestBackendConfigurationAndStartFailures(t *testing.T) {
	b, q := fixture(t, "normal")
	cfg := Config{Runtime: b.Runtime(), ArtifactRoot: b.root, Python: b.python}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Runtime.Profile = p.Profile },
		func(c *Config) { c.Runtime.ArtifactID = strings.Repeat("0", 64) },
		func(c *Config) { c.Runtime.Dense.RopeTheta++ },
		func(c *Config) { c.ArtifactRoot = "relative" },
		func(c *Config) { c.Python = "python" },
		func(c *Config) { c.MaxConcurrent = 3 },
		func(c *Config) { c.SessionTimeout = time.Minute },
	} {
		x := cfg
		x.Runtime = p.Clone(cfg.Runtime)
		mutate(&x)
		if _, err := New(x); err == nil {
			t.Fatal("invalid backend configuration admitted")
		}
	}
	b.python = filepath.Join(t.TempDir(), "unavailable-python")
	if _, err := b.Open(context.Background(), q); err == nil || len(b.slots) != 0 {
		t.Fatal("start failure retained capacity")
	}
	q.PromptTokenIDs = []int{3}
	if _, err := b.Open(context.Background(), q); err == nil || len(b.slots) != 0 {
		t.Fatal("bad physical request started child")
	}
}

func FuzzBoundedFrame(f *testing.F) {
	f.Add([]byte{0, 0, 0, 2, '{', '}'})
	f.Add([]byte{255, 255, 255, 255})
	f.Fuzz(func(t *testing.T, raw []byte) {
		frame, err := readFrame(bytes.NewReader(raw))
		if err == nil && (len(frame) < 1 || len(frame) > maxFrame) {
			t.Fatal("frame bounds escaped")
		}
	})
}
