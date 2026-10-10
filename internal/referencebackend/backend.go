// Package referencebackend runs the artifact-only Python FP64 reference on a
// trusted local machine. It is not an untrusted-server or GPU inference backend.
package referencebackend

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sync"
	"syscall"
	"time"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/gateway"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
	"github.com/yakumioto/AloeForge/internal/strictjson"
)

const maxFrame = 4096

var errBackend = errors.New("reference_backend_error")

type Config struct {
	Runtime        p.Runtime
	ArtifactRoot   string        // Published artifact only; no source/Secret directory.
	Python         string        // Trusted absolute interpreter with the installed package.
	MaxConcurrent  int           // Default 1, maximum 2 independent worker processes.
	SessionTimeout time.Duration // Default and maximum 30 seconds.
}

type Backend struct {
	runtime      p.Runtime
	root, python string
	timeout      time.Duration
	slots        chan struct{}
}

func New(c Config) (*Backend, error) {
	if c.Runtime.Profile != p.LocalDenseProfile || c.Runtime.Validate() != nil || !filepath.IsAbs(c.ArtifactRoot) || !filepath.IsAbs(c.Python) {
		return nil, errBackend
	}
	if c.MaxConcurrent == 0 {
		c.MaxConcurrent = 1
	}
	if c.SessionTimeout == 0 {
		c.SessionTimeout = 30 * time.Second
	}
	if c.MaxConcurrent < 1 || c.MaxConcurrent > 2 || c.SessionTimeout <= 0 || c.SessionTimeout > 30*time.Second {
		return nil, errBackend
	}
	manifest, id, err := artifact.Verify(c.ArtifactRoot)
	if err != nil || id != c.Runtime.ArtifactID || manifest.Kind != artifact.DenseKind || manifest.Dense == nil || !p.Equal(manifest.Dense, c.Runtime.Dense) {
		return nil, errBackend
	}
	return &Backend{runtime: p.Clone(c.Runtime), root: c.ArtifactRoot, python: c.Python, timeout: c.SessionTimeout, slots: make(chan struct{}, c.MaxConcurrent)}, nil
}

func (b *Backend) Runtime() p.Runtime { return p.Clone(b.runtime) }

// Open starts one isolated child and validates its ready identity. Work is pulled
// synchronously; no background producer, retry, shared cache or output queue.
func (b *Backend) Open(ctx context.Context, req p.Request) (gateway.TokenStream, error) {
	if ctx.Err() != nil || req.Validate(b.runtime) != nil {
		return nil, errBackend
	}
	select {
	case b.slots <- struct{}{}:
	default:
		return nil, errBackend
	}
	workerCtx, cancel := context.WithTimeout(ctx, b.timeout)
	cmd := exec.CommandContext(workerCtx, b.python, "-I", "-m", "aloeforge_runtime.reference_worker")
	// Do not inherit credentials, PYTHONPATH, preload hooks or proxy settings.
	cmd.Env = []string{"PATH=/usr/bin:/bin", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"}
	cmd.Dir = b.root
	cmd.Stderr = io.Discard
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if err == syscall.ESRCH {
			return os.ErrProcessDone
		}
		return err
	}
	cmd.WaitDelay = time.Second
	owned := req
	owned.PromptTokenIDs = slices.Clone(req.PromptTokenIDs)
	owned.StopTokenIDs = slices.Clone(req.StopTokenIDs)
	temperature := *req.Sampling.Temperature
	owned.Sampling.Temperature = &temperature
	s := &stream{cmd: cmd, cancel: cancel, release: func() { <-b.slots }, req: owned, vocab: b.runtime.VocabularySize(), eos: slices.Clone(b.runtime.MappedSpecialIDs()["eos"]), ctx: workerCtx}
	var err error
	s.input, err = cmd.StdinPipe()
	if err == nil {
		s.output, err = cmd.StdoutPipe()
	}
	if err == nil {
		err = cmd.Start()
		s.started = err == nil
	}
	if err != nil {
		s.Close()
		return nil, errBackend
	}
	// A caller that stops pulling still has a bounded child lifetime and a
	// reaped process. Close is safe concurrently with a pending pipe read/write.
	context.AfterFunc(workerCtx, s.Close)
	open := struct {
		Schema    int       `json:"schema_version"`
		Operation string    `json:"operation"`
		Root      string    `json:"artifact_root"`
		Runtime   p.Runtime `json:"runtime"`
		Request   p.Request `json:"request"`
	}{1, "open", b.root, b.runtime, owned}
	if writeFrame(s.input, open, p.MaxBody) != nil {
		s.Close()
		return nil, errBackend
	}
	raw, err := readFrame(s.output)
	var ready struct {
		Schema     int    `json:"schema_version"`
		Event      string `json:"event"`
		RequestID  string `json:"request_id"`
		ArtifactID string `json:"artifact_id"`
		BindingID  string `json:"binding_id"`
		Tokenizer  string `json:"tokenizer_sha256"`
	}
	if err != nil || decode(raw, &ready, "schema_version", "event", "request_id", "artifact_id", "binding_id", "tokenizer_sha256") != nil || ready.Schema != 1 || ready.Event != "ready" || ready.RequestID != req.RequestID || ready.ArtifactID != b.runtime.ArtifactID || ready.BindingID != b.runtime.BindingID() || ready.Tokenizer != b.runtime.TokenizerFingerprint() || workerCtx.Err() != nil {
		s.Close()
		return nil, errBackend
	}
	return s, nil
}

type stream struct {
	cmd             *exec.Cmd
	input           io.WriteCloser
	output          io.ReadCloser
	ctx             context.Context
	cancel          context.CancelFunc
	release         func()
	started         bool
	req             p.Request
	vocab, sequence int
	terminal        bool
	eos             []int
	mu              sync.Mutex // Serial Next calls; Close remains cancellation-safe.
	once            sync.Once
}

func (s *stream) Close() {
	s.once.Do(func() {
		s.cancel() // Kill before waiting; never block trying to send graceful close.
		if s.input != nil {
			_ = s.input.Close()
		}
		if s.started {
			// Kill the group explicitly as well: the leader may have exited
			// before CommandContext's watcher observes cancellation.
			_ = syscall.Kill(-s.cmd.Process.Pid, syscall.SIGKILL)
			_ = s.cmd.Wait()
		}
		if s.output != nil {
			_ = s.output.Close()
		}
		s.release()
	})
}

func (s *stream) Next(ctx context.Context) (int, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ctx.Err() != nil || s.ctx.Err() != nil {
		s.Close()
		return 0, false, errBackend
	}
	if s.terminal {
		return 0, false, nil
	}
	stop := context.AfterFunc(ctx, s.cancel)
	defer stop()
	s.sequence++
	if s.sequence > s.req.Sampling.MaxTokens {
		return 0, false, nil
	}
	pull := struct {
		Schema    int    `json:"schema_version"`
		Operation string `json:"operation"`
		RequestID string `json:"request_id"`
		Sequence  int    `json:"sequence"`
	}{1, "pull", s.req.RequestID, s.sequence}
	if writeFrame(s.input, pull, maxFrame) != nil {
		s.Close()
		return 0, false, errBackend
	}
	raw, err := readFrame(s.output)
	var token struct {
		Schema    int    `json:"schema_version"`
		Event     string `json:"event"`
		RequestID string `json:"request_id"`
		Sequence  int    `json:"sequence"`
		ID        int    `json:"physical_token_id"`
	}
	if err != nil || decode(raw, &token, "schema_version", "event", "request_id", "sequence", "physical_token_id") != nil || token.Schema != 1 || token.Event != "token" || token.RequestID != s.req.RequestID || token.Sequence != s.sequence || token.ID < 0 || token.ID >= s.vocab || ctx.Err() != nil || s.ctx.Err() != nil {
		s.Close()
		return 0, false, errBackend
	}
	s.terminal = s.sequence == s.req.Sampling.MaxTokens || slices.Contains(s.req.StopTokenIDs, token.ID) || slices.Contains(s.eos, token.ID)
	return token.ID, true, nil
}

func writeFrame(w io.Writer, value any, limit int) error {
	raw, err := json.Marshal(value)
	if err != nil || len(raw) < 1 || len(raw) > limit {
		return errBackend
	}
	frame := make([]byte, 4+len(raw))
	binary.BigEndian.PutUint32(frame, uint32(len(raw)))
	copy(frame[4:], raw)
	for len(frame) > 0 {
		n, err := w.Write(frame)
		if err != nil || n < 1 {
			return errBackend
		}
		frame = frame[n:]
	}
	return nil
}
func readFrame(r io.Reader) ([]byte, error) {
	var prefix [4]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		return nil, errBackend
	}
	size := binary.BigEndian.Uint32(prefix[:])
	if size < 1 || size > maxFrame {
		return nil, errBackend
	}
	raw := make([]byte, size)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, errBackend
	}
	return raw, nil
}
func decode(raw []byte, value any, names ...string) error {
	if strictjson.Decode(raw, value, true) != nil {
		return errBackend
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || len(fields) != len(names) {
		return errBackend
	}
	for _, name := range names {
		data, ok := fields[name]
		if !ok || bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
			return errBackend
		}
	}
	return nil
}
