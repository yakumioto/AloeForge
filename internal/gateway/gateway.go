// Package gateway implements bounded TLS/SSE diagnostic relays. It has no access
// to tokenizer, Client Secret or weights; the local reference adapter owns the
// public publication path on a trusted machine.
package gateway

import (
	"context"
	"crypto/subtle"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"slices"
	"sync"
	"time"

	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

const MaxRequests = 4096
const MaxConcurrent = 8

// Fake holds a fixed physical-coordinate script independent of the prompt.
// Delay is context-aware and bounded, for exercising cancellation on CPU.
type Fake struct {
	OutputTokenIDs []int
	Delay          time.Duration
	FailureAfter   int // Zero disables fault injection; positive fails after N IDs.
}
type Config struct {
	Runtime        p.Runtime
	BearerToken    string
	Fake           Fake
	RequestTimeout time.Duration
	WriteTimeout   time.Duration
}
type Gateway struct {
	runtime            p.Runtime
	auth               string
	fake               Fake
	backend            Backend
	timeout, timeWrite time.Duration
	mu                 sync.Mutex
	seen               map[string]struct{}
	active             int
}

func New(c Config) (*Gateway, error) {
	if c.Runtime.Profile != p.Profile {
		return nil, errors.New("fake gateway requires P3 profile")
	}
	return buildGateway(c)
}

func buildGateway(c Config) (*Gateway, error) {
	if c.Runtime.Validate() != nil || !p.ValidAuth(c.BearerToken) || len(c.Fake.OutputTokenIDs) > p.MaxTokens || !p.ValidIDs(c.Fake.OutputTokenIDs, c.Runtime.VocabularySize()) || c.Fake.Delay < 0 || c.Fake.Delay > time.Second || c.Fake.FailureAfter < 0 || c.Fake.FailureAfter > len(c.Fake.OutputTokenIDs) {
		return nil, errors.New("invalid fake gateway configuration")
	}
	if c.RequestTimeout == 0 {
		c.RequestTimeout = 30 * time.Second
	}
	if c.WriteTimeout == 0 {
		c.WriteTimeout = 5 * time.Second
	}
	if c.RequestTimeout <= 0 || c.RequestTimeout > 30*time.Second || c.WriteTimeout <= 0 || c.WriteTimeout > 5*time.Second {
		return nil, errors.New("invalid gateway deadlines")
	}
	fake := c.Fake
	fake.OutputTokenIDs = slices.Clone(fake.OutputTokenIDs)
	return &Gateway{runtime: p.Clone(c.Runtime), auth: c.BearerToken, fake: fake, timeout: c.RequestTimeout, timeWrite: c.WriteTimeout, seen: make(map[string]struct{})}, nil
}

// Server supplies bounded HTTP headers/timeouts and suppresses standard-library
// diagnostics containing untrusted request material. Serve using TLS only;
// shutdown via Server.Shutdown. Reverse proxy TLS offload is not admitted in P3.
func (g *Gateway) Server(addr string) *http.Server {
	// LocalServer is required for local dense sessions. Invalid addresses fail
	// to bind even if a caller accidentally uses this older helper.
	if g.runtime.Profile == p.LocalDenseProfile && !loopbackAddress(addr) {
		addr = "invalid-local-dense-address"
	}
	return &http.Server{Addr: addr, Handler: g, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: g.timeout + g.timeWrite, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 8192, TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12}, ErrorLog: log.New(io.Discard, "", 0)}
}

func loopbackAddress(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	ip := net.ParseIP(host)
	return err == nil && ip != nil && ip.IsLoopback()
}

// LocalServer admits literal loopback listeners only; TLS is still mandatory.
func (g *Gateway) LocalServer(addr string) (*http.Server, error) {
	if g.runtime.Profile != p.LocalDenseProfile || !loopbackAddress(addr) {
		return nil, errors.New("local dense loopback listener required")
	}
	return g.Server(addr), nil
}
func fail(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(struct {
		Error string `json:"error"`
	}{code})
}
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// No request logging, even on rejection. Never echo paths, body, headers or IDs.
	if g.runtime.Profile == p.LocalDenseProfile && !loopbackAddress(r.RemoteAddr) {
		fail(w, 403, "local_only")
		return
	}
	if r.TLS == nil {
		fail(w, 400, "tls_required")
		return
	}
	if len(r.Header.Values("Authorization")) != 1 || subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+g.auth)) != 1 {
		fail(w, 401, "unauthorized")
		return
	}
	if r.URL.RawQuery != "" || r.Header.Get("Last-Event-ID") != "" {
		fail(w, 400, "unsupported_request")
		return
	}
	if r.Method == http.MethodGet && r.URL.Path == "/v1/models/"+g.runtime.ModelID+"/runtime" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(g.runtime)
		return
	}
	if r.Method != http.MethodPost || r.URL.Path != "/v1/generate" {
		fail(w, 404, "not_found")
		return
	}
	if r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Accept") != "text/event-stream" || r.Header.Get("Content-Encoding") != "" {
		fail(w, 400, "unsupported_request")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, p.MaxBody))
	if err != nil {
		fail(w, 413, "body_limit")
		return
	}
	var req p.Request
	if p.DecodeRequest(body, &req) != nil {
		fail(w, 400, "invalid_request")
		return
	}
	if err = req.Validate(g.runtime); err != nil {
		status := 400
		if err.Error() == "binding_mismatch" {
			status = 409
		}
		fail(w, status, err.Error())
		return
	}
	g.mu.Lock()
	_, duplicate := g.seen[req.RequestID]
	if duplicate {
		g.mu.Unlock()
		fail(w, 409, "duplicate_request")
		return
	}
	if len(g.seen) >= MaxRequests || g.active >= MaxConcurrent {
		g.mu.Unlock()
		fail(w, 429, "capacity")
		return
	}
	g.seen[req.RequestID] = struct{}{}
	g.active++
	g.mu.Unlock()
	defer func() { g.mu.Lock(); g.active--; g.mu.Unlock() }()
	g.stream(w, r, req)
}
func (g *Gateway) stream(w http.ResponseWriter, r *http.Request, req p.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), g.timeout)
	defer cancel()
	control := http.NewResponseController(w)
	// Fail before streaming if middleware/transport cannot enforce deadlines.
	if control.SetWriteDeadline(time.Now().Add(g.timeWrite)) != nil {
		fail(w, 500, "unsupported_transport")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	sequence := 0
	send := func(ids []int, reason, code string) bool {
		if r.Context().Err() != nil || ctx.Err() != nil && reason != "error" {
			return false
		}
		sequence++
		data, _ := json.Marshal(p.Event{RequestID: req.RequestID, Sequence: sequence, OutputTokenIDs: ids, FinishReason: reason, Error: code})
		deadline := time.Now().Add(g.timeWrite)
		if limit, ok := ctx.Deadline(); ok && limit.Before(deadline) && reason != "error" {
			deadline = limit
		}
		if control.SetWriteDeadline(deadline) != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "event: token\ndata: %s\n\n", data); err != nil {
			return false
		}
		return control.Flush() == nil
	}
	// Flush headers immediately so a waiting client can cancel before first token.
	if control.Flush() != nil {
		return
	}
	backendError := func() {
		code := "backend_error"
		if ctx.Err() != nil {
			code = "timeout"
		}
		if r.Context().Err() == nil {
			send([]int{}, "error", code)
		}
	}
	stream, err := g.open(ctx, req)
	if stream != nil {
		defer stream.Close()
	}
	if err != nil || stream == nil {
		backendError()
		return
	}
	count := 0
	reason := "backend_done"
	for count < req.Sampling.MaxTokens {
		id, ok, err := stream.Next(ctx)
		if err != nil || ctx.Err() != nil {
			backendError()
			return
		}
		if !ok {
			break
		}
		if id < 0 || id >= g.runtime.VocabularySize() {
			backendError()
			return
		}
		if !send([]int{id}, "", "") {
			return
		}
		count++
		if slices.Contains(g.runtime.MappedSpecialIDs()["eos"], id) {
			reason = "eos"
			break
		}
		if slices.Contains(req.StopTokenIDs, id) {
			reason = "stop_token"
			break
		}
		// Preserve P3's fault injection timing, even on the max_tokens token.
		if g.backend == nil && g.fake.FailureAfter > 0 && count == g.fake.FailureAfter {
			backendError()
			return
		}
	}
	if count == req.Sampling.MaxTokens && reason == "backend_done" {
		reason = "length"
	}
	send([]int{}, reason, "")
}
