package gateway

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yakumioto/AloeForge/internal/artifact"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

const auth = "gateway-test-only-auth-token"
const marker = "FORBIDDEN_SERVER_LOG_MARKER_a21b"

func runtimeFixture() p.Runtime {
	return p.Runtime{ProtocolVersion: 1, Profile: p.Profile, ModelID: "fake", ArtifactID: strings.Repeat("a", 64), Binding: artifact.TokenBinding{SchemaVersion: 1, Profile: artifact.TokenProfile, BindingID: strings.Repeat("b", 64), TokenizerFingerprint: strings.Repeat("c", 64), MethodRevision: artifact.TokenMethod, RNGAlgorithm: artifact.TokenRNG, RNGRevision: artifact.TokenRNGRevision, VocabularySize: 16, HiddenSize: 2, TiedEmbeddings: true, MappedSpecialIDs: map[string][]int{"bos": {7}, "eos": {9}, "pad": {5}}}, Features: []string{"greedy", "sse", "stop_token_ids"}, MaxPromptTokens: p.MaxTokens, MaxOutputTokens: p.MaxTokens, ContextTokens: 2 * p.MaxTokens}
}
func requestFixture() p.Request {
	r := runtimeFixture()
	temp := 0.0
	return p.Request{ProtocolVersion: 1, ModelID: r.ModelID, ArtifactID: r.ArtifactID, BindingID: r.Binding.BindingID, RequestID: strings.Repeat("a", 32), PromptTokenIDs: []int{7, 8, 3}, StopTokenIDs: []int{}, Sampling: p.Sampling{Temperature: &temp, MaxTokens: 8}, Stream: true}
}
func newGateway(t *testing.T, fake Fake) *Gateway {
	t.Helper()
	g, err := New(Config{Runtime: runtimeFixture(), BearerToken: auth, Fake: fake})
	if err != nil {
		t.Fatal(err)
	}
	return g
}
func post(t *testing.T, c *http.Client, base string, q p.Request) *http.Response {
	t.Helper()
	data, _ := json.Marshal(q)
	r, _ := http.NewRequest("POST", base+"/v1/generate", bytes.NewReader(data))
	r.Header.Set("Authorization", "Bearer "+auth)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Accept", "text/event-stream")
	resp, err := c.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}
func TestRejectedRequestsAreStaticAndNeverLogPlaintext(t *testing.T) {
	g := newGateway(t, Fake{OutputTokenIDs: []int{3, 9}})
	var logs bytes.Buffer
	ts := httptest.NewUnstartedServer(g)
	ts.Config = g.Server("")
	ts.Config.ErrorLog = log.New(&logs, "", 0)
	ts.StartTLS()
	defer ts.Close()
	base := requestFixture()
	data, _ := json.Marshal(base)
	cases := []struct {
		name         string
		method, path string
		body         []byte
		headers      map[string]string
		status       int
	}{
		{"auth missing", "POST", "/v1/generate", data, map[string]string{"Authorization": ""}, 401},
		{"auth invalid", "POST", "/v1/generate", data, map[string]string{"Authorization": "Bearer " + marker}, 401},
		{"plain text field", "POST", "/v1/generate", append(slices.Clone(data[:len(data)-1]), []byte(`,"prompt":"`+marker+`"}`)...), nil, 400},
		{"text stops", "POST", "/v1/generate", append(slices.Clone(data[:len(data)-1]), []byte(`,"stop_strings":["`+marker+`"]}`)...), nil, 400},
		{"duplicate key", "POST", "/v1/generate", append(slices.Clone(data[:len(data)-1]), []byte(`,"request_id":"`+marker+`"}`)...), nil, 400},
		{"bad JSON", "POST", "/v1/generate", []byte(marker), nil, 400},
		{"oversized body", "POST", "/v1/generate", bytes.Repeat([]byte(marker), p.MaxBody/len(marker)+1), nil, 413},
		{"bad content type", "POST", "/v1/generate", data, map[string]string{"Content-Type": marker}, 400},
		{"compressed", "POST", "/v1/generate", data, map[string]string{"Content-Encoding": "gzip"}, 400},
		{"text accept", "POST", "/v1/generate", data, map[string]string{"Accept": "application/json"}, 400},
		{"last event", "POST", "/v1/generate", data, map[string]string{"Last-Event-ID": marker}, 400},
		{"query prompt", "POST", "/v1/generate?prompt=" + marker, data, nil, 400},
		{"bad route", "POST", "/" + marker, data, nil, 404},
		{"bad method", "PUT", "/v1/generate", data, nil, 404},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(tc.method, ts.URL+tc.path, bytes.NewReader(tc.body))
			req.Header.Set("Authorization", "Bearer "+auth)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			resp, err := ts.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode != tc.status || bytes.Contains(b, []byte(marker)) || strings.Contains(logs.String(), marker) || bytes.Contains(b, []byte(auth)) {
				t.Fatalf("unsafe rejection: %d %s", resp.StatusCode, b)
			}
		})
	}
	for name, mutate := range map[string]func(*p.Request){"binding": func(q *p.Request) { q.BindingID = strings.Repeat("e", 64) }, "artifact": func(q *p.Request) { q.ArtifactID = strings.Repeat("e", 64) }, "model": func(q *p.Request) { q.ModelID = "other" }, "protocol": func(q *p.Request) { q.ProtocolVersion++ }, "temperature": func(q *p.Request) { x := 1.0; q.Sampling.Temperature = &x }, "null temperature": func(q *p.Request) { q.Sampling.Temperature = nil }, "max tokens": func(q *p.Request) { q.Sampling.MaxTokens = p.MaxTokens + 1 }, "context": func(q *p.Request) {
		q.PromptTokenIDs = make([]int, p.MaxTokens)
		q.Sampling.MaxTokens = p.MaxTokens + 1
	}, "null stops": func(q *p.Request) { q.StopTokenIDs = nil }, "too many stops": func(q *p.Request) { q.StopTokenIDs = make([]int, 17) }, "duplicate stops": func(q *p.Request) { q.StopTokenIDs = []int{3, 3} }, "negative ID": func(q *p.Request) { q.PromptTokenIDs = []int{-1} }, "large ID": func(q *p.Request) { q.PromptTokenIDs = []int{16} }, "request ID": func(q *p.Request) { q.RequestID = marker }, "nonstream": func(q *p.Request) { q.Stream = false }, "empty prompt": func(q *p.Request) { q.PromptTokenIDs = []int{} }} {
		t.Run(name, func(t *testing.T) {
			q := requestFixture()
			mutate(&q)
			resp := post(t, ts.Client(), ts.URL, q)
			defer resp.Body.Close()
			if resp.StatusCode != 400 && resp.StatusCode != 409 {
				t.Fatal("invalid request admitted", resp.StatusCode)
			}
		})
	}
	// Double authorization fields are rejected even when the first is valid.
	req, _ := http.NewRequest("POST", ts.URL+"/v1/generate", bytes.NewReader(data))
	req.Header.Add("Authorization", "Bearer "+auth)
	req.Header.Add("Authorization", marker)
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatal("ambiguous auth accepted")
	}
	if logs.Len() != 0 {
		t.Fatal("gateway unexpectedly logged request diagnostics")
	}
}
func TestTLSRequiredAndUnsupportedWriter(t *testing.T) {
	g := newGateway(t, Fake{OutputTokenIDs: []int{3}})
	q := requestFixture()
	data, _ := json.Marshal(q)
	req := httptest.NewRequest("POST", "http://localhost/v1/generate", bytes.NewReader(data))
	req.Header.Set("Authorization", "Bearer "+auth)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	w := httptest.NewRecorder()
	g.ServeHTTP(w, req)
	if w.Code != 400 || !strings.Contains(w.Body.String(), "tls_required") {
		t.Fatal("plaintext transport accepted")
	}
	req.TLS = &tls.ConnectionState{}
	w = httptest.NewRecorder()
	g.ServeHTTP(w, req)
	if w.Code != 500 || !strings.Contains(w.Body.String(), "unsupported_transport") {
		t.Fatal("unbounded transport accepted")
	}
}
func TestBoundedLedgerAndConcurrency(t *testing.T) {
	g := newGateway(t, Fake{OutputTokenIDs: []int{3}, Delay: time.Second})
	done := make(chan struct{}, MaxConcurrent)
	var tracking atomic.Bool
	tracking.Store(true)
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.ServeHTTP(w, r)
		if r.Method == "POST" && tracking.Load() {
			done <- struct{}{}
		}
	})
	ts := httptest.NewUnstartedServer(h)
	ts.Config = g.Server("")
	ts.Config.Handler = h
	ts.StartTLS()
	defer ts.Close()
	var cancels []context.CancelFunc
	var responses []*http.Response
	for i := 0; i < MaxConcurrent; i++ {
		q := requestFixture()
		q.RequestID = fmt.Sprintf("%032x", i+1)
		data, _ := json.Marshal(q)
		ctx, cancel := context.WithCancel(context.Background())
		cancels = append(cancels, cancel)
		req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/generate", bytes.NewReader(data))
		req.Header.Set("Authorization", "Bearer "+auth)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		resp, err := ts.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 200 {
			t.Fatal("concurrency unexpectedly rejected")
		}
		responses = append(responses, resp)
	}
	q := requestFixture()
	q.RequestID = fmt.Sprintf("%032x", 1)
	resp := post(t, ts.Client(), ts.URL, q)
	resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Fatal("concurrent duplicate admitted")
	}
	q.RequestID = fmt.Sprintf("%032x", MaxConcurrent+1)
	resp = post(t, ts.Client(), ts.URL, q)
	resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatal("inflight limit not enforced")
	}
	for i, cancel := range cancels {
		cancel()
		responses[i].Body.Close()
	}
	// Includes the two rejected requests. All active fake generators must leave.
	for i := 0; i < MaxConcurrent+2; i++ {
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("cancellation left active generator")
		}
	}
	tracking.Store(false)
	g.mu.Lock()
	if g.active != 0 {
		t.Error("active requests leaked")
	}
	g.fake.Delay = 0
	g.mu.Unlock()
	// Exercise the lifetime ledger through actual accepted requests, not eviction.
	for i := MaxConcurrent; i < MaxRequests; i++ {
		q.RequestID = fmt.Sprintf("%032x", i+1)
		resp = post(t, ts.Client(), ts.URL, q)
		_, err := io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if err != nil || resp.StatusCode != 200 {
			t.Fatal("ledger filled too early", resp.StatusCode, err)
		}
	}
	q.RequestID = fmt.Sprintf("%032x", MaxRequests+1)
	resp = post(t, ts.Client(), ts.URL, q)
	resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatal("request ledger grew without bound")
	}
	q.RequestID = fmt.Sprintf("%032x", 1)
	resp = post(t, ts.Client(), ts.URL, q)
	resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Fatal("canceled request was evicted and replayed")
	}
}

type smallBufferListener struct{ net.Listener }

type countingWriter struct {
	http.ResponseWriter
	writes *atomic.Int64
}

func (w countingWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w countingWriter) Write(data []byte) (int, error) {
	n, err := w.ResponseWriter.Write(data)
	if err == nil {
		w.writes.Add(1)
	}
	return n, err
}

func (l smallBufferListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err == nil {
		if tcp, ok := c.(*net.TCPConn); ok {
			_ = tcp.SetWriteBuffer(1024)
		}
	}
	return c, err
}
func TestSlowTLSReaderHasBoundedBackpressure(t *testing.T) {
	r := runtimeFixture()
	ids := make([]int, p.MaxTokens)
	for i := range ids {
		ids[i] = 3
	}
	g, err := New(Config{Runtime: r, BearerToken: auth, Fake: Fake{OutputTokenIDs: ids}, WriteTimeout: 25 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var writes atomic.Int64
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.ServeHTTP(countingWriter{w, &writes}, r)
		close(done)
	})
	ts := httptest.NewUnstartedServer(h)
	ts.Config = g.Server("")
	ts.Config.Handler = h
	ts.Listener = smallBufferListener{ts.Listener}
	ts.StartTLS()
	defer ts.Close()
	roots := x509.NewCertPool()
	roots.AddCert(ts.Certificate())
	conn, err := tls.Dial("tcp", ts.Listener.Addr().String(), &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if tcp, ok := conn.NetConn().(*net.TCPConn); ok {
		_ = tcp.SetReadBuffer(1024)
	}
	q := requestFixture()
	q.Sampling.MaxTokens = p.MaxTokens
	data, _ := json.Marshal(q)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/generate", bytes.NewReader(data))
	req.Header.Set("Authorization", "Bearer "+auth)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if req.Write(conn) != nil {
		t.Fatal("request write failed")
	}
	resp, err := http.ReadResponse(bufio.NewReader(conn), req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatal("stream rejected")
	}
	// Deliberately never drain body. The write deadline must release the handler.
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("slow reader blocked gateway indefinitely")
	}
	if writes.Load() >= p.MaxTokens {
		t.Fatal("slow-reader test did not exercise a blocked write")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.active != 0 {
		t.Fatal("slow reader kept inference slot")
	}
}
func TestImmutableConfigAndInvalidFake(t *testing.T) {
	c := Config{Runtime: runtimeFixture(), BearerToken: auth, Fake: Fake{OutputTokenIDs: []int{3, 9}}}
	g, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	c.Runtime.Binding.MappedSpecialIDs["eos"][0] = 3
	c.Fake.OutputTokenIDs[0] = 9
	if g.runtime.Binding.MappedSpecialIDs["eos"][0] != 9 || g.fake.OutputTokenIDs[0] != 3 {
		t.Fatal("caller can mutate running gateway")
	}
	for _, f := range []Fake{{OutputTokenIDs: []int{16}}, {OutputTokenIDs: make([]int, p.MaxTokens+1)}, {Delay: -1}, {Delay: 2 * time.Second}, {FailureAfter: -1}, {FailureAfter: 1}} {
		if _, err := New(Config{Runtime: runtimeFixture(), BearerToken: auth, Fake: f}); err == nil {
			t.Fatal("invalid fake admitted")
		}
	}
	for _, cfg := range []Config{{Runtime: runtimeFixture(), BearerToken: "short"}, {Runtime: runtimeFixture(), BearerToken: auth, RequestTimeout: time.Minute}, {Runtime: runtimeFixture(), BearerToken: auth, WriteTimeout: time.Minute}} {
		if _, err := New(cfg); err == nil {
			t.Fatal("invalid limits admitted")
		}
	}
	server := g.Server("")
	if server.ReadHeaderTimeout == 0 || server.ReadTimeout == 0 || server.IdleTimeout == 0 || server.MaxHeaderBytes > 8192 {
		t.Fatal("server defaults unbounded")
	}
}

// Independent callers can share a gateway; the race check exercises ledger and
// streaming ownership without sharing client-side output buffers.
func TestConcurrentUniqueRequests(t *testing.T) {
	g := newGateway(t, Fake{OutputTokenIDs: []int{3, 9}})
	ts := httptest.NewUnstartedServer(g)
	ts.Config = g.Server("")
	ts.StartTLS()
	defer ts.Close()
	var wg sync.WaitGroup
	for i := 0; i < MaxConcurrent; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			q := requestFixture()
			q.RequestID = fmt.Sprintf("%032x", i)
			resp := post(t, ts.Client(), ts.URL, q)
			_, err := io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if err != nil || resp.StatusCode != 200 {
				t.Error("concurrent stream failed")
			}
		}(i)
	}
	wg.Wait()
}
