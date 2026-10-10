package clientruntime

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/gateway"
	"github.com/yakumioto/AloeForge/internal/model"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
	"github.com/yakumioto/AloeForge/internal/tokenbaseline"
)

const auth = "test-only-authentication-token"
const forbidden = "FORBIDDEN_PLAINTEXT_PROMPT_a19c"

func fixture(t *testing.T, roles ...map[string][]int) (*tokenbaseline.Secret, p.Runtime, Piece) {
	t.Helper()
	special := map[string][]int{"bos": {0}, "eos": {1}, "pad": {2}}
	if len(roles) != 0 {
		special = roles[0]
	}
	secret, err := tokenbaseline.NewSecret(tokenbaseline.Context{Source: model.SourceBinding{InventoryHash: strings.Repeat("a", 64), ContentHash: strings.Repeat("b", 64)}, TokenizerFingerprint: strings.Repeat("c", 64), VocabularySize: 16, HiddenSize: 2, TiedEmbeddings: true, SpecialIDs: special, MethodRevision: artifact.TokenMethod, RNGAlgorithm: artifact.TokenRNG, RNGRevision: artifact.TokenRNGRevision})
	if err != nil {
		t.Fatal(err)
	}
	runtime := p.Runtime{ProtocolVersion: p.Version, Profile: p.Profile, ModelID: "fake-demo", ArtifactID: strings.Repeat("d", 64), Binding: *secret.PublicBinding(), Features: []string{"greedy", "sse", "stop_token_ids"}, MaxPromptTokens: 32, MaxOutputTokens: 32, ContextTokens: 64}
	pieces := map[int][]byte{3: []byte("H"), 4: {0xe4}, 5: {0xb8}, 6: {0xad}, 7: []byte("!"), 8: []byte("ST"), 9: []byte("OP"), 10: []byte("tail"), 11: []byte("x"), 12: []byte(forbidden), 13: {0xff}, 14: []byte("中文"), 15: []byte("STx")}
	piece := func(id int) ([]byte, error) {
		b, ok := pieces[id]
		if !ok {
			return nil, errors.New("local-only-invalid-piece")
		}
		return b, nil
	}
	return secret, runtime, piece
}
func clientFor(t *testing.T, server *httptest.Server, s *tokenbaseline.Secret, r p.Runtime, piece Piece) *Client {
	t.Helper()
	roots := x509.NewCertPool()
	roots.AddCert(server.Certificate())
	c, err := New(Config{URL: server.URL, BearerToken: auth, Expected: r, Secret: s, TokenizerFingerprint: r.Binding.TokenizerFingerprint, Piece: piece, TLS: &tls.Config{RootCAs: roots}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}
func fakeServer(t *testing.T, s *tokenbaseline.Secret, r p.Runtime, ids []int, delay time.Duration) *httptest.Server {
	t.Helper()
	physical, err := s.MapIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gateway.New(gateway.Config{Runtime: r, BearerToken: auth, Fake: gateway.Fake{OutputTokenIDs: physical, Delay: delay}})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewUnstartedServer(g)
	ts.Config = g.Server("")
	ts.StartTLS()
	t.Cleanup(ts.Close)
	return ts
}
func TestTLSRoundTripExactIDsUTF8AndNoPlaintext(t *testing.T) {
	s, r, piece := fixture(t)
	expected := []int{0, 3, 4, 5, 6, 2, 7, 1}
	physical, _ := s.MapIDs(append(slices.Clone(expected), 10))
	g, err := gateway.New(gateway.Config{Runtime: r, BearerToken: auth, Fake: gateway.Fake{OutputTokenIDs: physical}})
	if err != nil {
		t.Fatal(err)
	}
	var logbuf bytes.Buffer
	var captured []byte
	var mu sync.Mutex
	h := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method == "POST" {
			body, _ := io.ReadAll(req.Body)
			mu.Lock()
			captured = slices.Clone(body)
			mu.Unlock()
			req.Body = io.NopCloser(bytes.NewReader(body))
		}
		g.ServeHTTP(w, req)
	})
	ts := httptest.NewUnstartedServer(h)
	ts.Config = g.Server("")
	ts.Config.Handler = h
	ts.Config.ErrorLog = log.New(&logbuf, "", 0)
	ts.StartTLS()
	defer ts.Close()
	c := clientFor(t, ts, s, r, piece)
	var chunks []string
	result, err := c.Generate(context.Background(), []int{12, 3}, Options{MaxTokens: 16, OnText: func(text string) error {
		if !utf8.ValidString(text) {
			t.Error("callback received split rune")
		}
		chunks = append(chunks, text)
		return nil
	}})
	if err != nil || !slices.Equal(result.TokenIDs, expected) || result.Text != "H中!" || result.FinishReason != "eos" || strings.Join(chunks, "") != "H中!" {
		t.Fatalf("round trip: %+v %v", result, err)
	}
	mu.Lock()
	body := slices.Clone(captured)
	mu.Unlock()
	if bytes.Contains(body, []byte(forbidden)) || strings.Contains(logbuf.String(), forbidden) {
		t.Fatal("plaintext reached server boundary/logs")
	}
	var req p.Request
	if json.Unmarshal(body, &req) != nil {
		t.Fatal("invalid wire request")
	}
	want, _ := s.MapIDs([]int{12, 3})
	if !slices.Equal(req.PromptTokenIDs, want) {
		t.Fatal("original prompt IDs were not mapped")
	}
	for _, key := range []string{"permutation", "nonce", "source", "stop_strings", "text", "client_secret"} {
		if bytes.Contains(body, []byte(`"`+key+`"`)) {
			t.Fatalf("private field reached wire: %s", key)
		}
	}
	// No implicit re-encoding API exists. The local piece callback never sees input
	// ID 12 (the forbidden prompt); it decodes only restored output IDs.
	if strings.Contains(result.Text, forbidden) {
		t.Fatal("fake backend used prompt plaintext")
	}
}
func TestStopsSpecialsAndBounds(t *testing.T) {
	cases := []struct {
		name         string
		ids          []int
		options      Options
		wantIDs      []int
		text, reason string
	}{
		{"text across tokens", []int{3, 8, 9, 10, 1}, Options{MaxTokens: 16, StopStrings: []string{"STOP"}}, []int{3, 8, 9}, "H", "text_stop"},
		{"token mapped", []int{3, 7, 10, 1}, Options{MaxTokens: 16, StopTokenIDs: []int{7}}, []int{3, 7}, "H", "stop_token"},
		{"length", []int{3, 7, 10}, Options{MaxTokens: 2}, []int{3, 7}, "H!", "length"},
		{"backend ends", []int{3, 7}, Options{MaxTokens: 16}, []int{3, 7}, "H!", "backend_done"},
		{"unmatched stop prefix flushed", []int{3, 8}, Options{MaxTokens: 16, StopStrings: []string{"STOP"}}, []int{3, 8}, "HST", "backend_done"},
		{"unicode stop", []int{3, 4, 5, 6, 7}, Options{MaxTokens: 16, StopStrings: []string{"中"}}, []int{3, 4, 5, 6}, "H", "text_stop"},
		{"overlapping earliest stop", []int{3, 8, 9, 10}, Options{MaxTokens: 16, StopStrings: []string{"STOPtail", "STOP"}}, []int{3, 8, 9}, "H", "text_stop"},
		{"BOS PAD suppression", []int{0, 2, 3, 0, 2}, Options{MaxTokens: 16}, []int{0, 2, 3, 0, 2}, "H", "backend_done"},
		{"EOS wins stop", []int{3, 1, 7}, Options{MaxTokens: 16, StopTokenIDs: []int{1}}, []int{3, 1}, "H", "eos"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, r, piece := fixture(t)
			server := fakeServer(t, s, r, tc.ids, 0)
			c := clientFor(t, server, s, r, piece)
			var text strings.Builder
			tc.options.OnText = func(s string) error { text.WriteString(s); return nil }
			got, err := c.Generate(context.Background(), []int{12}, tc.options)
			if err != nil || !slices.Equal(got.TokenIDs, tc.wantIDs) || got.Text != tc.text || text.String() != tc.text || got.FinishReason != tc.reason {
				t.Fatalf("stop: %+v %v", got, err)
			}
		})
	}
}
func TestRejectUnsafeClientConfiguration(t *testing.T) {
	s, r, piece := fixture(t)
	server := fakeServer(t, s, r, []int{3}, 0)
	valid := Config{URL: server.URL, BearerToken: auth, Expected: r, Secret: s, TokenizerFingerprint: r.Binding.TokenizerFingerprint, Piece: piece}
	cases := map[string]func(*Config){"http": func(c *Config) { c.URL = "http://localhost" }, "userinfo": func(c *Config) { c.URL = "https://user:pass@localhost" }, "query": func(c *Config) { c.URL += "?prompt=" + forbidden }, "insecure TLS": func(c *Config) { c.TLS = &tls.Config{InsecureSkipVerify: true} }, "wrong tokenizer": func(c *Config) { c.TokenizerFingerprint = strings.Repeat("e", 64) }, "wrong binding": func(c *Config) { c.Expected.Binding.BindingID = strings.Repeat("e", 64) }, "wrong specials": func(c *Config) {
		c.Expected.Binding.MappedSpecialIDs["eos"] = []int{(c.Expected.Binding.MappedSpecialIDs["eos"][0] + 1) % c.Expected.Binding.VocabularySize}
	}, "nil secret": func(c *Config) { c.Secret = nil }, "nil piece": func(c *Config) { c.Piece = nil }, "auth": func(c *Config) { c.BearerToken = "short" }}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := valid
			cfg.Expected = p.Clone(r)
			mutate(&cfg)
			if c, err := New(cfg); err == nil {
				c.Close()
				t.Fatal("unsafe config accepted")
			}
		})
	}
	// A root pool that does not trust the TLS server must fail; no implicit bypass.
	c, err := New(valid)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if c.Handshake(context.Background()) == nil {
		t.Fatal("untrusted server certificate accepted")
	}
}
func TestHandshakeMismatchPreventsGenerationAndRedirect(t *testing.T) {
	for _, field := range []string{"artifact", "tokenizer", "binding", "method", "specials", "features", "bounds", "protocol", "unknown", "redirect"} {
		t.Run(field, func(t *testing.T) {
			s, r, piece := fixture(t)
			var posts atomic.Int32
			ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.Method == "POST" {
					posts.Add(1)
				}
				got := p.Clone(r)
				switch field {
				case "artifact":
					got.ArtifactID = strings.Repeat("e", 64)
				case "tokenizer":
					got.Binding.TokenizerFingerprint = strings.Repeat("e", 64)
				case "binding":
					got.Binding.BindingID = strings.Repeat("e", 64)
				case "method":
					got.Binding.MethodRevision = "future"
				case "specials":
					got.Binding.MappedSpecialIDs["eos"] = []int{(got.Binding.MappedSpecialIDs["eos"][0] + 1) % got.Binding.VocabularySize}
				case "features":
					got.Features = []string{"plaintext"}
				case "bounds":
					got.MaxPromptTokens--
				case "protocol":
					got.ProtocolVersion++
				case "redirect":
					w.Header().Set("Location", "http://invalid.example/"+forbidden)
					w.WriteHeader(302)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				data, _ := json.Marshal(got)
				if field == "unknown" {
					data = append(data[:len(data)-1], []byte(`,"text":"`+forbidden+`"}`)...)
				}
				_, _ = w.Write(data)
			}))
			defer ts.Close()
			c := clientFor(t, ts, s, r, piece)
			_, err := c.Generate(context.Background(), []int{12}, Options{MaxTokens: 16})
			if err == nil || posts.Load() != 0 || strings.Contains(err.Error(), forbidden) {
				t.Fatal("handshake failed open or echoed text")
			}
		})
	}
}
func TestPartialStreamsAreExplicitAndNeverRetried(t *testing.T) {
	cases := []string{"EOF", "sequence", "request-id", "range", "unknown-field", "oversized", "duplicate-key", "truncated-boundary", "early-eos", "extra-after-eos", "error", "wrong-type"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			s, r, piece := fixture(t)
			physical, _ := s.MapIDs([]int{3, 1})
			var posts atomic.Int32
			ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.Method == "GET" {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(r)
					return
				}
				posts.Add(1)
				var q p.Request
				_ = json.NewDecoder(req.Body).Decode(&q)
				w.Header().Set("Content-Type", "text/event-stream")
				send := func(e p.Event) { data, _ := json.Marshal(e); _, _ = fmt.Fprintf(w, "event: token\ndata: %s\n\n", data) }
				send(p.Event{RequestID: q.RequestID, Sequence: 1, OutputTokenIDs: physical[:1]})
				event := p.Event{RequestID: q.RequestID, Sequence: 2, OutputTokenIDs: []int{}, FinishReason: "backend_done"}
				switch name {
				case "EOF":
					return
				case "sequence":
					event.Sequence = 1
				case "request-id":
					event.RequestID = strings.Repeat("0", 32)
				case "range":
					event.OutputTokenIDs = []int{16}
					event.FinishReason = ""
				case "early-eos":
					event.FinishReason = "eos"
				case "extra-after-eos":
					event.OutputTokenIDs = physical[1:]
					event.FinishReason = ""
					send(event)
					event.Sequence = 3
					event.OutputTokenIDs = physical[:1]
				case "error":
					event.FinishReason = "error"
					event.Error = "backend_error"
				case "wrong-type":
					_, _ = fmt.Fprint(w, "data: {}\n\n")
					return
				case "oversized":
					_, _ = fmt.Fprintf(w, "event: token\ndata: %s\n\n", strings.Repeat("x", p.MaxEvent+1))
					return
				case "unknown-field", "duplicate-key", "truncated-boundary":
					data, _ := json.Marshal(event)
					if name == "unknown-field" {
						data = append(data[:len(data)-1], []byte(`,"text":"`+forbidden+`"}`)...)
					}
					if name == "duplicate-key" {
						data = append(data[:len(data)-1], []byte(`,"sequence":2}`)...)
					}
					end := "\n\n"
					if name == "truncated-boundary" {
						end = "\n"
					}
					_, _ = fmt.Fprintf(w, "event: token\ndata: %s%s", data, end)
					return
				}
				send(event)
			}))
			defer ts.Close()
			c := clientFor(t, ts, s, r, piece)
			got, err := c.Generate(context.Background(), []int{12}, Options{MaxTokens: 16})
			var partial *PartialError
			if !errors.As(err, &partial) || partial.ReceivedTokens != len(got.TokenIDs) || got.Text != "H" || posts.Load() != 1 || strings.Contains(err.Error(), forbidden) {
				t.Fatalf("partial result: %+v %v", got, err)
			}
		})
	}
}
func TestCancellationLocalErrorsAndReplay(t *testing.T) {
	s, r, piece := fixture(t)
	var done = make(chan struct{}, 4)
	physical, _ := s.MapIDs([]int{3, 7, 10, 1})
	g, err := gateway.New(gateway.Config{Runtime: r, BearerToken: auth, Fake: gateway.Fake{OutputTokenIDs: physical, Delay: 100 * time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	h := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		g.ServeHTTP(w, req)
		if req.Method == "POST" {
			done <- struct{}{}
		}
	})
	ts := httptest.NewUnstartedServer(h)
	ts.Config = g.Server("")
	ts.Config.Handler = h
	ts.StartTLS()
	defer ts.Close()
	c := clientFor(t, ts, s, r, piece)
	ctx, cancel := context.WithCancel(context.Background())
	got, err := c.Generate(ctx, []int{12}, Options{RequestID: strings.Repeat("a", 32), MaxTokens: 16, OnText: func(string) error { cancel(); return nil }})
	var partial *PartialError
	if !errors.As(err, &partial) || got.Text != "H" {
		t.Fatalf("cancel: %+v %v", got, err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("backend did not observe cancellation")
	}
	_, err = c.Generate(context.Background(), []int{12}, Options{RequestID: strings.Repeat("a", 32), MaxTokens: 16})
	if err == nil {
		t.Fatal("replay accepted")
	}
	got, err = c.Generate(context.Background(), []int{12}, Options{MaxTokens: 16, OnText: func(string) error { return errors.New(forbidden) }})
	if !errors.As(err, &partial) || partial.Code != "consumer_failed" || strings.Contains(err.Error(), forbidden) {
		t.Fatalf("consumer error: %+v %v", got, err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("consumer stop did not cancel backend")
	}
}
func TestInvalidPiecesAndLocalOptions(t *testing.T) {
	for _, ids := range [][]int{{13}, {4}, {3, 4, 1}} {
		t.Run(fmt.Sprint(ids), func(t *testing.T) {
			s, r, piece := fixture(t)
			ts := fakeServer(t, s, r, ids, 0)
			c := clientFor(t, ts, s, r, piece)
			_, err := c.Generate(context.Background(), []int{12}, Options{MaxTokens: 16})
			var partial *PartialError
			if !errors.As(err, &partial) || partial.Code != "invalid_utf8" {
				t.Fatal("invalid UTF-8 accepted", err)
			}
		})
	}
	s, r, piece := fixture(t)
	ts := fakeServer(t, s, r, []int{3}, 0)
	c := clientFor(t, ts, s, r, piece)
	for _, opts := range []Options{{MaxTokens: 0}, {MaxTokens: 33}, {MaxTokens: 16, RequestID: forbidden}, {MaxTokens: 16, StopStrings: []string{""}}, {MaxTokens: 16, StopStrings: []string{string([]byte{0xff})}}, {MaxTokens: 16, StopStrings: []string{strings.Repeat("x", 257)}}, {MaxTokens: 16, StopTokenIDs: []int{7, 7}}, {MaxTokens: 16, StopTokenIDs: []int{16}}} {
		if _, err := c.Generate(context.Background(), []int{12}, opts); err == nil {
			t.Fatal("invalid options accepted")
		}
	}
	for _, prompt := range [][]int{nil, {-1}, {16}, make([]int, 33)} {
		if _, err := c.Generate(context.Background(), prompt, Options{MaxTokens: 16}); err == nil {
			t.Fatal("invalid prompt accepted")
		}
	}
	bad := clientFor(t, ts, s, r, func(int) ([]byte, error) { return bytes.Repeat([]byte("x"), MaxPieceBytes+1), nil })
	_, err := bad.Generate(context.Background(), []int{12}, Options{MaxTokens: 16})
	var partial *PartialError
	if !errors.As(err, &partial) || partial.Code != "invalid_piece" {
		t.Fatal("oversized piece accepted")
	}
}
func TestFakeBackendErrorAndTimeout(t *testing.T) {
	for _, name := range []string{"backend_error", "timeout"} {
		t.Run(name, func(t *testing.T) {
			s, r, piece := fixture(t)
			physical, _ := s.MapIDs([]int{3, 7})
			cfg := gateway.Config{Runtime: r, BearerToken: auth, Fake: gateway.Fake{OutputTokenIDs: physical, FailureAfter: 1}}
			if name == "timeout" {
				cfg.Fake.FailureAfter = 0
				cfg.Fake.Delay = 200 * time.Millisecond
				cfg.RequestTimeout = 20 * time.Millisecond
			}
			g, err := gateway.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewUnstartedServer(g)
			ts.Config = g.Server("")
			ts.StartTLS()
			defer ts.Close()
			c := clientFor(t, ts, s, r, piece)
			_, err = c.Generate(context.Background(), []int{12}, Options{MaxTokens: 16})
			var partial *PartialError
			if !errors.As(err, &partial) || partial.Code != name {
				t.Fatal("explicit error missing", err)
			}
		})
	}
}

func TestSpecialRoleAliasesAndPromptIndependentFake(t *testing.T) {
	s, r, piece := fixture(t, map[string][]int{"bos": {1}, "eos": {1}, "pad": {1}})
	ts := fakeServer(t, s, r, []int{3, 1, 7}, 0)
	c := clientFor(t, ts, s, r, piece)
	for _, prompt := range [][]int{{12}, {3, 7, 14}} {
		got, err := c.Generate(context.Background(), prompt, Options{MaxTokens: 16})
		if err != nil || got.FinishReason != "eos" || got.Text != "H" || !slices.Equal(got.TokenIDs, []int{3, 1}) {
			t.Fatalf("aliased EOS or prompt-dependent output: %+v %v", got, err)
		}
	}
}
func TestCancelBeforeFirstTokenAndLocalTextStopPropagates(t *testing.T) {
	for _, name := range []string{"cancel_before_first", "local_text_stop"} {
		t.Run(name, func(t *testing.T) {
			s, r, piece := fixture(t)
			physical, _ := s.MapIDs([]int{8, 9, 10, 1})
			g, err := gateway.New(gateway.Config{Runtime: r, BearerToken: auth, Fake: gateway.Fake{OutputTokenIDs: physical, Delay: 100 * time.Millisecond}})
			if err != nil {
				t.Fatal(err)
			}
			started := make(chan struct{})
			done := make(chan struct{})
			h := http.HandlerFunc(func(w http.ResponseWriter, q *http.Request) {
				if q.Method == "POST" {
					close(started)
					defer close(done)
				}
				g.ServeHTTP(w, q)
			})
			ts := httptest.NewUnstartedServer(h)
			ts.Config = g.Server("")
			ts.Config.Handler = h
			ts.StartTLS()
			defer ts.Close()
			c := clientFor(t, ts, s, r, piece)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if name == "cancel_before_first" {
				go func() { <-started; cancel() }()
			}
			got, err := c.Generate(ctx, []int{12}, Options{MaxTokens: 16, StopStrings: []string{"STOP"}})
			if name == "cancel_before_first" {
				var partial *PartialError
				if !errors.As(err, &partial) || partial.ReceivedTokens != 0 {
					t.Fatalf("ambiguous submission was not explicit: %+v %v", got, err)
				}
			} else if err != nil || got.FinishReason != "text_stop" || got.Text != "" {
				t.Fatalf("local stop: %+v %v", got, err)
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("client did not cancel fake generator")
			}
		})
	}
}
