package gateway

import (
	"context"
	"errors"
	"io"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

type relayMock struct {
	meta                 p.Runtime
	ids                  []int
	block                bool
	failOpen, failNext   bool
	mutate               bool
	opens, pulls, closes atomic.Int64
	request              chan p.Request
	closed               chan struct{}
}

func (b *relayMock) Runtime() p.Runtime { return b.meta }
func (b *relayMock) Open(ctx context.Context, req p.Request) (TokenStream, error) {
	b.opens.Add(1)
	copy := req
	copy.PromptTokenIDs = slices.Clone(req.PromptTokenIDs)
	copy.StopTokenIDs = slices.Clone(req.StopTokenIDs)
	if b.request != nil {
		b.request <- copy
	}
	if b.mutate {
		req.StopTokenIDs[0] = 12
	}
	s := &relayMockStream{backend: b}
	if b.failOpen {
		return s, errors.New(marker)
	}
	return s, nil
}

type relayMockStream struct {
	backend *relayMock
	index   int
}

func (s *relayMockStream) Next(ctx context.Context) (int, bool, error) {
	b := s.backend
	b.pulls.Add(1)
	if b.block {
		<-ctx.Done()
		return 0, false, ctx.Err()
	}
	if b.failNext {
		return 0, false, errors.New(marker)
	}
	if s.index == len(b.ids) {
		return 0, false, nil
	}
	id := b.ids[s.index]
	s.index++
	return id, true, nil
}
func (s *relayMockStream) Close() {
	s.backend.closes.Add(1)
	if s.backend.closed != nil {
		close(s.backend.closed)
	}
}

func TestRelayPullStopErrorsAndRelease(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		ids                               []int
		stop                              []int
		bound                             int
		mutate, failOpen, failNext, block bool
		reason                            string
		pulls                             int64
	}{
		{name: "EOS precedence", ids: []int{3, 9, 4}, stop: []int{9}, bound: 2, reason: "eos", pulls: 2},
		{name: "stop isolated", ids: []int{3, 4}, stop: []int{3}, bound: 8, mutate: true, reason: "stop_token", pulls: 1},
		{name: "length no prefetch", ids: []int{3, 4, 6}, bound: 2, reason: "length", pulls: 2},
		{name: "backend done", ids: []int{3}, bound: 8, reason: "backend_done", pulls: 2},
		{name: "bad physical ID", ids: []int{16}, bound: 8, reason: "error", pulls: 1},
		{name: "negative physical ID", ids: []int{-1}, bound: 8, reason: "error", pulls: 1},
		{name: "open failure", bound: 8, failOpen: true, reason: "error"},
		{name: "pull failure", bound: 8, failNext: true, reason: "error", pulls: 1},
		{name: "timeout", bound: 8, block: true, reason: "error", pulls: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := &relayMock{meta: runtimeFixture(), ids: tc.ids, mutate: tc.mutate, failOpen: tc.failOpen, failNext: tc.failNext, block: tc.block, request: make(chan p.Request, 1)}
			g, err := NewRelay(Config{Runtime: runtimeFixture(), BearerToken: auth, RequestTimeout: 100 * time.Millisecond}, b)
			if err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewUnstartedServer(g)
			ts.Config = g.Server("")
			ts.StartTLS()
			defer ts.Close()
			q := requestFixture()
			q.Sampling.MaxTokens = tc.bound
			q.StopTokenIDs = []int{}
			if tc.stop != nil {
				q.StopTokenIDs = tc.stop
			}
			resp := post(t, ts.Client(), ts.URL, q)
			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil || !strings.Contains(string(body), `"finish_reason":"`+tc.reason+`"`) || strings.Contains(string(body), marker) {
				t.Fatal("invalid relay terminal")
			}
			if tc.block && !strings.Contains(string(body), `"error":"timeout"`) {
				t.Fatal("timeout not classified")
			}
			if b.opens.Load() != 1 || b.pulls.Load() != tc.pulls || b.closes.Load() != 1 {
				t.Fatal("unbounded pull or leaked stream")
			}
			seen := <-b.request
			if !slices.Equal(seen.PromptTokenIDs, q.PromptTokenIDs) || !slices.Equal(seen.StopTokenIDs, q.StopTokenIDs) {
				t.Fatal("physical request altered")
			}
			// A failed/canceled stream cannot reopen through the request ledger.
			duplicate := post(t, ts.Client(), ts.URL, q)
			duplicate.Body.Close()
			if duplicate.StatusCode != 409 || b.opens.Load() != 1 {
				t.Fatal("relay retry admitted")
			}
		})
	}
}

func TestRelayDisconnectClosesBlockedBackend(t *testing.T) {
	b := &relayMock{meta: runtimeFixture(), block: true, closed: make(chan struct{})}
	g, err := NewRelay(Config{Runtime: runtimeFixture(), BearerToken: auth}, b)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewUnstartedServer(g)
	ts.Config = g.Server("")
	ts.StartTLS()
	defer ts.Close()
	resp := post(t, ts.Client(), ts.URL, requestFixture())
	resp.Body.Close()
	select {
	case <-b.closed:
	case <-time.After(2 * time.Second):
		t.Fatal("disconnect leaked backend/cache")
	}
}

func TestRelayRejectsIdentityAndFakeMixture(t *testing.T) {
	c := Config{Runtime: runtimeFixture(), BearerToken: auth}
	b := &relayMock{meta: runtimeFixture()}
	for _, cfg := range []Config{{Runtime: runtimeFixture(), BearerToken: auth, Fake: Fake{OutputTokenIDs: []int{3}}}, {Runtime: runtimeFixture(), BearerToken: auth, Fake: Fake{Delay: time.Millisecond}}} {
		if _, err := NewRelay(cfg, b); err == nil {
			t.Fatal("mixed fake/relay admitted")
		}
	}
	if _, err := NewRelay(c, nil); err == nil {
		t.Fatal("nil backend admitted")
	}
	b.meta.ArtifactID = strings.Repeat("f", 64)
	if _, err := NewRelay(c, b); err == nil {
		t.Fatal("backend identity mismatch admitted")
	}
	c.Runtime.Profile = "p6-dense-production"
	b.meta = c.Runtime
	if _, err := NewRelay(c, b); err == nil {
		t.Fatal("dense production admitted")
	}
}
