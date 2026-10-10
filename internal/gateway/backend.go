package gateway

import (
	"context"
	"errors"
	"slices"
	"time"

	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

// Backend is a trusted, in-process diagnostic adapter. It receives physical IDs
// only. Open and Next MUST honor context cancellation, must not log requests,
// and must not retry. No production dense-artifact wire profile is admitted.
type Backend interface {
	Runtime() p.Runtime
	Open(context.Context, p.Request) (TokenStream, error)
}

// TokenStream is pulled synchronously, once per flushed token. Close releases
// backend request/cache resources, including on disconnect, stop and failure.
// No producer goroutine or unbounded token queue is provided by the gateway.
type TokenStream interface {
	Next(context.Context) (physicalID int, ok bool, err error)
	Close()
}

// NewRelay admits only the existing P3 mock identity contract. This exercises
// relay lifecycle without admitting diagnostic weights to untrusted servers.
func NewRelay(c Config, backend Backend) (*Gateway, error) {
	if backend == nil || len(c.Fake.OutputTokenIDs) != 0 || c.Fake.Delay != 0 || c.Fake.FailureAfter != 0 || !p.Equal(c.Runtime, backend.Runtime()) {
		return nil, errors.New("invalid diagnostic backend configuration")
	}
	g, err := New(c)
	if err != nil {
		return nil, err
	}
	g.backend = backend
	return g, nil
}

type fakeStream struct {
	fake  Fake
	index int
}

func (s *fakeStream) Next(ctx context.Context) (int, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if s.index == len(s.fake.OutputTokenIDs) {
		return 0, false, nil
	}
	if s.fake.Delay > 0 {
		timer := time.NewTimer(s.fake.Delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return 0, false, ctx.Err()
		case <-timer.C:
		}
	}
	id := s.fake.OutputTokenIDs[s.index]
	s.index++
	return id, true, nil
}

func (s *fakeStream) Close() { s.fake.OutputTokenIDs = nil }

func (g *Gateway) open(ctx context.Context, req p.Request) (TokenStream, error) {
	if g.backend != nil {
		// Isolate caller-owned request slices from the relay's stop decisions.
		copy := req
		copy.PromptTokenIDs = slices.Clone(req.PromptTokenIDs)
		copy.StopTokenIDs = slices.Clone(req.StopTokenIDs)
		if req.Sampling.Temperature != nil {
			temperature := *req.Sampling.Temperature
			copy.Sampling.Temperature = &temperature
		}
		return g.backend.Open(ctx, copy)
	}
	fake := g.fake
	fake.OutputTokenIDs = slices.Clone(fake.OutputTokenIDs)
	return &fakeStream{fake: fake}, nil
}
