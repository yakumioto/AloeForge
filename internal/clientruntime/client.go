// Package clientruntime is the trusted client for P3 and the local P6 reference.
// The application supplies original IDs and a local tokenizer's byte pieces;
// neither plaintext nor a Client Secret is sent to the gateway.
package clientruntime

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"time"
	"unicode/utf8"

	"github.com/yakumioto/AloeForge/internal/dense"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
	"github.com/yakumioto/AloeForge/internal/tokenbaseline"
)

const MaxPieceBytes = 256
const MaxStopStrings = 8
const MaxStopBytes = 256

// Piece is a trusted local tokenizer's exact byte contribution for an original
// ID. Pieces may split UTF-8 runes. No encode/decode round-trip API exists here.
type Piece func(originalID int) ([]byte, error)
type Config struct {
	URL                  string
	BearerToken          string
	Expected             p.Runtime // From the trusted compiler/configuration channel.
	Secret               *tokenbaseline.Secret
	DenseSecret          *dense.Secret // Only for the trusted local dense profile.
	TokenizerFingerprint string        // Actual local tokenizer's independently checked hash.
	Piece                Piece
	TLS                  *tls.Config
}
type Client struct {
	base, auth string
	expected   p.Runtime
	secret     tokenMapper
	piece      Piece
	special    map[string][]int
	http       *http.Client
}

type tokenMapper interface {
	MapIDs([]int) ([]int, error)
	RestoreIDs([]int) ([]int, error)
}

func New(c Config) (*Client, error) {
	u, err := url.Parse(c.URL)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Path != "" || u.Opaque != "" || !p.ValidAuth(c.BearerToken) || c.Expected.Validate() != nil || c.Piece == nil || c.TokenizerFingerprint != c.Expected.TokenizerFingerprint() {
		return nil, errors.New("invalid trusted client configuration")
	}
	var secret tokenMapper
	if c.Expected.Profile == p.LocalDenseProfile {
		ip := net.ParseIP(u.Hostname())
		if ip == nil || !ip.IsLoopback() || c.Secret != nil || c.DenseSecret == nil || c.DenseSecret.MatchesContract(*c.Expected.Dense) != nil {
			return nil, errors.New("invalid local dense client configuration")
		}
		secret = c.DenseSecret
	} else {
		if c.DenseSecret != nil || c.Secret == nil || !p.Equal(c.Secret.PublicBinding(), c.Expected.Binding) {
			return nil, errors.New("invalid trusted client configuration")
		}
		secret = c.Secret
	}
	tc := &tls.Config{MinVersion: tls.VersionTLS12}
	if c.TLS != nil {
		tc = c.TLS.Clone()
		if tc.InsecureSkipVerify {
			return nil, errors.New("TLS verification required")
		}
		if tc.MinVersion < tls.VersionTLS12 {
			tc.MinVersion = tls.VersionTLS12
		}
	}
	transport := &http.Transport{TLSClientConfig: tc, Proxy: http.ProxyFromEnvironment, TLSHandshakeTimeout: 5 * time.Second, ResponseHeaderTimeout: 10 * time.Second, MaxResponseHeaderBytes: 8192, MaxConnsPerHost: 8, MaxIdleConnsPerHost: 8, IdleConnTimeout: 30 * time.Second, DisableCompression: true}
	if c.Expected.Profile == p.LocalDenseProfile {
		transport.Proxy = nil
	}
	h := &http.Client{Transport: transport, Timeout: 35 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	special := make(map[string][]int)
	for role, ids := range c.Expected.MappedSpecialIDs() {
		special[role], err = secret.RestoreIDs(ids)
		if err != nil {
			return nil, errors.New("invalid special binding")
		}
	}
	return &Client{base: c.URL, auth: c.BearerToken, expected: p.Clone(c.Expected), secret: secret, piece: c.Piece, special: special, http: h}, nil
}
func (c *Client) Close() { c.http.CloseIdleConnections() }
func (c *Client) request(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	r, err := http.NewRequestWithContext(ctx, method, c.base+path, bytes.NewReader(body))
	if err != nil {
		return nil, errors.New("request_failed")
	}
	r.Header.Set("Authorization", "Bearer "+c.auth)
	if method == http.MethodPost {
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Accept", "text/event-stream")
	}
	resp, err := c.http.Do(r)
	if err != nil {
		return nil, errors.New("transport_failed")
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, errors.New("gateway_rejected")
	}
	return resp, nil
}

// Handshake refuses all metadata mismatches, including features and bounds.
func (c *Client) Handshake(ctx context.Context) error {
	resp, err := c.request(ctx, http.MethodGet, "/v1/models/"+c.expected.ModelID+"/runtime", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Type") != "application/json" {
		return errors.New("invalid_handshake")
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, p.MaxBody+1))
	if err != nil || len(data) > p.MaxBody {
		return errors.New("invalid_handshake")
	}
	var got p.Runtime
	if p.DecodeRuntime(data, &got) != nil || got.Validate() != nil || !p.Equal(got, c.expected) {
		return errors.New("binding_mismatch")
	}
	return nil
}

type Options struct {
	RequestID    string // Empty generates 128 random bits. No automatic retries.
	MaxTokens    int
	StopTokenIDs []int              // Original coordinates; mapped before transmission.
	StopStrings  []string           // Plaintext, retained only on the trusted client.
	OnText       func(string) error // Receives complete UTF-8 and never a stop-string prefix.
}
type Result struct {
	RequestID    string
	TokenIDs     []int // Exact original IDs received, including specials/stop IDs.
	Text         string
	FinishReason string
}

// PartialError means the submitted stream did not complete. TokenIDs/Text in
// Result contain only accepted output. Retrying cannot safely resume this stream.
// No untrusted server diagnostics, tokens, plaintext or transport URLs are echoed.
type PartialError struct {
	ReceivedTokens int
	Code           string
}

func (e *PartialError) Error() string { return "incomplete token stream: " + e.Code }

func (c *Client) Generate(ctx context.Context, prompt []int, o Options) (Result, error) {
	result := Result{TokenIDs: []int{}}
	if o.RequestID == "" {
		var id [16]byte
		if _, err := rand.Read(id[:]); err != nil {
			return result, errors.New("request_id_failed")
		}
		o.RequestID = hex.EncodeToString(id[:])
	}
	result.RequestID = o.RequestID
	if !p.ValidRequestID(o.RequestID) || len(o.StopStrings) > MaxStopStrings || len(o.StopTokenIDs) > 16 {
		return result, errors.New("invalid_local_request")
	}
	for _, stop := range o.StopStrings {
		if len(stop) == 0 || len(stop) > MaxStopBytes || !utf8.ValidString(stop) {
			return result, errors.New("invalid_local_stop")
		}
	}
	mapped, err := c.secret.MapIDs(prompt)
	if err != nil {
		return result, errors.New("invalid_local_request")
	}
	stops, err := c.secret.MapIDs(o.StopTokenIDs)
	if err != nil {
		return result, errors.New("invalid_local_request")
	}
	if stops == nil {
		stops = []int{}
	}
	temp := 0.0
	req := p.Request{ProtocolVersion: c.expected.ProtocolVersion, ModelID: c.expected.ModelID, ArtifactID: c.expected.ArtifactID, BindingID: c.expected.BindingID(), RequestID: o.RequestID, PromptTokenIDs: mapped, StopTokenIDs: stops, Sampling: p.Sampling{Temperature: &temp, MaxTokens: o.MaxTokens}, Stream: true}
	if req.Validate(c.expected) != nil {
		return result, errors.New("invalid_local_request")
	}
	if err = c.Handshake(ctx); err != nil {
		return result, err
	}
	data, _ := json.Marshal(req)
	// Cancelling closes the HTTP body and propagates local stop/callback cancellation.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	resp, err := c.request(ctx, http.MethodPost, "/v1/generate", data)
	if err != nil {
		if err.Error() == "transport_failed" {
			return result, &PartialError{Code: "submission_unknown"}
		}
		return result, err
	}
	defer resp.Body.Close()
	incomplete := func(code string) (Result, error) {
		return result, &PartialError{ReceivedTokens: len(result.TokenIDs), Code: code}
	}
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		return incomplete("invalid_stream")
	}
	reader := newEvents(resp.Body)
	decoder := textDecoder{stops: slices.Clone(o.StopStrings)}
	emit := func(s string) error {
		if s == "" {
			return nil
		}
		result.Text += s
		if o.OnText != nil {
			if o.OnText(s) != nil {
				return errors.New("consumer_failed")
			}
		}
		return nil
	}
	localReason := ""
	for sequence := 1; sequence <= o.MaxTokens+1; sequence++ {
		event, err := reader.next()
		if err != nil {
			return incomplete("stream_interrupted")
		}
		if event.Validate(o.RequestID, sequence, c.expected.VocabularySize()) != nil {
			return incomplete("invalid_stream")
		}
		if event.FinishReason != "" {
			if event.FinishReason == "error" {
				return incomplete(event.Error)
			}
			if event.FinishReason == "eos" && localReason != "eos" || event.FinishReason == "stop_token" && localReason != "stop_token" || localReason != "" && event.FinishReason != localReason || event.FinishReason == "length" && len(result.TokenIDs) != o.MaxTokens || event.FinishReason == "backend_done" && len(result.TokenIDs) >= o.MaxTokens {
				return incomplete("invalid_finish")
			}
			tail, err := decoder.finish()
			if err != nil {
				return incomplete("invalid_utf8")
			}
			if emit(tail) != nil {
				return incomplete("consumer_failed")
			}
			result.FinishReason = event.FinishReason
			return result, nil
		}
		if localReason != "" || len(result.TokenIDs) >= o.MaxTokens {
			return incomplete("invalid_stream")
		}
		ids, err := c.secret.RestoreIDs(event.OutputTokenIDs)
		if err != nil {
			return incomplete("invalid_stream")
		}
		id := ids[0]
		result.TokenIDs = append(result.TokenIDs, id)
		// EOS wins over aliased BOS/PAD/stop roles. All special IDs remain in TokenIDs.
		if slices.Contains(c.special["eos"], id) {
			localReason = "eos"
			continue
		}
		if slices.Contains(o.StopTokenIDs, id) {
			localReason = "stop_token"
			continue
		}
		if slices.Contains(c.special["bos"], id) || slices.Contains(c.special["pad"], id) {
			continue
		}
		piece, err := c.piece(id)
		if err != nil || len(piece) > MaxPieceBytes {
			return incomplete("invalid_piece")
		}
		text, stopped, err := decoder.add(piece)
		if err != nil {
			return incomplete("invalid_utf8")
		}
		if emit(text) != nil {
			return incomplete("consumer_failed")
		}
		if stopped {
			result.FinishReason = "text_stop"
			return result, nil
		}
	}
	return incomplete("invalid_stream")
}
