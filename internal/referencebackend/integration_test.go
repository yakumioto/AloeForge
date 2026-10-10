package referencebackend

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
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/clientruntime"
	"github.com/yakumioto/AloeForge/internal/dense"
	"github.com/yakumioto/AloeForge/internal/gateway"
	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

// The Python base CI suite compiles real synthetic publications, computes an
// independent scalar source-model oracle, removes source files and supplies a
// private test-only specification. No private specification enters worker IPC.
func TestLocalReferenceIntegration(t *testing.T) {
	path := os.Getenv("ALOE_REFERENCE_TEST_SPEC")
	if path == "" {
		t.Skip("executed by Python CPU integration suite with compiled publications")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal("test specification unavailable")
	}
	var cases []struct {
		Root, Secret, Python, ArtifactID, Tokenizer string
		Prompt, Expected                            []int
		MaxTokens                                   int
	}
	if json.Unmarshal(data, &cases) != nil || len(cases) == 0 {
		t.Fatal("invalid private specification")
	}
	for index, tc := range cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			if _, err := dense.VerifyClient(tc.Root, tc.Secret, tc.ArtifactID, tc.Tokenizer); err != nil {
				t.Fatal("trusted publication verification failed")
			}
			secret, err := dense.LoadSecret(tc.Secret)
			if err != nil {
				t.Fatal("private client secret unavailable")
			}
			manifest, id, err := artifact.Verify(tc.Root)
			if err != nil {
				t.Fatal("public artifact verification failed")
			}
			cap := p.Runtime{ProtocolVersion: 1, Profile: p.LocalDenseProfile, ModelID: "local-reference", ArtifactID: id, Dense: manifest.Dense, Features: []string{"greedy", "sse", "stop_token_ids"}, MaxPromptTokens: 32, MaxOutputTokens: 16, ContextTokens: 64}
			b, err := New(Config{Runtime: cap, ArtifactRoot: tc.Root, Python: tc.Python})
			if err != nil {
				t.Fatal(err)
			}
			const auth = "local-test-only-authentication"
			g, err := gateway.NewLocalDense(gateway.Config{Runtime: cap, BearerToken: auth}, b)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := gateway.New(gateway.Config{Runtime: cap, BearerToken: auth}); err == nil {
				t.Fatal("dense admitted to P3 fake constructor")
			}
			if _, err := gateway.NewRelay(gateway.Config{Runtime: cap, BearerToken: auth}, b); err == nil {
				t.Fatal("dense admitted to remote mock relay")
			}
			for _, addr := range []string{":0", "0.0.0.0:0", "localhost:0", "192.0.2.1:0"} {
				if _, err := g.LocalServer(addr); err == nil {
					t.Fatal("nonliteral loopback listener admitted")
				}
			}
			request := httptest.NewRequest(http.MethodGet, "https://127.0.0.1/v1/models/local-reference/runtime", nil)
			request.RemoteAddr = "192.0.2.1:9999"
			rejected := httptest.NewRecorder()
			g.ServeHTTP(rejected, request)
			if rejected.Code != 403 {
				t.Fatal("remote peer admitted")
			}
			requests := make(chan p.Request, 16)
			var logs bytes.Buffer
			ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					raw, _ := io.ReadAll(io.LimitReader(r.Body, p.MaxBody+1))
					r.Body = io.NopCloser(bytes.NewReader(raw))
					var req p.Request
					if p.DecodeRequest(raw, &req) == nil {
						requests <- req
					}
				}
				g.ServeHTTP(w, r)
			}))
			server, err := g.LocalServer("127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			server.Handler = ts.Config.Handler
			server.ErrorLog = log.New(&logs, "", 0)
			ts.Config = server
			ts.StartTLS()
			defer ts.Close()
			roots := x509.NewCertPool()
			roots.AddCert(ts.Certificate())
			cfg := clientruntime.Config{URL: ts.URL, BearerToken: auth, Expected: cap, DenseSecret: secret, TokenizerFingerprint: tc.Tokenizer, Piece: func(int) ([]byte, error) { return []byte("Q"), nil }, TLS: &tls.Config{RootCAs: roots}}
			client, err := clientruntime.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			for _, url := range []string{"https://localhost:9999", "https://192.0.2.1:9999", "https://example.invalid"} {
				bad := cfg
				bad.URL = url
				if _, err := clientruntime.New(bad); err == nil {
					t.Fatal("remote/DNS endpoint admitted")
				}
			}
			bad := cfg
			bad.TokenizerFingerprint = strings.Repeat("0", 64)
			if _, err := clientruntime.New(bad); err == nil {
				t.Fatal("wrong tokenizer admitted")
			}
			bad = cfg
			bad.Expected = p.Clone(cap)
			bad.Expected.Dense.BindingID = strings.Repeat("0", 64)
			if _, err := clientruntime.New(bad); err == nil {
				t.Fatal("wrong private binding admitted")
			}
			bad = cfg
			bad.Expected = p.Clone(cap)
			bad.Expected.Dense.RopeTheta += 1
			mismatch, err := clientruntime.New(bad)
			if err != nil {
				t.Fatal(err)
			}
			defer mismatch.Close()
			if mismatch.Handshake(context.Background()) == nil {
				t.Fatal("changed dense runtime metadata admitted")
			}
			result, err := client.Generate(context.Background(), tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("1", 32), MaxTokens: tc.MaxTokens})
			if err != nil || !slices.Equal(result.TokenIDs, tc.Expected) {
				t.Fatal("reference output differs from independent source-model oracle")
			}
			req := <-requests
			mapped, _ := secret.MapIDs(tc.Prompt)
			if !slices.Equal(req.PromptTokenIDs, mapped) || req.BindingID != cap.BindingID() {
				t.Fatal("original IDs or wrong binding on wire")
			}
			waitEmpty(t, b)
			if _, err := client.Generate(context.Background(), tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("1", 32), MaxTokens: tc.MaxTokens}); err == nil {
				t.Fatal("duplicate stream accepted")
			}
			<-requests
			stop := tc.Expected[0]
			result, err = client.Generate(context.Background(), tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("2", 32), MaxTokens: tc.MaxTokens, StopTokenIDs: []int{stop}})
			if err != nil || !slices.Equal(result.TokenIDs, []int{stop}) || result.Text != "" {
				t.Fatal("mapped physical stop failed")
			}
			req = <-requests
			stops, _ := secret.MapIDs([]int{stop})
			if !slices.Equal(req.StopTokenIDs, stops) {
				t.Fatal("stop not mapped")
			}
			waitEmpty(t, b)
			// Each request creates/reaps a new process/cache, retaining exact output.
			result, err = client.Generate(context.Background(), tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("3", 32), MaxTokens: tc.MaxTokens})
			if err != nil || !slices.Equal(result.TokenIDs, tc.Expected) {
				t.Fatal("cache leaked across requests")
			}
			<-requests
			waitEmpty(t, b)
			// Local plaintext stops never enter physical request control.
			result, err = client.Generate(context.Background(), tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("4", 32), MaxTokens: tc.MaxTokens, StopStrings: []string{"Q"}})
			if err != nil || result.Text != "" || result.FinishReason != "text_stop" {
				t.Fatal("local text stop failed")
			}
			<-requests
			waitEmpty(t, b)
			if index == 0 {
				ctx, cancel := context.WithCancel(context.Background())
				result, err = client.Generate(ctx, tc.Prompt, clientruntime.Options{RequestID: strings.Repeat("5", 32), MaxTokens: tc.MaxTokens, OnText: func(string) error { cancel(); return nil }})
				cancel()
				var partial *clientruntime.PartialError
				if !errors.As(err, &partial) || len(result.TokenIDs) == 0 {
					t.Fatal("real process cancellation not observed")
				}
				<-requests
				waitEmpty(t, b)
				// Timeout during actual interpreter/artifact startup, with a real
				// TLS error terminal and process cleanup rather than a mock delay.
				short, err := gateway.NewLocalDense(gateway.Config{Runtime: cap, BearerToken: auth, RequestTimeout: time.Millisecond}, b)
				if err != nil {
					t.Fatal(err)
				}
				timed := httptest.NewUnstartedServer(short)
				timed.Config, _ = short.LocalServer("127.0.0.1:0")
				timed.StartTLS()
				defer timed.Close()
				timedCfg := cfg
				timedCfg.URL = timed.URL
				timedRoots := x509.NewCertPool()
				timedRoots.AddCert(timed.Certificate())
				timedCfg.TLS = &tls.Config{RootCAs: timedRoots}
				timedClient, err := clientruntime.New(timedCfg)
				if err != nil {
					t.Fatal(err)
				}
				defer timedClient.Close()
				_, err = timedClient.Generate(context.Background(), tc.Prompt, clientruntime.Options{MaxTokens: tc.MaxTokens})
				if !errors.As(err, &partial) || partial.Code != "timeout" {
					t.Fatal("real reference startup timeout not classified")
				}
				waitEmpty(t, b)
			}
			if logs.Len() != 0 {
				t.Fatal("HTTP diagnostics escaped")
			}
			if index == 0 {
				// Go verified at construction; a later modified publication must
				// still fail the independent loader in the actual Python process.
				shard := filepath.Join(tc.Root, manifest.Files[0].Name)
				payload, err := os.ReadFile(shard)
				if err != nil {
					t.Fatal("fixture shard unavailable")
				}
				original := slices.Clone(payload)
				payload[len(payload)-1] ^= 1
				if os.Chmod(shard, 0600) != nil || os.WriteFile(shard, payload, 0600) != nil {
					t.Fatal("could not corrupt fixture")
				}
				defer func() { _ = os.WriteFile(shard, original, 0600) }()
				if _, err := b.Open(context.Background(), req); err == nil {
					t.Fatal("changed publication executed")
				}
				waitEmpty(t, b)
			}
		})
	}
}

func waitEmpty(t *testing.T, b *Backend) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for len(b.slots) != 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if len(b.slots) != 0 {
		t.Fatal("request worker was not released")
	}
}
