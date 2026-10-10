// Package runtimeprotocol defines the P3 diagnostic token-only wire contract.
// No model, tokenizer, Client Secret or plaintext belongs in this package.
package runtimeprotocol

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

const (
	Version   = 1
	Profile   = "p3-fake-token-only"
	MaxTokens = 4096
	MaxBody   = 64 << 10
	MaxEvent  = 4096
	MaxStream = (MaxTokens + 1) * (MaxEvent + 32)
)

// Runtime is public capability metadata, supplied via the trusted configuration
// channel to both client and gateway. ArtifactID is an expected identity only;
// the fake gateway must not load or receive P1/P2 diagnostic weight artifacts.
type Runtime struct {
	ProtocolVersion int                   `json:"protocol_version"`
	Profile         string                `json:"profile"`
	ModelID         string                `json:"model_id"`
	ArtifactID      string                `json:"artifact_id"`
	Binding         artifact.TokenBinding `json:"token_binding"`
	Features        []string              `json:"supported_features"`
	MaxPromptTokens int                   `json:"max_prompt_tokens"`
	MaxOutputTokens int                   `json:"max_output_tokens"`
	ContextTokens   int                   `json:"context_tokens"`
}

func (r Runtime) Validate() error {
	b := r.Binding
	if r.ProtocolVersion != Version || r.Profile != Profile || !ValidModelID(r.ModelID) || !ValidHash(r.ArtifactID) || b.SchemaVersion != 1 || b.Profile != artifact.TokenProfile || !ValidHash(b.BindingID) || !ValidHash(b.TokenizerFingerprint) || b.MethodRevision != artifact.TokenMethod || b.RNGAlgorithm != artifact.TokenRNG || b.RNGRevision != artifact.TokenRNGRevision || b.VocabularySize < 2 || b.VocabularySize > artifact.MaxDiagnosticVocab || b.HiddenSize < 1 || b.HiddenSize > artifact.MaxDiagnosticHidden || b.VocabularySize > artifact.MaxDiagnosticElements/b.HiddenSize || artifact.ValidateSpecialIDs(b.MappedSpecialIDs, b.VocabularySize) != nil || !slices.Equal(r.Features, []string{"greedy", "sse", "stop_token_ids"}) || r.MaxPromptTokens < 1 || r.MaxPromptTokens > MaxTokens || r.MaxOutputTokens < 1 || r.MaxOutputTokens > MaxTokens || r.ContextTokens < 2 || r.ContextTokens > 2*MaxTokens || r.MaxPromptTokens > r.ContextTokens || r.MaxOutputTokens > r.ContextTokens {
		return errors.New("unsupported runtime capability")
	}
	return nil
}

func Equal(a, b any) bool {
	x, ex := json.Marshal(a)
	y, ey := json.Marshal(b)
	return ex == nil && ey == nil && bytes.Equal(x, y)
}

func Clone(r Runtime) Runtime {
	data, _ := json.Marshal(r)
	var copy Runtime
	_ = json.Unmarshal(data, &copy)
	return copy
}

func ValidHash(s string) bool {
	if len(s) != 64 || s != strings.ToLower(s) {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
func ValidRequestID(s string) bool {
	if len(s) != 32 || s != strings.ToLower(s) {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
func ValidModelID(s string) bool {
	if len(s) < 1 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}
func ValidAuth(s string) bool {
	if len(s) < 16 || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if !(c >= '!' && c <= '~') {
			return false
		}
	}
	return true
}

type Sampling struct {
	Temperature *float64 `json:"temperature"`
	MaxTokens   int      `json:"max_tokens"`
}
type Request struct {
	ProtocolVersion int      `json:"protocol_version"`
	ModelID         string   `json:"model_id"`
	ArtifactID      string   `json:"artifact_id"`
	BindingID       string   `json:"binding_id"`
	RequestID       string   `json:"request_id"`
	PromptTokenIDs  []int    `json:"prompt_token_ids"`
	StopTokenIDs    []int    `json:"stop_token_ids"`
	Sampling        Sampling `json:"sampling"`
	Stream          bool     `json:"stream"`
}

func (r Request) Validate(cap Runtime) error {
	if r.ProtocolVersion != cap.ProtocolVersion || r.ModelID != cap.ModelID || r.ArtifactID != cap.ArtifactID || r.BindingID != cap.Binding.BindingID {
		return errors.New("binding_mismatch")
	}
	if !ValidRequestID(r.RequestID) || !r.Stream || r.Sampling.Temperature == nil || *r.Sampling.Temperature != 0 || r.Sampling.MaxTokens < 1 || r.Sampling.MaxTokens > cap.MaxOutputTokens || len(r.PromptTokenIDs) < 1 || len(r.PromptTokenIDs) > cap.MaxPromptTokens || len(r.PromptTokenIDs)+r.Sampling.MaxTokens > cap.ContextTokens || r.StopTokenIDs == nil || len(r.StopTokenIDs) > 16 {
		return errors.New("invalid_request")
	}
	if !ValidIDs(r.PromptTokenIDs, cap.Binding.VocabularySize) || !ValidIDs(r.StopTokenIDs, cap.Binding.VocabularySize) {
		return errors.New("invalid_request")
	}
	for i, id := range r.StopTokenIDs {
		if slices.Contains(r.StopTokenIDs[:i], id) {
			return errors.New("invalid_request")
		}
	}
	return nil
}
func ValidIDs(ids []int, vocab int) bool {
	for _, id := range ids {
		if id < 0 || id >= vocab {
			return false
		}
	}
	return true
}

// Event emits one physical token or a terminal record. Sequences start at 1,
// advance by one, and include the terminal record. No SSE id/reconnect support.
type Event struct {
	RequestID      string `json:"request_id"`
	Sequence       int    `json:"sequence"`
	OutputTokenIDs []int  `json:"output_token_ids"`
	FinishReason   string `json:"finish_reason"`
	Error          string `json:"error"`
}

func (e Event) Validate(requestID string, sequence, vocab int) error {
	if e.RequestID != requestID || e.Sequence != sequence || e.OutputTokenIDs == nil || !ValidIDs(e.OutputTokenIDs, vocab) {
		return errors.New("invalid_stream")
	}
	if len(e.OutputTokenIDs) == 1 && e.FinishReason == "" && e.Error == "" {
		return nil
	}
	if len(e.OutputTokenIDs) != 0 {
		return errors.New("invalid_stream")
	}
	switch e.FinishReason {
	case "eos", "stop_token", "length", "backend_done":
		if e.Error == "" {
			return nil
		}
	case "error":
		if e.Error == "backend_error" || e.Error == "timeout" {
			return nil
		}
	}
	return errors.New("invalid_stream")
}
