package cli

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yakumioto/AloeForge/internal/artifact"
)

func TestInspectSummaryAndRedactedErrors(t *testing.T) {
	root := t.TempDir()
	config := `{"model_type":"llama","architectures":["LlamaForCausalLM"]}`
	if err := os.WriteFile(filepath.Join(root, "config.json"), []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	header := `{"synthetic":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 0)
	if err := os.WriteFile(filepath.Join(root, "model.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := Run([]string{"inspect", "--model-root", root}, &out, &errOut); code != 0 {
		t.Fatal(code, errOut.String())
	}
	var result struct {
		Shards  int    `json:"shards"`
		Tensors int    `json:"tensors"`
		Hash    string `json:"inventory_sha256"`
	}
	if err := json.Unmarshal(out.Bytes(), &result); err != nil || result.Shards != 1 || result.Tensors != 1 || len(result.Hash) != 64 {
		t.Fatal("bad inspection summary", err)
	}
	if strings.Contains(out.String(), root) || strings.Contains(out.String(), "synthetic") {
		t.Fatal("summary echoed source paths/names")
	}
	if err := os.WriteFile(filepath.Join(root, "config.json"), []byte(`{"private-marker"`), 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	if code := Run([]string{"inspect", "--model-root", root}, &out, &errOut); code != 1 || out.Len() != 0 {
		t.Fatal("bad inspection failure")
	}
	if strings.Contains(errOut.String(), "private-marker") || strings.Contains(errOut.String(), root) {
		t.Fatal("raw input leaked into diagnostics")
	}
}

func TestVerifySummaryAndTamperFailure(t *testing.T) {
	parent := t.TempDir()
	payload := []byte{1}
	digest := sha256.Sum256(payload)
	plan := artifact.Plan{SchemaVersion: 1, SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("0", 64), MethodRevision: "p1-diagnostic", RNGAlgorithm: "none", RNGRevision: "none", Outputs: []artifact.OutputTensor{{Name: "output", Shard: "out.safetensors", DType: "U8", Shape: []int64{1}, PayloadHash: hex.EncodeToString(digest[:])}}}
	stage, err := artifact.Create(parent, ".stage-cli", plan)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if err := stage.WriteShard("out.safetensors", map[string]io.Reader{"output": bytes.NewReader(payload)}); err != nil {
		t.Fatal(err)
	}
	if _, err := stage.Publish("visible"); err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(parent, "visible")
	var out, errOut bytes.Buffer
	if code := Run([]string{"verify", "--artifact-root", root}, &out, &errOut); code != 0 || !strings.Contains(out.String(), artifact.Kind) {
		t.Fatal("bad verification summary", code, errOut.String())
	}
	if err := os.WriteFile(filepath.Join(root, "manifest.json"), []byte(`{"private-marker"`), 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	if code := Run([]string{"verify", "--artifact-root", root}, &out, &errOut); code != 1 || out.Len() != 0 {
		t.Fatal("bad verification failure")
	}
	if strings.Contains(errOut.String(), "private-marker") || strings.Contains(errOut.String(), root) {
		t.Fatal("raw artifact input leaked into diagnostics")
	}
}
