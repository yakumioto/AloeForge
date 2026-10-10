package tokenbaseline

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/safetensors"
)

var toyEmbedding = [][]float32{{1, 2}, {3, -1}, {0.5, 4}, {-2, 0.25}}
var toyHead = [][]float32{{2, -1}, {0, 3}, {-4, 0.5}, {1, 1}}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

func fixture(t *testing.T, tied bool) Options {
	t.Helper()
	common := t.TempDir()
	options := Options{SourceRoot: filepath.Join(common, "source"), OutputParent: filepath.Join(common, "output"), SecretRoot: filepath.Join(common, "secret"), StageName: ".stage-p2", TargetName: "visible"}
	for _, dir := range []string{options.SourceRoot, options.OutputParent, options.SecretRoot} {
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON(t, filepath.Join(options.SourceRoot, "config.json"), map[string]any{"model_type": "llama", "architectures": []string{"LlamaForCausalLM"}, "aloeforge_synthetic": true})
	writeJSON(t, filepath.Join(options.SourceRoot, "token-baseline.json"), fixtureSpec{SchemaVersion: 1, Kind: "synthetic-embedding-head-v1", VocabularySize: 4, HiddenSize: 2, TiedEmbeddings: &tied, SpecialIDs: map[string][]int{"bos": {0}, "eos": {1, 3}, "pad": {0}}})
	writeJSON(t, filepath.Join(options.SourceRoot, "tokenizer.json"), map[string]any{"schema_version": 1, "kind": "synthetic-vocabulary-v1", "tokens": []string{"<bos>", "<eos>", "synthetic", "<eos2>"}})
	header := map[string]any{}
	var payload bytes.Buffer
	for i, matrix := range [][][]float32{toyEmbedding, toyHead} {
		if tied && i == 1 {
			continue
		}
		name := "embedding.weight"
		if i == 1 {
			name = "lm_head.weight"
		}
		start := payload.Len()
		for _, row := range matrix {
			for _, value := range row {
				if err := binary.Write(&payload, binary.LittleEndian, value); err != nil {
					t.Fatal(err)
				}
			}
		}
		header[name] = map[string]any{"dtype": "F32", "shape": []int{4, 2}, "data_offsets": []int{start, payload.Len()}}
	}
	encoded, _ := json.Marshal(header)
	encoded = append(encoded, bytes.Repeat([]byte(" "), -len(encoded)&7)...)
	var file bytes.Buffer
	binary.Write(&file, binary.LittleEndian, uint64(len(encoded)))
	file.Write(encoded)
	file.Write(payload.Bytes())
	if err := os.WriteFile(filepath.Join(options.SourceRoot, "model.safetensors"), file.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return options
}

func readMatrix(t *testing.T, path, name string) [][]float64 {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	stat, _ := file.Stat()
	header, err := safetensors.Read(file, stat.Size())
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range header.Tensors {
		if tensor.Name != name {
			continue
		}
		rows := make([][]float64, int(tensor.Shape[0]))
		reader := io.NewSectionReader(file, header.DataStart+tensor.Offsets[0], tensor.Offsets[1]-tensor.Offsets[0])
		for i := range rows {
			rows[i] = make([]float64, int(tensor.Shape[1]))
			for j := range rows[i] {
				var value float32
				if err := binary.Read(reader, binary.LittleEndian, &value); err != nil {
					t.Fatal(err)
				}
				rows[i][j] = float64(value)
			}
		}
		return rows
	}
	t.Fatal("matrix missing")
	return nil
}

func TestTiedUntiedLogitsAndStableResume(t *testing.T) {
	for _, tied := range []bool{false, true} {
		t.Run(map[bool]string{false: "untied", true: "tied"}[tied], func(t *testing.T) {
			opts := fixture(t, tied)
			src, err := loadSource(opts.SourceRoot)
			if err != nil {
				t.Fatal(err)
			}
			secret := fixedSecret(t, src.context)
			src.close()
			if err := SaveSecret(opts.SecretRoot, secret); err != nil {
				t.Fatal(err)
			}
			opts.Resume = true // Secret persisted before stage creation: recover that boundary.
			result, err := Compile(opts)
			if err != nil {
				t.Fatal(err)
			}
			pub := filepath.Join(opts.OutputParent, opts.TargetName)
			if _, err := VerifyClient(pub, opts.SecretRoot, result.ArtifactID, result.TokenizerFingerprint); err != nil {
				t.Fatal(err)
			}
			emb := readMatrix(t, filepath.Join(pub, "model.safetensors"), "embedding.weight")
			head := emb
			sourceHead := toyEmbedding
			if !tied {
				head = readMatrix(t, filepath.Join(pub, "model.safetensors"), "lm_head.weight")
				sourceHead = toyHead
			}
			for original, row := range toyEmbedding {
				mapped, _ := secret.MapIDs([]int{original})
				physical := make([]float64, 4)
				want := make([]float64, 4)
				for i := range physical {
					for j := range row {
						physical[i] += emb[mapped[0]][j] * head[i][j]
						want[i] += float64(row[j]) * float64(sourceHead[i][j])
					}
				}
				restored, err := secret.RestoreLogits(physical)
				if err != nil || !slices.Equal(restored, want) {
					t.Fatal("model logits mismatch after vocab remapping")
				}
				if original == 1 {
					known := []float64{7, -3, -12.5, 2}
					if tied {
						known = []float64{1, 10, -2.5, -6.25}
					}
					if !slices.Equal(want, known) {
						t.Fatal("known synthetic logits mismatch")
					}
					if slices.Equal(physical, want) {
						t.Fatal("fixture must detect comparison in wrong coordinates")
					}
				}
			}
			data, _ := os.ReadFile(filepath.Join(pub, "manifest.json"))
			for _, private := range []string{`"permutation":`, `"nonce":`, `"special_token_ids":`, `"source":`, "source_inventory", "source_content", "client-secret", "<bos>", "<eos>"} {
				if bytes.Contains(data, []byte(private)) {
					t.Fatal("private material in public manifest")
				}
			}
			entries, _ := os.ReadDir(pub)
			if len(entries) != 2 {
				t.Fatal("private controls published")
			}
			before, _ := os.ReadFile(filepath.Join(opts.SecretRoot, secretFile))
			stageFile := filepath.Join(opts.OutputParent, opts.StageName, "model.safetensors")
			stat, _ := os.Stat(stageFile)
			retry, err := Compile(opts)
			if err != nil || retry != result {
				t.Fatal("resume regenerated artifact identity", err)
			}
			after, _ := os.ReadFile(filepath.Join(opts.SecretRoot, secretFile))
			current, _ := os.Stat(stageFile)
			if !bytes.Equal(before, after) || !stat.ModTime().Equal(current.ModTime()) {
				t.Fatal("resume rewrote secret or completed shard")
			}
			opts.Resume = false
			if _, err := Compile(opts); err == nil {
				t.Fatal("fresh compile replaced existing secret")
			}
		})
	}
}

func TestCompilerMismatchAndTamper(t *testing.T) {
	mutations := map[string]func(*testing.T, Options){
		"source-bytes": func(t *testing.T, o Options) {
			p := filepath.Join(o.SourceRoot, "model.safetensors")
			data, _ := os.ReadFile(p)
			data[len(data)-1] ^= 1
			os.WriteFile(p, data, 0600)
		},
		"tokenizer": func(t *testing.T, o Options) {
			p := filepath.Join(o.SourceRoot, "tokenizer.json")
			data, _ := os.ReadFile(p)
			os.WriteFile(p, append(data, '\n'), 0600)
		},
		"special-IDs": func(t *testing.T, o Options) {
			p := filepath.Join(o.SourceRoot, "token-baseline.json")
			data, _ := os.ReadFile(p)
			var spec fixtureSpec
			json.Unmarshal(data, &spec)
			spec.SpecialIDs["bos"] = []int{2}
			writeJSON(t, p, spec)
		},
		"secret-swap": func(t *testing.T, o Options) {
			s, err := LoadSecret(o.SecretRoot)
			if err != nil {
				t.Fatal(err)
			}
			s.disk.Material.Nonce = strings.Repeat("1", 64)
			s.disk.BindingID = commitment(s.disk.Material)
			writeJSON(t, filepath.Join(o.SecretRoot, secretFile), s.disk)
		},
		"missing-secret": func(t *testing.T, o Options) { os.Remove(filepath.Join(o.SecretRoot, secretFile)) },
		"partial-secret": func(t *testing.T, o Options) {
			os.WriteFile(filepath.Join(o.SecretRoot, secretFile), []byte("{"), 0600)
		},
		"completed-payload": func(t *testing.T, o Options) {
			p := filepath.Join(o.OutputParent, o.StageName, "model.safetensors")
			data, _ := os.ReadFile(p)
			data[len(data)-1] ^= 1
			os.WriteFile(p, data, 0600)
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			opts := fixture(t, false)
			result, err := Compile(opts)
			if err != nil {
				t.Fatal(err)
			}
			mutate(t, opts)
			opts.Resume = true
			if _, err := Compile(opts); err == nil {
				t.Fatal("mismatch accepted")
			}
			_, digest, err := artifact.Verify(filepath.Join(opts.OutputParent, opts.TargetName))
			if err != nil || digest != result.ArtifactID {
				t.Fatal("failed resume changed published output")
			}
		})
	}
	opts := fixture(t, false)
	result, err := Compile(opts)
	if err != nil {
		t.Fatal(err)
	}
	pub := filepath.Join(opts.OutputParent, opts.TargetName)
	for _, ids := range [][2]string{{strings.Repeat("0", 64), result.TokenizerFingerprint}, {result.ArtifactID, strings.Repeat("0", 64)}, {"private-input", result.TokenizerFingerprint}} {
		if _, err := VerifyClient(pub, opts.SecretRoot, ids[0], ids[1]); err == nil {
			t.Fatal("wrong expected client identity accepted")
		}
	}
	s, err := LoadSecret(opts.SecretRoot)
	if err != nil {
		t.Fatal(err)
	}
	s.disk.Material.Nonce = strings.Repeat("1", 64)
	s.disk.BindingID = commitment(s.disk.Material)
	wrong := privateDir(t)
	if err := SaveSecret(wrong, s); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyClient(pub, wrong, result.ArtifactID, result.TokenizerFingerprint); err == nil {
		t.Fatal("different valid secret accepted")
	}
	manifest, _, _ := artifact.Verify(pub)
	manifest.Binding.MappedSpecialIDs["bos"] = []int{(manifest.Binding.MappedSpecialIDs["bos"][0] + 1) % 4}
	writeJSON(t, filepath.Join(pub, "manifest.json"), manifest)
	_, digest, err := artifact.Verify(pub)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyClient(pub, opts.SecretRoot, digest, result.TokenizerFingerprint); err == nil {
		t.Fatal("mapped special IDs ignored by client binding")
	}
}

func TestFailClosedFixturesAndDirectories(t *testing.T) {
	for _, kind := range []string{"overlap", "symlink-overlap", "unknown-kind", "unknown-layout-field", "missing-tied", "duplicate-token", "too-long-token", "special-bounds", "shape", "dtype", "extra-tensor", "nonfinite", "invalid-stage", "invalid-target", "public-output"} {
		t.Run(kind, func(t *testing.T) {
			opts := fixture(t, false)
			switch kind {
			case "overlap":
				opts.SecretRoot = opts.OutputParent
			case "symlink-overlap":
				link := filepath.Join(t.TempDir(), "secret-link")
				os.Symlink(opts.OutputParent, link)
				opts.SecretRoot = link
			case "invalid-stage":
				opts.StageName = "../private-input"
			case "invalid-target":
				opts.TargetName = "../private-input"
			case "public-output":
				os.Chmod(opts.OutputParent, 0755)
			case "duplicate-token":
				writeJSON(t, filepath.Join(opts.SourceRoot, "tokenizer.json"), map[string]any{"schema_version": 1, "kind": "synthetic-vocabulary-v1", "tokens": []string{"a", "a", "b", "c"}})
			case "too-long-token":
				writeJSON(t, filepath.Join(opts.SourceRoot, "tokenizer.json"), map[string]any{"schema_version": 1, "kind": "synthetic-vocabulary-v1", "tokens": []string{strings.Repeat("x", 65), "a", "b", "c"}})
			case "unknown-kind", "unknown-layout-field", "missing-tied", "special-bounds":
				p := filepath.Join(opts.SourceRoot, "token-baseline.json")
				data, _ := os.ReadFile(p)
				var spec map[string]any
				json.Unmarshal(data, &spec)
				switch kind {
				case "unknown-kind":
					spec["kind"] = "real-transformer"
				case "unknown-layout-field":
					spec["attention"] = "unimplemented"
				case "missing-tied":
					delete(spec, "tied_embeddings")
				case "special-bounds":
					spec["special_token_ids"] = map[string][]int{"bos": {4}, "eos": {1}, "pad": {0}}
				}
				writeJSON(t, p, spec)
			case "nonfinite":
				p := filepath.Join(opts.SourceRoot, "model.safetensors")
				data, _ := os.ReadFile(p)
				binary.LittleEndian.PutUint32(data[len(data)-4:], math.Float32bits(float32(math.Inf(1))))
				os.WriteFile(p, data, 0600)
			case "shape", "dtype", "extra-tensor":
				p := filepath.Join(opts.SourceRoot, "model.safetensors")
				data, _ := os.ReadFile(p)
				length := int(binary.LittleEndian.Uint64(data[:8]))
				var hdr map[string]map[string]any
				json.Unmarshal(data[8:8+length], &hdr)
				switch kind {
				case "shape":
					hdr["embedding.weight"]["shape"] = []int{2, 4}
				case "dtype":
					hdr["embedding.weight"]["dtype"] = "I32"
				case "extra-tensor":
					hdr["unused.weight"] = hdr["lm_head.weight"]
					delete(hdr, "lm_head.weight")
				}
				encoded, _ := json.Marshal(hdr)
				var buf bytes.Buffer
				binary.Write(&buf, binary.LittleEndian, uint64(len(encoded)))
				buf.Write(encoded)
				buf.Write(data[8+length:])
				os.WriteFile(p, buf.Bytes(), 0600)
			}
			if _, err := Compile(opts); err == nil {
				t.Fatal("unsupported fixture accepted")
			}
			if _, err := os.Stat(filepath.Join(opts.OutputParent, opts.TargetName)); !os.IsNotExist(err) {
				t.Fatal("failed compilation published output")
			}
			if _, err := os.Stat(filepath.Join(opts.SecretRoot, secretFile)); !os.IsNotExist(err) {
				t.Fatal("invalid fixture saved a secret")
			}
		})
	}
}

func TestP2CrashChild(t *testing.T) {
	encoded := os.Getenv("ALOE_P2_CRASH_OPTIONS")
	if encoded == "" {
		return
	}
	var opts Options
	if json.Unmarshal([]byte(encoded), &opts) != nil {
		os.Exit(91)
	}
	s, err := loadSource(opts.SourceRoot)
	if err != nil {
		os.Exit(92)
	}
	secret, err := NewSecret(s.context)
	if err != nil || SaveSecret(opts.SecretRoot, secret) != nil {
		os.Exit(93)
	}
	if os.Getenv("ALOE_P2_CRASH_POINT") == "secret" {
		os.Exit(77)
	}
	plan, err := buildPlan(s, secret)
	if err != nil {
		os.Exit(94)
	}
	stage, err := artifact.Create(opts.OutputParent, opts.StageName, plan)
	if err != nil {
		os.Exit(95)
	}
	if os.Getenv("ALOE_P2_CRASH_POINT") == "completed" {
		readers := map[string]io.Reader{}
		for _, output := range plan.Outputs {
			readers[output.Name] = s.reader(output.Name, secret)
		}
		if stage.WriteShard("model.safetensors", readers) != nil {
			os.Exit(96)
		}
	}
	os.Exit(77) // Deliberately skip Close; kernel locks release on process death.
}

func TestP2ProcessDeathRecovery(t *testing.T) {
	for _, point := range []string{"secret", "stage", "completed"} {
		t.Run(point, func(t *testing.T) {
			opts := fixture(t, false)
			encoded, _ := json.Marshal(opts)
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(executable, "-test.run=^TestP2CrashChild$")
			cmd.Env = append(os.Environ(), "ALOE_P2_CRASH_OPTIONS="+string(encoded), "ALOE_P2_CRASH_POINT="+point)
			err = cmd.Run()
			if exit, ok := err.(*exec.ExitError); !ok || exit.ExitCode() != 77 {
				t.Fatal("crash child did not reach boundary", err)
			}
			before, _ := os.ReadFile(filepath.Join(opts.SecretRoot, secretFile))
			opts.Resume = true
			result, err := Compile(opts)
			if err != nil {
				t.Fatal(err)
			}
			after, _ := os.ReadFile(filepath.Join(opts.SecretRoot, secretFile))
			if !bytes.Equal(before, after) {
				t.Fatal("recovery regenerated secret")
			}
			if _, err := VerifyClient(filepath.Join(opts.OutputParent, opts.TargetName), opts.SecretRoot, result.ArtifactID, result.TokenizerFingerprint); err != nil {
				t.Fatal(err)
			}
		})
	}
}
