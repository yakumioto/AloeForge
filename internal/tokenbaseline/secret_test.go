package tokenbaseline

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
	"golang.org/x/sys/unix"
)

func testContext() Context {
	return Context{Source: model.SourceBinding{InventoryHash: strings.Repeat("0", 64), ContentHash: strings.Repeat("1", 64)}, TokenizerFingerprint: strings.Repeat("2", 64), VocabularySize: 4, HiddenSize: 2, SpecialIDs: map[string][]int{"bos": {0}, "eos": {1, 3}, "pad": {0}}, MethodRevision: artifact.TokenMethod, RNGAlgorithm: artifact.TokenRNG, RNGRevision: artifact.TokenRNGRevision}
}

func fixedSecret(t *testing.T, c Context) *Secret {
	t.Helper()
	m := material{Context: c, Nonce: strings.Repeat("0", 64), Permutation: []int{2, 0, 3, 1}}
	s, err := validateSecret(secretDisk{SchemaVersion: 1, BindingID: commitment(m), Material: m})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func privateDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "private")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestBijectionAndIndependentGolden(t *testing.T) {
	context := testContext()
	secret := fixedSecret(t, context)
	// Golden is calculated separately with Python json/hashlib, not Go helpers.
	const golden = "f7cd207dc66f23b4e7e160a9327b974f90b2f962c2a6dc6ba2d88ebfc6432d0c"
	if secret.disk.BindingID != golden {
		t.Fatalf("commitment golden mismatch: %s", secret.disk.BindingID)
	}
	context.SpecialIDs["bos"][0] = 2
	if secret.disk.Material.Context.SpecialIDs["bos"][0] != 0 {
		t.Fatal("caller map aliases private secret")
	}
	mapped, err := secret.MapIDs([]int{0, 1, 2, 3, 1})
	if err != nil || !slices.Equal(mapped, []int{2, 0, 3, 1, 0}) {
		t.Fatal("original-to-physical lookup")
	}
	back, err := secret.RestoreIDs(mapped)
	if err != nil || !slices.Equal(back, []int{0, 1, 2, 3, 1}) {
		t.Fatal("inverse lookup")
	}
	logits, err := secret.RestoreLogits([]float64{-3, 2, 7, -12.5})
	if err != nil || !slices.Equal(logits, []float64{7, -3, -12.5, 2}) {
		t.Fatal("logits compared in wrong coordinates")
	}
	public := secret.publicBinding()
	if !slices.Equal(public.MappedSpecialIDs["eos"], []int{0, 1}) || public.MappedSpecialIDs["bos"][0] != 2 || public.MappedSpecialIDs["pad"][0] != 2 {
		t.Fatal("special alias mapping failed")
	}
	public.MappedSpecialIDs["bos"][0] = 3
	if secret.publicBinding().MappedSpecialIDs["bos"][0] != 2 {
		t.Fatal("returned public map aliases secret")
	}
	if _, err := json.Marshal(secret); err == nil {
		t.Fatal("private secret serialized through public API")
	}
	if fmt.Sprintf("%v %+v %#v", secret, secret, secret) != "[ClientSecret redacted] [ClientSecret redacted] [ClientSecret redacted]" {
		t.Fatal("secret formatting leaked material")
	}
	for _, ids := range [][]int{{-1}, {4}, make([]int, 4097)} {
		if _, err := secret.MapIDs(ids); err == nil {
			t.Fatal("invalid mapping accepted")
		}
		if _, err := secret.RestoreIDs(ids); err == nil {
			t.Fatal("invalid inverse accepted")
		}
	}
	for _, values := range [][]float64{{1}, {1, 2, math.NaN(), 4}, {1, 2, 3, math.Inf(1)}} {
		if _, err := secret.RestoreLogits(values); err == nil {
			t.Fatal("invalid logits accepted")
		}
	}
	var missing *Secret
	if _, err := missing.MapIDs([]int{0}); err == nil {
		t.Fatal("nil secret accepted")
	}
}

func TestCorruptedSecretMaterial(t *testing.T) {
	tests := map[string]func(*secretDisk){
		"version":              func(d *secretDisk) { d.SchemaVersion = 2 },
		"duplicate":            func(d *secretDisk) { d.Material.Permutation = []int{0, 0, 2, 3} },
		"short":                func(d *secretDisk) { d.Material.Permutation = []int{0, 1} },
		"negative":             func(d *secretDisk) { d.Material.Permutation = []int{-1, 0, 1, 2} },
		"out-of-range":         func(d *secretDisk) { d.Material.Permutation = []int{0, 1, 2, 4} },
		"nonce":                func(d *secretDisk) { d.Material.Nonce = "private-input" },
		"source":               func(d *secretDisk) { d.Material.Context.Source.ContentHash = "invalid" },
		"tokenizer":            func(d *secretDisk) { d.Material.Context.TokenizerFingerprint = "invalid" },
		"method":               func(d *secretDisk) { d.Material.Context.MethodRevision = "unknown" },
		"rng":                  func(d *secretDisk) { d.Material.Context.RNGAlgorithm = "math-rand" },
		"rng-revision":         func(d *secretDisk) { d.Material.Context.RNGRevision = "2" },
		"vocab":                func(d *secretDisk) { d.Material.Context.VocabularySize = 1 },
		"hidden":               func(d *secretDisk) { d.Material.Context.HiddenSize = 257 },
		"missing-special":      func(d *secretDisk) { delete(d.Material.Context.SpecialIDs, "eos") },
		"unknown-special":      func(d *secretDisk) { d.Material.Context.SpecialIDs["secret"] = []int{0} },
		"null-special":         func(d *secretDisk) { d.Material.Context.SpecialIDs["bos"] = nil },
		"duplicate-eos":        func(d *secretDisk) { d.Material.Context.SpecialIDs["eos"] = []int{1, 1} },
		"unsorted-eos":         func(d *secretDisk) { d.Material.Context.SpecialIDs["eos"] = []int{3, 1} },
		"out-of-range-special": func(d *secretDisk) { d.Material.Context.SpecialIDs["bos"] = []int{4} },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			d := fixedSecret(t, testContext()).disk
			mutate(&d)
			d.BindingID = commitment(d.Material)
			if _, err := validateSecret(d); err == nil {
				t.Fatal("corruption accepted despite self-consistent hash")
			}
		})
	}
	d := fixedSecret(t, testContext()).disk
	d.Material.Permutation = []int{0, 1, 2, 3}
	if _, err := validateSecret(d); err == nil {
		t.Fatal("permutation replacement without new commitment accepted")
	}
}

type brokenEntropy struct{}

func (brokenEntropy) Read([]byte) (int, error) { return 0, errors.New("private-entropy-error") }

func TestEntropyAndGeneratedInverse(t *testing.T) {
	for _, entropy := range []io.Reader{brokenEntropy{}, bytes.NewReader(make([]byte, 32))} {
		if _, err := generateSecret(testContext(), entropy); err == nil || strings.Contains(err.Error(), "private-entropy-error") {
			t.Fatal("entropy failure not redacted")
		}
	}
	for i := 0; i < 20; i++ {
		s, err := NewSecret(testContext())
		if err != nil {
			t.Fatal(err)
		}
		mapped, _ := s.MapIDs([]int{0, 1, 2, 3})
		back, _ := s.RestoreIDs(mapped)
		if !slices.Equal(back, []int{0, 1, 2, 3}) {
			t.Fatal("generated permutation has no inverse")
		}
	}
}

func TestPrivateSecretPersistence(t *testing.T) {
	root := privateDir(t)
	s := fixedSecret(t, testContext())
	if err := SaveSecret(root, s); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, secretFile)
	stat, _ := os.Stat(path)
	if stat.Mode().Perm() != 0600 {
		t.Fatal("secret not 0600")
	}
	loaded, err := LoadSecret(root)
	if err != nil || loaded.disk.BindingID != s.disk.BindingID {
		t.Fatal("stored secret changed", err)
	}
	before, _ := os.ReadFile(path)
	if err := SaveSecret(root, s); err == nil {
		t.Fatal("existing secret overwritten")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("overwrite changed secret")
	}
	if err := os.Chmod(path, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSecret(root); err == nil {
		t.Fatal("widely readable secret accepted")
	}
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(root, 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSecret(root); err == nil {
		t.Fatal("public root accepted")
	}
	if err := SaveSecret(root, s); err == nil {
		t.Fatal("public root writable")
	}
}

func TestSecretFileFailures(t *testing.T) {
	for _, kind := range []string{"missing", "symlink", "fifo", "directory", "oversized", "unknown", "duplicate", "noncanonical", "null"} {
		t.Run(kind, func(t *testing.T) {
			root := privateDir(t)
			path := filepath.Join(root, secretFile)
			data, _ := json.Marshal(fixedSecret(t, testContext()).disk)
			switch kind {
			case "symlink":
				outside := filepath.Join(t.TempDir(), "outside")
				os.WriteFile(outside, data, 0600)
				os.Symlink(outside, path)
			case "fifo":
				if err := unix.Mkfifo(path, 0600); err != nil {
					t.Fatal(err)
				}
			case "directory":
				os.Mkdir(path, 0700)
			case "oversized":
				os.WriteFile(path, make([]byte, MaxSecretBytes+1), 0600)
			case "unknown":
				os.WriteFile(path, append([]byte(`{"secret_extra":1,`), data[1:]...), 0600)
			case "duplicate":
				os.WriteFile(path, append([]byte(`{"schema_version":1,`), data[1:]...), 0600)
			case "noncanonical":
				os.WriteFile(path, append(data, '\n'), 0600)
			case "null":
				os.WriteFile(path, []byte("null"), 0600)
			}
			if _, err := LoadSecret(root); err == nil {
				t.Fatal("invalid secret file accepted")
			}
		})
	}
}

func TestConcurrentSecretCreationAndLookup(t *testing.T) {
	root := privateDir(t)
	secret := fixedSecret(t, testContext())
	var wg sync.WaitGroup
	results := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mapped, err := secret.MapIDs([]int{0, 1, 2, 3})
			if err != nil || !slices.Equal(mapped, []int{2, 0, 3, 1}) {
				results <- errors.New("concurrent mapping failed")
				return
			}
			results <- SaveSecret(root, secret)
		}()
	}
	wg.Wait()
	close(results)
	successes := 0
	for err := range results {
		if err == nil {
			successes++
		}
	}
	if successes != 1 {
		t.Fatal("secret creation was not exclusive")
	}
	loaded, err := LoadSecret(root)
	if err != nil || loaded.disk.BindingID != secret.disk.BindingID {
		t.Fatal("concurrent creation corrupted secret", err)
	}
}

func FuzzPermutationInverse(f *testing.F) {
	f.Add([]byte{2, 0, 3, 1})
	f.Add([]byte{0, 0})
	f.Add([]byte{1, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 2 || len(data) > 256 {
			return
		}
		context := testContext()
		context.VocabularySize = len(data)
		context.SpecialIDs = map[string][]int{"bos": {}, "eos": {0}, "pad": {0}}
		p := make([]int, len(data))
		seen := make(map[int]bool)
		valid := true
		for i, value := range data {
			p[i] = int(value)
			if p[i] >= len(data) || seen[p[i]] {
				valid = false
			}
			seen[p[i]] = true
		}
		m := material{Context: context, Nonce: strings.Repeat("0", 64), Permutation: p}
		s, err := validateSecret(secretDisk{SchemaVersion: 1, BindingID: commitment(m), Material: m})
		if !valid {
			if err == nil {
				t.Fatal("non-bijection accepted")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]int, len(data))
		for i := range ids {
			ids[i] = i
		}
		mapped, err := s.MapIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		back, err := s.RestoreIDs(mapped)
		if err != nil || !slices.Equal(ids, back) {
			t.Fatal("bijection inverse failed")
		}
	})
}
