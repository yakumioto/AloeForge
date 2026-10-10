package artifact

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

func payloadHash(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }

func privateParent(t *testing.T) string {
	t.Helper()
	// testing.TempDir's child permissions depend on umask. Create a private
	// parent explicitly rather than weakening the production stage contract.
	root := filepath.Join(t.TempDir(), "private")
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	return root
}

func testPlan() Plan {
	return Plan{SchemaVersion: 1, SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("0", 64), MethodRevision: "p1-diagnostic", RNGAlgorithm: "none", RNGRevision: "none", Outputs: []OutputTensor{
		{Name: "changed.output", Shard: "out-1.safetensors", DType: "U8", Shape: []int64{2}, PayloadHash: payloadHash([]byte("ab"))},
		{Name: "other.output", Shard: "out-2.safetensors", DType: "F32", Shape: []int64{1}, PayloadHash: payloadHash(make([]byte, 4))},
	}}
}

func streams(name string) map[string]io.Reader {
	if name == "out-1.safetensors" {
		return map[string]io.Reader{"changed.output": bytes.NewReader([]byte("ab"))}
	}
	return map[string]io.Reader{"other.output": bytes.NewReader(make([]byte, 4))}
}

func complete(t *testing.T, stage *Stage) {
	t.Helper()
	for _, name := range shardNames(stage.plan.Outputs) {
		if stage.checkpoint.Completed[name] == "" {
			if err := stage.WriteShard(name, streams(name)); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestDescriptorGoldenAndBinding(t *testing.T) {
	plan := testPlan()
	digest, err := DescriptorHash(plan.Outputs)
	// Independently generated with Python hashlib/json, not production code.
	if err != nil || digest != "2c2956bdf9e08e86b30a8f2a902aa0e3f085778d9b1f3080a642260de84cbb6d" {
		t.Fatal("descriptor golden mismatch", digest, err)
	}
	slices.Reverse(plan.Outputs)
	reordered, err := DescriptorHash(plan.Outputs)
	if err != nil || reordered != digest {
		t.Fatal("descriptor order changed hash")
	}
	for _, change := range []func(*OutputTensor){func(v *OutputTensor) { v.Name = "new.name" }, func(v *OutputTensor) { v.Shape = []int64{2} }, func(v *OutputTensor) { v.DType = "I32" }, func(v *OutputTensor) { v.PayloadHash = strings.Repeat("f", 64) }, func(v *OutputTensor) { v.Shard = "new.safetensors" }} {
		other := testPlan()
		change(&other.Outputs[1])
		h, err := DescriptorHash(other.Outputs)
		if err != nil || h == digest {
			t.Fatal("output change not hash-bound", err)
		}
	}
}

func TestPublishVerifiedSnapshotAndIdempotence(t *testing.T) {
	parent := privateParent(t)
	plan := testPlan()
	stage, err := Create(parent, ".stage-demo", plan)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	// Mutation of caller-owned slices must not alter the bound staging contract.
	plan.Outputs[0].Shape[0] = 999
	complete(t, stage)
	digest, err := stage.Publish("demo")
	if err != nil {
		t.Fatal(err)
	}
	manifest, verified, err := Verify(filepath.Join(parent, "demo"))
	if err != nil || verified != digest || manifest.Kind != Kind || len(manifest.Files) != 2 {
		t.Fatal("published verification failed", err)
	}
	for _, name := range []string{"plan.json", "checkpoint.json"} {
		if _, err := os.Stat(filepath.Join(parent, "demo", name)); !os.IsNotExist(err) {
			t.Fatal("private control metadata published")
		}
	}
	if again, err := stage.Publish("demo"); err != nil || again != digest {
		t.Fatal("idempotent publish failed", err)
	}
	source, _ := os.Stat(filepath.Join(parent, ".stage-demo", "out-1.safetensors"))
	public, _ := os.Stat(filepath.Join(parent, "demo", "out-1.safetensors"))
	if os.SameFile(source, public) {
		t.Fatal("published output aliases writable staging bytes")
	}
	if err := os.WriteFile(filepath.Join(parent, ".stage-demo", "out-1.safetensors"), []byte("tampered"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Verify(filepath.Join(parent, "demo")); err != nil {
		t.Fatal("staging mutation changed published snapshot", err)
	}
	if _, err := stage.Publish("another"); err == nil {
		t.Fatal("tampered stage was published")
	}
	if _, err := os.Stat(filepath.Join(parent, "another")); !os.IsNotExist(err) {
		t.Fatal("verification failure made output visible")
	}
}

type failedReader struct{}

func (failedReader) Read([]byte) (int, error) { return 0, errors.New("reader failed") }

func TestWriteFailuresNeverPublish(t *testing.T) {
	for _, test := range []struct {
		name   string
		reader io.Reader
	}{
		{"short", bytes.NewReader([]byte("a"))}, {"extra", bytes.NewReader([]byte("abc"))},
		{"wrong content", bytes.NewReader([]byte("xy"))}, {"reader error", failedReader{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			parent := privateParent(t)
			stage, err := Create(parent, ".stage-fail", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			defer stage.Close()
			if err := stage.WriteShard("out-1.safetensors", map[string]io.Reader{"changed.output": test.reader}); err == nil {
				t.Fatal("invalid stream accepted")
			}
			if _, err := stage.Publish("visible"); err == nil {
				t.Fatal("incomplete stage published")
			}
			if _, err := os.Stat(filepath.Join(parent, "visible")); !os.IsNotExist(err) {
				t.Fatal("output visible after failure")
			}
			if _, err := os.Stat(filepath.Join(parent, ".stage-fail", "out-1.safetensors.part")); !os.IsNotExist(err) {
				t.Fatal("failed writer left a partial file")
			}
		})
	}
}

func TestRejectMismatchedCheckpointBindings(t *testing.T) {
	for _, change := range []struct {
		name   string
		mutate func(*Plan)
	}{
		{"source", func(p *Plan) { p.SourceInventoryHash = strings.Repeat("a", 64) }},
		{"source content", func(p *Plan) { p.SourceContentHash = strings.Repeat("b", 64) }},
		{"method", func(p *Plan) { p.MethodRevision = "different-method" }},
		{"rng", func(p *Plan) { p.RNGAlgorithm = "different-rng" }},
		{"rng revision", func(p *Plan) { p.RNGRevision = "different-revision" }},
		{"output", func(p *Plan) { p.Outputs[0].Shape = []int64{3} }},
	} {
		t.Run(change.name, func(t *testing.T) {
			parent := privateParent(t)
			stage, err := Create(parent, ".stage-binding", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			if err := stage.Close(); err != nil {
				t.Fatal(err)
			}
			plan := testPlan()
			change.mutate(&plan)
			if resumed, err := Resume(parent, ".stage-binding", plan); err == nil {
				resumed.Close()
				t.Fatal("wrong binding resumed")
			}
			if _, err := os.Stat(filepath.Join(parent, "visible")); !os.IsNotExist(err) {
				t.Fatal("binding failure published output")
			}
		})
	}
}

func TestRejectStageCorruption(t *testing.T) {
	for _, kind := range []string{"checkpoint version", "checkpoint hash", "checkpoint missing shard", "orphan corrupt", "unknown file", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			parent := privateParent(t)
			stage, err := Create(parent, ".stage-corrupt", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "checkpoint version":
				stage.checkpoint.SchemaVersion = 2
				err = atomicJSON(stage.root, "checkpoint.json", stage.checkpoint)
			case "checkpoint hash":
				stage.checkpoint.PlanHash = strings.Repeat("f", 64)
				err = atomicJSON(stage.root, "checkpoint.json", stage.checkpoint)
			case "checkpoint missing shard":
				stage.checkpoint.Completed["out-1.safetensors"] = strings.Repeat("a", 64)
				err = atomicJSON(stage.root, "checkpoint.json", stage.checkpoint)
			case "orphan corrupt":
				err = stage.root.WriteFile("out-1.safetensors", []byte("truncated"), 0600)
			case "unknown file":
				err = stage.root.WriteFile("unexpected", nil, 0600)
			case "symlink":
				err = stage.root.Symlink("plan.json", "out-1.safetensors")
			}
			if err != nil {
				t.Fatal(err)
			}
			stage.Close()
			if resumed, err := Resume(parent, ".stage-corrupt", testPlan()); err == nil {
				resumed.Close()
				t.Fatal("corrupt stage resumed")
			}
		})
	}
}

func TestNeverOverwriteDifferentTarget(t *testing.T) {
	for _, kind := range []string{"file", "directory", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			parent := privateParent(t)
			stage, err := Create(parent, ".stage-demo", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			defer stage.Close()
			complete(t, stage)
			target := filepath.Join(parent, "visible")
			switch kind {
			case "file":
				err = os.WriteFile(target, []byte("keep"), 0600)
			case "directory":
				err = os.Mkdir(target, 0700)
			case "symlink":
				err = os.Symlink(privateParent(t), target)
			}
			if err != nil {
				t.Fatal(err)
			}
			before, err := os.Lstat(target)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := stage.Publish("visible"); err == nil {
				t.Fatal("different target overwritten")
			}
			after, err := os.Lstat(target)
			if err != nil || !os.SameFile(before, after) {
				t.Fatal("target changed", err)
			}
		})
	}
}

func TestVerificationRejectsTampering(t *testing.T) {
	for _, kind := range []string{"payload", "manifest hash", "unknown manifest field", "extra file", "symlink", "private tensor metadata"} {
		t.Run(kind, func(t *testing.T) {
			parent := privateParent(t)
			stage, err := Create(parent, ".stage-demo", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			complete(t, stage)
			if _, err := stage.Publish("visible"); err != nil {
				t.Fatal(err)
			}
			stage.Close()
			root, err := os.OpenRoot(filepath.Join(parent, "visible"))
			if err != nil {
				t.Fatal(err)
			}
			defer root.Close()
			switch kind {
			case "payload":
				data, err := root.ReadFile("out-1.safetensors")
				if err != nil {
					t.Fatal(err)
				}
				data[len(data)-1] ^= 1
				err = root.WriteFile("out-1.safetensors", data, 0600)
				if err != nil {
					t.Fatal(err)
				}
			case "manifest hash":
				var m Manifest
				if err := readControl(root, "manifest.json", &m); err != nil {
					t.Fatal(err)
				}
				m.Files[0].SHA256 = strings.Repeat("f", 64)
				if err := atomicJSON(root, "manifest.json", m); err != nil {
					t.Fatal(err)
				}
			case "unknown manifest field":
				data, err := root.ReadFile("manifest.json")
				if err != nil {
					t.Fatal(err)
				}
				data = append([]byte(`{"unexpected":null,`), data[1:]...)
				if err := root.WriteFile("manifest.json", data, 0600); err != nil {
					t.Fatal(err)
				}
			case "extra file":
				if err := root.WriteFile("private-control", nil, 0600); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := root.Remove("out-1.safetensors"); err != nil {
					t.Fatal(err)
				}
				if err := root.Symlink("../.stage-demo/out-1.safetensors", "out-1.safetensors"); err != nil {
					t.Fatal(err)
				}
			case "private tensor metadata":
				data, err := root.ReadFile("out-1.safetensors")
				if err != nil {
					t.Fatal(err)
				}
				length := int(binary.LittleEndian.Uint64(data[:8]))
				header := append([]byte(`{"__metadata__":{"private":"synthetic-control"},`), bytes.TrimRight(data[8:8+length], " ")[1:]...)
				for len(header)%8 != 0 {
					header = append(header, ' ')
				}
				changed := make([]byte, 8)
				binary.LittleEndian.PutUint64(changed, uint64(len(header)))
				changed = append(changed, header...)
				data = append(changed, data[8+length:]...)
				if err := root.WriteFile("out-1.safetensors", data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, _, err := Verify(filepath.Join(parent, "visible")); err == nil {
				t.Fatal("tampered artifact verified")
			}
		})
	}
}

func TestInvalidPlansAndWriterRequests(t *testing.T) {
	for _, mutate := range []func(*Plan){
		func(p *Plan) { p.SchemaVersion = 2 }, func(p *Plan) { p.SourceInventoryHash = "invalid" }, func(p *Plan) { p.MethodRevision = "" },
		func(p *Plan) { p.Outputs = nil }, func(p *Plan) { p.Outputs[0].Shard = "../outside.safetensors" },
		func(p *Plan) { p.Outputs[0].PayloadHash = "not-a-digest" }, func(p *Plan) { p.Outputs[0].Shape = nil },
		func(p *Plan) { p.Outputs[1].Name = p.Outputs[0].Name }, func(p *Plan) { p.Outputs[0].DType = "FP8" },
	} {
		plan := testPlan()
		mutate(&plan)
		if stage, err := Create(privateParent(t), ".stage-invalid", plan); err == nil {
			stage.Close()
			t.Fatal("invalid plan accepted")
		}
	}
	parent := privateParent(t)
	if stage, err := Create(parent, "../escape", testPlan()); err == nil {
		stage.Close()
		t.Fatal("staging traversal accepted")
	}
	if err := os.Chmod(parent, 0755); err != nil {
		t.Fatal(err)
	}
	if stage, err := Create(parent, ".stage-invalid", testPlan()); err == nil {
		stage.Close()
		t.Fatal("nonprivate parent accepted")
	}
	if err := os.Chmod(parent, 0700); err != nil {
		t.Fatal(err)
	}
	stage, err := Create(parent, ".stage-valid", testPlan())
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if err := stage.WriteShard("out-1.safetensors", map[string]io.Reader{"wrong": bytes.NewReader([]byte("ab"))}); err == nil {
		t.Fatal("wrong output name accepted")
	}
	if err := stage.WriteShard("unknown.safetensors", streams("out-1.safetensors")); err == nil {
		t.Fatal("unknown shard accepted")
	}
	if err := stage.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stage.WriteShard("out-1.safetensors", streams("out-1.safetensors")); err == nil {
		t.Fatal("closed session accepted write")
	}
	if _, err := stage.Publish("visible"); err == nil {
		t.Fatal("closed session published")
	}
}

func TestScalarAndEmptyOutputs(t *testing.T) {
	plan := testPlan()
	plan.Outputs = []OutputTensor{
		{Name: "empty", Shard: "out.safetensors", DType: "U8", Shape: []int64{0}, PayloadHash: payloadHash(nil)},
		{Name: "scalar", Shard: "out.safetensors", DType: "U8", Shape: []int64{}, PayloadHash: payloadHash([]byte{1})},
	}
	parent := privateParent(t)
	stage, err := Create(parent, ".stage-zero", plan)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if err := stage.WriteShard("out.safetensors", map[string]io.Reader{"empty": bytes.NewReader(nil), "scalar": bytes.NewReader([]byte{1})}); err != nil {
		t.Fatal(err)
	}
	if _, err := stage.Publish("zero"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Verify(filepath.Join(parent, "zero")); err != nil {
		t.Fatal(err)
	}
}

func TestLocksAndConcurrentWrites(t *testing.T) {
	parent := privateParent(t)
	stage, err := Create(parent, ".stage-demo", testPlan())
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	if other, err := Create(parent, ".stage-other", testPlan()); err == nil {
		other.Close()
		t.Fatal("concurrent session bypassed lock")
	}
	var group sync.WaitGroup
	errors := make(chan error, 2)
	for range 2 {
		group.Add(1)
		go func() {
			defer group.Done()
			errors <- stage.WriteShard("out-1.safetensors", streams("out-1.safetensors"))
		}()
	}
	group.Wait()
	close(errors)
	successes := 0
	for err := range errors {
		if err == nil {
			successes++
		}
	}
	if successes != 1 {
		t.Fatal("write serialization failed")
	}
}

type crashReader struct{ read bool }

func (r *crashReader) Read(buffer []byte) (int, error) {
	if r.read {
		os.Exit(77)
	}
	r.read = true
	buffer[0] = 'a'
	return 1, nil
}

// Child intentionally exits without Close: the OS must release the flock.
func TestCrashHelper(t *testing.T) {
	parent := os.Getenv("ALOE_TEST_CRASH_PARENT")
	if parent == "" {
		return
	}
	stage, err := Create(parent, ".stage-recovery", testPlan())
	if err != nil {
		t.Fatal(err)
	}
	point := os.Getenv("ALOE_TEST_CRASH_POINT")
	if point == "created" {
		os.Exit(77)
	}
	if point == "partial" {
		stage.WriteShard("out-1.safetensors", map[string]io.Reader{"changed.output": &crashReader{}})
		t.Fatal("crash reader returned")
	}
	if err := stage.WriteShard("out-1.safetensors", streams("out-1.safetensors")); err != nil {
		t.Fatal(err)
	}
	if point == "orphan" {
		stage.checkpoint.Completed = map[string]string{}
		if err := atomicJSON(stage.root, "checkpoint.json", stage.checkpoint); err != nil {
			t.Fatal(err)
		}
	}
	if point == "copy" {
		complete(t, stage)
		if err := os.Mkdir(filepath.Join(parent, ".publish-recovery"), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(parent, ".publish-recovery", "out-1.safetensors"), []byte("partial copy"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if point == "published" {
		complete(t, stage)
		if _, err := stage.Publish("visible"); err != nil {
			t.Fatal(err)
		}
	}
	os.Exit(77)
}

func TestProcessCrashAndRestart(t *testing.T) {
	for _, point := range []string{"created", "partial", "orphan", "checkpoint", "copy", "published"} {
		t.Run(point, func(t *testing.T) {
			parent := privateParent(t)
			child := exec.Command(os.Args[0], "-test.run=^TestCrashHelper$")
			child.Env = append(os.Environ(), "ALOE_TEST_CRASH_PARENT="+parent, "ALOE_TEST_CRASH_POINT="+point)
			output, err := child.CombinedOutput()
			var status *exec.ExitError
			if !errors.As(err, &status) || status.ExitCode() != 77 {
				t.Fatalf("child did not crash as expected: %v %s", err, output)
			}
			stage, err := Resume(parent, ".stage-recovery", testPlan())
			if err != nil {
				t.Fatal(err)
			}
			defer stage.Close()
			complete(t, stage)
			digest, err := stage.Publish("visible")
			if err != nil {
				t.Fatal(err)
			}
			if _, verified, err := Verify(filepath.Join(parent, "visible")); err != nil || verified != digest {
				t.Fatal("recovered publication invalid", err)
			}
		})
	}
}
