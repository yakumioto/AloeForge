package artifact

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/safetensors"
	"golang.org/x/sys/unix"
)

// Stage serializes operations and holds a kernel lock released even on process
// death. The trusted compiler must own the private parent; hostile local writers
// and filesystems/mounts are outside the P1 threat model.
type Stage struct {
	mu         sync.Mutex
	parent     *os.Root
	root       *os.Root
	lock       *os.File
	name       string
	plan       Plan
	checkpoint checkpoint
	closed     bool
}

func session(parentPath, name string, plan Plan) (*Stage, error) {
	if !strings.HasPrefix(name, ".stage-") || !identifier(strings.TrimPrefix(name, ".stage-")) {
		return nil, errors.New("invalid private staging name")
	}
	plan, err := normalizePlan(plan)
	if err != nil {
		return nil, err
	}
	parent, err := os.OpenRoot(parentPath)
	if err != nil {
		return nil, errors.New("cannot open staging parent")
	}
	stat, err := parent.Stat(".")
	if err != nil || stat.Mode().Perm()&0077 != 0 {
		parent.Close()
		return nil, errors.New("staging parent must be private (0700)")
	}
	lock, err := parent.OpenFile(".aloeforge.lock", os.O_CREATE|os.O_RDWR|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0600)
	if err != nil {
		parent.Close()
		return nil, errors.New("cannot open staging lock")
	}
	lockStat, err := lock.Stat()
	if err != nil || !lockStat.Mode().IsRegular() {
		lock.Close()
		parent.Close()
		return nil, errors.New("staging lock must be regular")
	}
	if err := unix.Flock(int(lock.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		lock.Close()
		parent.Close()
		return nil, errors.New("staging parent is locked by another session")
	}
	return &Stage{parent: parent, lock: lock, name: name, plan: plan}, nil
}

func syncDirectory(root *os.Root) error {
	directory, err := root.Open(".")
	if err != nil {
		return errors.New("cannot open directory for sync")
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return errors.New("directory sync failed")
	}
	return nil
}

func atomicJSON(root *os.Root, name string, value any) error {
	data, err := json.Marshal(value)
	if err != nil || len(data) > model.MaxMetadataBytes {
		return errors.New("control metadata exceeds bounds")
	}
	temporary := name + ".next"
	if err := root.Remove(temporary); err != nil && !os.IsNotExist(err) {
		return errors.New("cannot remove stale control temporary")
	}
	file, err := root.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return errors.New("cannot create control temporary")
	}
	defer file.Close()
	if _, err := file.Write(data); err != nil {
		return errors.New("cannot write control temporary")
	}
	if err := file.Sync(); err != nil {
		return errors.New("control sync failed")
	}
	if err := file.Close(); err != nil {
		return errors.New("control close failed")
	}
	if err := root.Rename(temporary, name); err != nil {
		return errors.New("control rename failed")
	}
	return syncDirectory(root)
}

// Create never writes to a published directory; existing stages are not replaced.
func Create(parentPath, name string, plan Plan) (*Stage, error) {
	stage, err := session(parentPath, name, plan)
	if err != nil {
		return nil, err
	}
	if err := stage.parent.Mkdir(name, 0700); err != nil {
		stage.Close()
		return nil, errors.New("staging directory already exists or cannot be created")
	}
	stage.root, err = stage.parent.OpenRoot(name)
	if err != nil {
		stage.Close()
		return nil, errors.New("cannot open staging directory")
	}
	stage.checkpoint = checkpoint{SchemaVersion: 1, PlanHash: hash(stage.plan), SourceInventoryHash: stage.plan.SourceInventoryHash, SourceContentHash: stage.plan.SourceContentHash, Completed: map[string]string{}}
	for _, item := range []struct {
		name  string
		value any
	}{{"plan.json", stage.plan}, {"checkpoint.json", stage.checkpoint}} {
		if err := atomicJSON(stage.root, item.name, item.value); err != nil {
			stage.Close()
			return nil, err
		}
	}
	if err := syncDirectory(stage.parent); err != nil {
		stage.Close()
		return nil, err
	}
	return stage, nil
}

// Resume revalidates every completed/orphan shard against the expected plan.
// Incomplete .part files are discarded; a fully verified orphan is adopted.
func Resume(parentPath, name string, plan Plan) (*Stage, error) {
	stage, err := session(parentPath, name, plan)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*Stage, error) { stage.Close(); return nil, err }
	stat, err := stage.parent.Lstat(name)
	if err != nil || !stat.IsDir() {
		return fail(errors.New("staging directory is missing or is a symlink"))
	}
	stage.root, err = stage.parent.OpenRoot(name)
	if err != nil {
		return fail(errors.New("cannot open staging directory"))
	}
	var stored Plan
	if err := readControl(stage.root, "plan.json", &stored); err != nil {
		return fail(err)
	}
	if hash(stored) != hash(stage.plan) {
		return fail(errors.New("staging plan/source/method/RNG binding mismatch"))
	}
	if err := readControl(stage.root, "checkpoint.json", &stage.checkpoint); err != nil {
		return fail(err)
	}
	cp := stage.checkpoint
	if cp.SchemaVersion != 1 || cp.Completed == nil || cp.PlanHash != hash(stage.plan) || cp.SourceInventoryHash != stage.plan.SourceInventoryHash || cp.SourceContentHash != stage.plan.SourceContentHash {
		return fail(errors.New("checkpoint binding mismatch"))
	}
	byShard := groups(stage.plan.Outputs)
	allowed := map[string]bool{"plan.json": true, "checkpoint.json": true, "checkpoint.json.next": true, "plan.json.next": true}
	for name := range byShard {
		allowed[name] = true
		allowed[name+".part"] = true
	}
	directory, err := stage.root.Open(".")
	if err != nil {
		return fail(errors.New("cannot list staging directory"))
	}
	entries, err := directory.ReadDir(2*model.MaxShards + 5)
	directory.Close()
	if err != nil && err != io.EOF || len(entries) > len(allowed) {
		return fail(errors.New("staging listing exceeds bounds"))
	}
	for _, entry := range entries {
		if !allowed[entry.Name()] || !entry.Type().IsRegular() {
			return fail(errors.New("unexpected staging file or symlink"))
		}
	}
	for name, digest := range cp.Completed {
		if byShard[name] == nil || !validHash(digest) {
			return fail(errors.New("checkpoint contains an invalid shard"))
		}
	}
	for _, name := range shardNames(stage.plan.Outputs) {
		_, err := stage.root.Lstat(name)
		if err == nil {
			digest, err := verifyShard(stage.root, name, byShard[name])
			if err != nil {
				return fail(err)
			}
			if recorded, ok := cp.Completed[name]; ok && recorded != digest {
				return fail(errors.New("checkpoint shard hash mismatch"))
			}
			stage.checkpoint.Completed[name] = digest
		} else if !os.IsNotExist(err) || cp.Completed[name] != "" {
			return fail(errors.New("checkpoint shard is missing"))
		}
		if err := stage.root.Remove(name + ".part"); err != nil && !os.IsNotExist(err) {
			return fail(errors.New("cannot discard partial shard"))
		}
	}
	if err := stage.root.Remove("plan.json.next"); err != nil && !os.IsNotExist(err) {
		return fail(errors.New("cannot discard control temporary"))
	}
	if err := atomicJSON(stage.root, "checkpoint.json", stage.checkpoint); err != nil {
		return fail(err)
	}
	return stage, nil
}

func renameNoReplace(root *os.Root, from, to string) error {
	directory, err := root.Open(".")
	if err != nil {
		return errors.New("cannot open rename directory")
	}
	defer directory.Close()
	if err := unix.Renameat2(int(directory.Fd()), from, int(directory.Fd()), to, unix.RENAME_NOREPLACE); err != nil {
		return errors.New("atomic rename failed (target exists or filesystem unsupported)")
	}
	return nil
}

// WriteShard consumes exactly the declared bytes from each finite reader. It
// rejects trailing data/content mismatches before checkpointing a complete file.
func (stage *Stage) WriteShard(name string, readers map[string]io.Reader) error {
	stage.mu.Lock()
	defer stage.mu.Unlock()
	if stage.closed {
		return errors.New("staging session is closed")
	}
	tensors := groups(stage.plan.Outputs)[name]
	if len(tensors) == 0 || len(readers) != len(tensors) || stage.checkpoint.Completed[name] != "" {
		return errors.New("invalid or already completed shard request")
	}
	for _, tensor := range tensors {
		if readers[tensor.Name] == nil {
			return errors.New("missing output tensor stream")
		}
	}
	header, err := makeHeader(tensors)
	if err != nil {
		return err
	}
	part := name + ".part"
	file, err := stage.root.OpenFile(part, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return errors.New("cannot create private partial shard")
	}
	defer func() { file.Close(); stage.root.Remove(part) }()
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], uint64(len(header)))
	if _, err := file.Write(prefix[:]); err != nil {
		return errors.New("cannot write shard prefix")
	}
	if _, err := file.Write(header); err != nil {
		return errors.New("cannot write shard header")
	}
	for _, tensor := range tensors {
		size, _ := safetensors.ByteSize(tensor.DType, tensor.Shape)
		digest := sha256.New()
		reader := readers[tensor.Name]
		count, err := io.CopyBuffer(io.MultiWriter(file, digest), io.LimitReader(reader, size), make([]byte, 32<<10))
		if err != nil || count != size {
			return errors.New("output stream is truncated or failed")
		}
		var extra [1]byte
		if count, err := io.ReadFull(reader, extra[:]); count != 0 || err != io.EOF {
			return errors.New("output stream has trailing data or failed")
		}
		if hex.EncodeToString(digest.Sum(nil)) != tensor.PayloadHash {
			return errors.New("output stream content hash mismatch")
		}
	}
	if err := file.Sync(); err != nil {
		return errors.New("shard sync failed")
	}
	if err := file.Close(); err != nil {
		return errors.New("shard close failed")
	}
	if err := renameNoReplace(stage.root, part, name); err != nil {
		return err
	}
	if err := syncDirectory(stage.root); err != nil {
		return err
	}
	digest, err := verifyShard(stage.root, name, tensors)
	if err != nil {
		return err
	}
	stage.checkpoint.Completed[name] = digest
	return atomicJSON(stage.root, "checkpoint.json", stage.checkpoint)
}

// Close releases the process lock. Persistent staging files remain for Resume.
func (stage *Stage) Close() error {
	stage.mu.Lock()
	defer stage.mu.Unlock()
	if stage.closed {
		return nil
	}
	stage.closed = true
	var failed bool
	if stage.root != nil {
		failed = stage.root.Close() != nil
	}
	if stage.lock != nil {
		failed = stage.lock.Close() != nil || failed
	}
	if stage.parent != nil {
		failed = stage.parent.Close() != nil || failed
	}
	if failed {
		return errors.New("staging session close failed")
	}
	return nil
}
