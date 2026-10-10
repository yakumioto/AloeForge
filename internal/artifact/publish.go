package artifact

import (
	"errors"
	"io"
	"os"
	"strings"
)

func copyShard(from, to *os.Root, name string) error {
	source, err := openRegular(from, name)
	if err != nil {
		return err
	}
	defer source.Close()
	stat, err := source.Stat()
	if err != nil {
		return errors.New("cannot stat copy source")
	}
	target, err := to.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return errors.New("cannot create publication copy")
	}
	defer target.Close()
	count, err := io.CopyBuffer(target, io.NewSectionReader(source, 0, stat.Size()), make([]byte, 32<<10))
	if err != nil || count != stat.Size() {
		return errors.New("publication copy is incomplete")
	}
	if err := target.Sync(); err != nil {
		return errors.New("publication copy sync failed")
	}
	if err := target.Close(); err != nil {
		return errors.New("publication copy close failed")
	}
	return nil
}

// Publish creates and verifies a separate snapshot, then publishes it using a
// single no-replace rename. Private plan/checkpoint files never enter the copy.
// Retry after a lost acknowledgement succeeds only for the identical artifact.
func (stage *Stage) Publish(name string) (string, error) {
	stage.mu.Lock()
	defer stage.mu.Unlock()
	if stage.closed {
		return "", errors.New("staging session is closed")
	}
	if !identifier(name) || strings.HasPrefix(name, ".") {
		return "", errors.New("invalid publication name")
	}
	names := shardNames(stage.plan.Outputs)
	if len(stage.checkpoint.Completed) != len(names) {
		return "", errors.New("cannot publish incomplete staging outputs")
	}
	manifest := Manifest{SchemaVersion: 1, Kind: Kind, DescriptorHash: hash(stage.plan.Outputs), Outputs: stage.plan.Outputs}
	if stage.plan.Binding != nil {
		manifest.Kind = TokenKind
		manifest.Binding = stage.plan.Binding
	}
	byShard := groups(stage.plan.Outputs)
	for _, name := range names {
		digest, err := verifyShard(stage.root, name, byShard[name])
		if err != nil {
			return "", err
		}
		if stage.checkpoint.Completed[name] != digest {
			return "", errors.New("staging checkpoint content changed")
		}
		manifest.Files = append(manifest.Files, FileDigest{Name: name, SHA256: digest})
	}
	expected := hash(manifest)
	if stat, err := stage.parent.Lstat(name); err == nil {
		if !stat.IsDir() {
			return "", errors.New("publication target is not a directory or is a symlink")
		}
		existing, err := stage.parent.OpenRoot(name)
		if err != nil {
			return "", errors.New("cannot open publication target")
		}
		_, digest, err := verifyRoot(existing)
		existing.Close()
		if err != nil || digest != expected {
			return "", errors.New("refusing to replace a different or invalid artifact")
		}
		return digest, nil
	} else if !os.IsNotExist(err) {
		return "", errors.New("cannot inspect publication target")
	}
	temporary := ".publish-" + strings.TrimPrefix(stage.name, ".stage-")
	// Only our deterministic private snapshot name is cleaned, under the parent lock.
	if err := stage.parent.RemoveAll(temporary); err != nil {
		return "", errors.New("cannot discard stale publication snapshot")
	}
	if err := stage.parent.Mkdir(temporary, 0700); err != nil {
		return "", errors.New("cannot create publication snapshot")
	}
	defer stage.parent.RemoveAll(temporary)
	copyRoot, err := stage.parent.OpenRoot(temporary)
	if err != nil {
		return "", errors.New("cannot open publication snapshot")
	}
	defer copyRoot.Close()
	for _, name := range names {
		if err := copyShard(stage.root, copyRoot, name); err != nil {
			return "", err
		}
	}
	if err := atomicJSON(copyRoot, "manifest.json", manifest); err != nil {
		return "", err
	}
	_, digest, err := verifyRoot(copyRoot)
	if err != nil || digest != expected {
		return "", errors.New("publication snapshot verification failed")
	}
	if err := syncDirectory(stage.parent); err != nil {
		return "", err
	}
	if err := renameNoReplace(stage.parent, temporary, name); err != nil {
		return "", err
	}
	if err := syncDirectory(stage.parent); err != nil {
		return "", errors.New("artifact is visible but parent durability sync failed; verify and retry")
	}
	return digest, nil
}
