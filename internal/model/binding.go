package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"

	"golang.org/x/sys/unix"
)

// SourceBinding distinguishes structural identity from actual checkpoint bytes.
// It is private compiler metadata, never a public server manifest field.
type SourceBinding struct {
	InventoryHash string `json:"inventory_sha256"`
	ContentHash   string `json:"content_sha256"`
}

// BindSource streams every source shard through a bounded hashing buffer. The
// trusted compiler must keep inputs immutable during binding and compilation,
// and recompute this binding before Resume; header identity alone is insufficient.
func BindSource(path string) (SourceBinding, error) {
	before, err := Inspect(path)
	if err != nil {
		return SourceBinding{}, err
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return SourceBinding{}, errors.New("cannot open source binding root")
	}
	defer root.Close()
	type digest struct {
		Name   string `json:"name"`
		SHA256 string `json:"sha256"`
	}
	files := make([]digest, 0, len(before.Shards))
	for _, shard := range before.Shards {
		file, err := root.OpenFile(shard.Name, os.O_RDONLY|unix.O_NONBLOCK, 0)
		if err != nil {
			return SourceBinding{}, errors.New("cannot open source binding shard")
		}
		stat, err := file.Stat()
		if err != nil || !stat.Mode().IsRegular() || stat.Size() != shard.FileBytes || stat.Size() == math.MaxInt64 {
			file.Close()
			return SourceBinding{}, errors.New("source binding shard changed or exceeds bounds")
		}
		sum := sha256.New()
		count, err := io.CopyBuffer(sum, io.LimitReader(file, stat.Size()+1), make([]byte, 32<<10))
		after, statErr := file.Stat()
		file.Close()
		if err != nil || statErr != nil || count != stat.Size() || stat.Size() != after.Size() || !stat.ModTime().Equal(after.ModTime()) {
			return SourceBinding{}, errors.New("source bytes changed or failed during binding")
		}
		files = append(files, digest{Name: shard.Name, SHA256: hex.EncodeToString(sum.Sum(nil))})
	}
	after, err := Inspect(path)
	if err != nil || after.Hash() != before.Hash() {
		return SourceBinding{}, errors.New("source metadata changed during binding")
	}
	// Include structural/config identity in the full content fingerprint.
	data := struct {
		InventoryHash string   `json:"inventory_sha256"`
		Files         []digest `json:"files"`
	}{before.Hash(), files}
	encoded, err := json.Marshal(data)
	if err != nil {
		return SourceBinding{}, errors.New("cannot encode source binding")
	}
	sum := sha256.Sum256(encoded)
	return SourceBinding{InventoryHash: before.Hash(), ContentHash: hex.EncodeToString(sum[:])}, nil
}
