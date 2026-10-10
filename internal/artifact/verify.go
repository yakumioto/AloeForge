package artifact

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"slices"

	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/safetensors"
	"github.com/yakumioto/AloeForge/internal/strictjson"
	"golang.org/x/sys/unix"
)

func openRegular(root *os.Root, name string) (*os.File, error) {
	stat, err := root.Lstat(name)
	if err != nil || !stat.Mode().IsRegular() {
		return nil, errors.New("artifact file must be regular and not a symlink")
	}
	file, err := root.OpenFile(name, os.O_RDONLY|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, errors.New("cannot open artifact file")
	}
	current, err := file.Stat()
	if err != nil || !current.Mode().IsRegular() || !os.SameFile(stat, current) {
		file.Close()
		return nil, errors.New("artifact file changed during open")
	}
	return file, nil
}

func readControl(root *os.Root, name string, target any) error {
	file, err := openRegular(root, name)
	if err != nil {
		return err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, model.MaxMetadataBytes+1))
	if err != nil || len(data) > model.MaxMetadataBytes {
		return errors.New("artifact metadata exceeds bounds")
	}
	return strictjson.Decode(data, target, true)
}

func streamHash(reader io.Reader) (string, error) {
	digest := sha256.New()
	if _, err := io.CopyBuffer(digest, reader, make([]byte, 32<<10)); err != nil {
		return "", errors.New("cannot hash artifact bytes")
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func verifyShard(root *os.Root, name string, tensors []OutputTensor) (string, error) {
	file, err := openRegular(root, name)
	if err != nil {
		return "", err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return "", errors.New("cannot stat artifact shard")
	}
	header, err := safetensors.Read(file, stat.Size())
	if err != nil {
		return "", err
	}
	expected, err := makeHeader(tensors)
	if err != nil {
		return "", err
	}
	if header.DataStart != int64(8+len(expected)) || len(header.Tensors) != len(tensors) {
		return "", errors.New("output header differs from descriptor contract")
	}
	actual := make([]byte, len(expected))
	if _, err := file.ReadAt(actual, 8); err != nil || !bytes.Equal(actual, expected) {
		return "", errors.New("output header is not canonical or contains extra metadata")
	}
	byName := make(map[string]OutputTensor)
	for _, tensor := range tensors {
		byName[tensor.Name] = tensor
	}
	for _, tensor := range header.Tensors {
		expected := byName[tensor.Name]
		if expected.DType != tensor.DType || !slices.Equal(expected.Shape, tensor.Shape) {
			return "", errors.New("output tensor shape or dtype mismatch")
		}
		digest, err := streamHash(io.NewSectionReader(file, header.DataStart+tensor.Offsets[0], tensor.Offsets[1]-tensor.Offsets[0]))
		if err != nil || digest != expected.PayloadHash {
			return "", errors.New("output tensor content mismatch")
		}
	}
	return streamHash(file)
}

func verifyRoot(root *os.Root) (Manifest, string, error) {
	var manifest Manifest
	if err := readControl(root, "manifest.json", &manifest); err != nil {
		return manifest, "", err
	}
	if manifest.SchemaVersion != 1 || manifest.Kind != Kind {
		return manifest, "", errors.New("unsupported artifact kind or manifest version")
	}
	outputs, err := normalizeOutputs(manifest.Outputs)
	if err != nil {
		return manifest, "", err
	}
	if hash(outputs) != manifest.DescriptorHash || hash(outputs) != hash(manifest.Outputs) {
		return manifest, "", errors.New("manifest descriptors are not canonical or hash-bound")
	}
	names := shardNames(outputs)
	if len(manifest.Files) != len(names) {
		return manifest, "", errors.New("manifest shard completeness mismatch")
	}
	directory, err := root.Open(".")
	if err != nil {
		return manifest, "", errors.New("cannot list artifact directory")
	}
	entries, err := directory.ReadDir(model.MaxShards + 2)
	directory.Close()
	if err != nil && err != io.EOF || len(entries) != len(names)+1 {
		return manifest, "", errors.New("artifact contains missing or unexpected files")
	}
	allowed := map[string]bool{"manifest.json": true}
	for _, name := range names {
		allowed[name] = true
	}
	for _, entry := range entries {
		if !allowed[entry.Name()] || !entry.Type().IsRegular() {
			return manifest, "", errors.New("artifact contains unexpected files or symlinks")
		}
	}
	byShard := groups(outputs)
	for i, name := range names {
		if manifest.Files[i].Name != name || !validHash(manifest.Files[i].SHA256) {
			return manifest, "", errors.New("manifest file digests are not canonical")
		}
		digest, err := verifyShard(root, name, byShard[name])
		if err != nil {
			return manifest, "", err
		}
		if digest != manifest.Files[i].SHA256 {
			return manifest, "", errors.New("manifest file hash mismatch")
		}
	}
	return manifest, hash(manifest), nil
}

// Verify streams payload hashes through bounded buffers; no Python bypass exists.
func Verify(path string) (Manifest, string, error) {
	root, err := os.OpenRoot(path)
	if err != nil {
		return Manifest{}, "", errors.New("cannot open artifact root")
	}
	defer root.Close()
	return verifyRoot(root)
}
