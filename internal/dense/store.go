package dense

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"

	"github.com/yakumioto/AloeForge/internal/strictjson"
	"golang.org/x/sys/unix"
)

const MaxSecretBytes = 8 << 20
const secretFile = "client-secret.json"

func privateRoot(path string) (*os.Root, error) {
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, errors.New("cannot open Client Secret directory")
	}
	stat, err := root.Stat(".")
	if err != nil || stat.Mode().Perm()&0077 != 0 {
		root.Close()
		return nil, errors.New("Client Secret directory must be private (0700)")
	}
	return root, nil
}

func syncRoot(root *os.Root) error {
	dir, err := root.Open(".")
	if err != nil {
		return errors.New("cannot open Client Secret directory for sync")
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return errors.New("Client Secret directory sync failed")
	}
	return nil
}

// SaveSecret is fsynced, serialized by flock and atomically no-replace. It never
// puts private material in the staging/output directories or replaces a secret.
func SaveSecret(path string, secret *Secret) error {
	if secret == nil {
		return errors.New("missing Client Secret")
	}
	if _, err := validateSecret(secret.disk); err != nil {
		return err
	}
	root, err := privateRoot(path)
	if err != nil {
		return err
	}
	defer root.Close()
	lock, err := root.OpenFile(".client-secret.lock", os.O_CREATE|os.O_RDWR|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0600)
	if err != nil {
		return errors.New("cannot open Client Secret lock")
	}
	defer lock.Close()
	stat, err := lock.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Mode().Perm()&0077 != 0 {
		return errors.New("invalid Client Secret lock")
	}
	if err := unix.Flock(int(lock.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		return errors.New("Client Secret directory is locked")
	}
	if _, err := root.Lstat(secretFile); err == nil || !os.IsNotExist(err) {
		return errors.New("Client Secret already exists or cannot be inspected")
	}
	data, err := json.Marshal(secret.disk)
	if err != nil || len(data) > MaxSecretBytes {
		return errors.New("Client Secret exceeds bounds")
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return errors.New("Client Secret temporary entropy failed")
	}
	temporary := ".secret-" + hex.EncodeToString(nonce[:])
	file, err := root.OpenFile(temporary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return errors.New("cannot create Client Secret temporary")
	}
	defer func() { file.Close(); root.Remove(temporary) }()
	if _, err := file.Write(data); err != nil {
		return errors.New("Client Secret write failed")
	}
	if err := file.Sync(); err != nil {
		return errors.New("Client Secret sync failed")
	}
	if err := file.Close(); err != nil {
		return errors.New("Client Secret close failed")
	}
	dir, err := root.Open(".")
	if err != nil {
		return errors.New("cannot open Client Secret rename directory")
	}
	defer dir.Close()
	if err := unix.Renameat2(int(dir.Fd()), temporary, int(dir.Fd()), secretFile, unix.RENAME_NOREPLACE); err != nil {
		return errors.New("Client Secret atomic creation failed")
	}
	if err := syncRoot(root); err != nil {
		return errors.New("Client Secret is visible but durability sync failed; reload before retry")
	}
	return nil
}

func LoadSecret(path string) (*Secret, error) {
	root, err := privateRoot(path)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	file, err := root.OpenFile(secretFile, os.O_RDONLY|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, errors.New("cannot open Client Secret file")
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Mode().Perm()&0077 != 0 || stat.Size() > MaxSecretBytes {
		return nil, errors.New("invalid Client Secret file type, permissions or size")
	}
	data, err := io.ReadAll(io.LimitReader(file, MaxSecretBytes+1))
	if err != nil || len(data) > MaxSecretBytes {
		return nil, errors.New("Client Secret read exceeds bounds")
	}
	var disk secretDisk
	if err := strictjson.Decode(data, &disk, true); err != nil {
		return nil, errors.New("invalid Client Secret JSON")
	}
	canonical, err := json.Marshal(disk)
	if err != nil || !bytes.Equal(canonical, data) {
		return nil, errors.New("Client Secret JSON is not canonical")
	}
	return validateSecret(disk)
}

// Reestablish durability before creating/resuming a stage, including recovery
// after a previous no-replace rename succeeded but directory fsync failed.
func syncStoredSecret(path string) error {
	root, err := privateRoot(path)
	if err != nil {
		return err
	}
	defer root.Close()
	file, err := root.OpenFile(secretFile, os.O_RDONLY|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0)
	if err != nil {
		return errors.New("cannot open persisted Client Secret for sync")
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Mode().Perm()&0077 != 0 {
		return errors.New("invalid persisted Client Secret for sync")
	}
	if err := file.Sync(); err != nil {
		return errors.New("persisted Client Secret sync failed")
	}
	return syncRoot(root)
}
