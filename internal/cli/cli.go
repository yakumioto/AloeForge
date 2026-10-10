// Package cli exposes bounded structural inspection and artifact verification.
package cli

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/yakumioto/AloeForge/internal/artifact"
	"github.com/yakumioto/AloeForge/internal/model"
	"github.com/yakumioto/AloeForge/internal/version"
)

const usage = "Usage: aloeforge <version|inspect|verify|compile-token|verify-token|compile-dense|verify-dense>\n"
const inspectHelp = "Usage: aloeforge inspect --model-root PATH\n\nRead bounded HF/safetensors metadata only. Structural identification does not imply transform support.\n"
const verifyHelp = "Usage: aloeforge verify --artifact-root PATH\n\nVerify a P1/P2/P5 diagnostic-only artifact and stream its content hashes.\n"

// Run returns 0 on success, 1 for verification or I/O failures,
// and 2 for invalid usage. User-supplied arguments are never echoed.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	switch args[0] {
	case "compile-dense", "verify-dense":
		return denseCommand(args, stdout, stderr)
	case "compile-token", "verify-token":
		return tokenCommand(args, stdout, stderr)
	case "help", "--help", "-h":
		if len(args) == 1 {
			fmt.Fprint(stdout, usage)
			return 0
		}
	case "version":
		if len(args) == 1 {
			fmt.Fprintf(stdout, "aloeforge %s\n", version.Version)
			return 0
		}
	case "inspect":
		if len(args) == 2 && (args[1] == "--help" || args[1] == "-h") {
			fmt.Fprint(stdout, inspectHelp)
			return 0
		}
		if len(args) != 3 || args[1] != "--model-root" || args[2] == "" {
			fmt.Fprint(stderr, inspectHelp)
			return 2
		}
		inventory, err := model.Inspect(args[2])
		if err != nil {
			fmt.Fprintln(stderr, "inspection failed:", err)
			return 1
		}
		return summary(stdout, stderr, struct {
			SchemaVersion int    `json:"schema_version"`
			Architecture  string `json:"architecture"`
			Shards        int    `json:"shards"`
			Tensors       int    `json:"tensors"`
			InventoryHash string `json:"inventory_sha256"`
		}{inventory.SchemaVersion, inventory.Architecture, len(inventory.Shards), len(inventory.Tensors), inventory.Hash()})
	case "verify":
		if len(args) == 2 && (args[1] == "--help" || args[1] == "-h") {
			fmt.Fprint(stdout, verifyHelp)
			return 0
		}
		if len(args) != 3 || args[1] != "--artifact-root" || args[2] == "" {
			fmt.Fprint(stderr, verifyHelp)
			return 2
		}
		manifest, digest, err := artifact.Verify(args[2])
		if err != nil {
			fmt.Fprintln(stderr, "verification failed:", err)
			return 1
		}
		return summary(stdout, stderr, struct {
			Kind         string `json:"kind"`
			Files        int    `json:"files"`
			ManifestHash string `json:"manifest_sha256"`
		}{manifest.Kind, len(manifest.Files), digest})
	}
	fmt.Fprint(stderr, usage)
	return 2
}

func summary(stdout, stderr io.Writer, value any) int {
	if err := json.NewEncoder(stdout).Encode(value); err != nil {
		fmt.Fprintln(stderr, "cannot write verification summary")
		return 1
	}
	return 0
}
