// Package cli implements only the bootstrap command surface.
package cli

import (
	"fmt"
	"io"

	"github.com/yakumioto/AloeForge/internal/version"
)

const usage = "Usage: aloeforge <version|inspect>\n"
const inspectHelp = "Usage: aloeforge inspect --help\n\nModel inspection is not implemented in P0. No model files are opened.\n"

// Run returns 0 for informational commands, 1 for unimplemented operations,
// and 2 for invalid usage. User-supplied arguments are never echoed.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	switch args[0] {
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
		fmt.Fprintln(stderr, "inspect is not implemented in P0")
		return 1
	}
	fmt.Fprint(stderr, usage)
	return 2
}
