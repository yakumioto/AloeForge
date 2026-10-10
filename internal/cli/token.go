package cli

import (
	"fmt"
	"io"

	"github.com/yakumioto/AloeForge/internal/tokenbaseline"
)

const compileTokenHelp = "Usage: aloeforge compile-token --model-root PATH --output-parent PATH --secret-root PATH --stage NAME --target NAME [--resume]\n\nCompile a synthetic token-permutation diagnostic fixture only. Output and secret directories must be private and disjoint.\n"
const verifyTokenHelp = "Usage: aloeforge verify-token --artifact-root PATH --secret-root PATH --artifact-id SHA256 --tokenizer-sha256 SHA256\n\nTrusted client verification of a P2 diagnostic artifact against external identities.\n"

// parseOptions accepts each known option exactly once, with no error echo of
// paths, arguments or private metadata. Values remain literal strings.
func parseOptions(args []string, required []string, allowResume bool) (map[string]string, bool) {
	allowed := make(map[string]bool)
	for _, name := range required {
		allowed[name] = true
	}
	if allowResume {
		allowed["--resume"] = true
	}
	values := make(map[string]string)
	for i := 0; i < len(args); i++ {
		name := args[i]
		if !allowed[name] {
			return nil, false
		}
		if _, exists := values[name]; exists {
			return nil, false
		}
		if name == "--resume" {
			values[name] = "true"
			continue
		}
		if i+1 == len(args) || args[i+1] == "" {
			return nil, false
		}
		i++
		values[name] = args[i]
	}
	for _, name := range required {
		if values[name] == "" {
			return nil, false
		}
	}
	return values, true
}

func tokenCommand(args []string, stdout, stderr io.Writer) int {
	help := compileTokenHelp
	if args[0] == "verify-token" {
		help = verifyTokenHelp
	}
	if len(args) == 2 && (args[1] == "--help" || args[1] == "-h") {
		fmt.Fprint(stdout, help)
		return 0
	}
	var result tokenbaseline.Result
	var err error
	if args[0] == "compile-token" {
		flags, ok := parseOptions(args[1:], []string{"--model-root", "--output-parent", "--secret-root", "--stage", "--target"}, true)
		if !ok {
			fmt.Fprint(stderr, help)
			return 2
		}
		result, err = tokenbaseline.Compile(tokenbaseline.Options{SourceRoot: flags["--model-root"], OutputParent: flags["--output-parent"], SecretRoot: flags["--secret-root"], StageName: flags["--stage"], TargetName: flags["--target"], Resume: flags["--resume"] == "true"})
	} else {
		flags, ok := parseOptions(args[1:], []string{"--artifact-root", "--secret-root", "--artifact-id", "--tokenizer-sha256"}, false)
		if !ok {
			fmt.Fprint(stderr, help)
			return 2
		}
		result, err = tokenbaseline.VerifyClient(flags["--artifact-root"], flags["--secret-root"], flags["--artifact-id"], flags["--tokenizer-sha256"])
	}
	if err != nil {
		fmt.Fprintln(stderr, "token baseline failed:", err)
		return 1
	}
	return summary(stdout, stderr, result)
}
