package cli

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/yakumioto/AloeForge/internal/dense"
)

const denseHelp = "Usage: aloeforge compile-dense --model-root PATH --output-parent PATH --secret-root PATH --stage NAME --target NAME --python PATH --expansion N [--resume]\n\nCompile the Qwen2.5 exact-isometry noise-off diagnostic graph. Production/paper profiles are blocked.\n"
const verifyDenseHelp = "Usage: aloeforge verify-dense --artifact-root PATH --secret-root PATH --artifact-id SHA256 --tokenizer-sha256 SHA256\n\nTrusted client verification against external P5 diagnostic identities.\n"

func denseCommand(args []string, stdout, stderr io.Writer) int {
	if args[0] == "verify-dense" {
		if len(args) == 2 && (args[1] == "--help" || args[1] == "-h") {
			fmt.Fprint(stdout, verifyDenseHelp)
			return 0
		}
		flags, ok := parseOptions(args[1:], []string{"--artifact-root", "--secret-root", "--artifact-id", "--tokenizer-sha256"}, false)
		if !ok {
			fmt.Fprint(stderr, verifyDenseHelp)
			return 2
		}
		result, err := dense.VerifyClient(flags["--artifact-root"], flags["--secret-root"], flags["--artifact-id"], flags["--tokenizer-sha256"])
		if err != nil {
			fmt.Fprintln(stderr, "dense verification failed:", err)
			return 1
		}
		return summary(stdout, stderr, result)
	}
	if len(args) == 2 && (args[1] == "--help" || args[1] == "-h") {
		fmt.Fprint(stdout, denseHelp)
		return 0
	}
	flags, ok := parseOptions(args[1:], []string{"--model-root", "--output-parent", "--secret-root", "--stage", "--target", "--python", "--expansion"}, true)
	if !ok {
		fmt.Fprint(stderr, denseHelp)
		return 2
	}
	expansion, err := strconv.Atoi(flags["--expansion"])
	if err != nil {
		fmt.Fprint(stderr, denseHelp)
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	result, err := dense.Compile(ctx, dense.Options{SourceRoot: flags["--model-root"], OutputParent: flags["--output-parent"], SecretRoot: flags["--secret-root"], StageName: flags["--stage"], TargetName: flags["--target"], Python: flags["--python"], Expansion: expansion, Resume: flags["--resume"] == "true"})
	if err != nil {
		fmt.Fprintln(stderr, "dense compilation failed:", err)
		return 1
	}
	return summary(stdout, stderr, result)
}
