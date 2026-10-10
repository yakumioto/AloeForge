package cli

import (
	"bytes"
	"strings"
	"testing"
)

func TestTokenCommandUsageAndRedaction(t *testing.T) {
	for _, args := range [][]string{{"compile-token"}, {"verify-token"}, {"compile-token", "--private-input", "private-input"}, {"verify-token", "--artifact-root", "private-input"}, {"compile-token", "--resume", "--resume"}} {
		var out, err bytes.Buffer
		if Run(args, &out, &err) != 2 || out.Len() != 0 || strings.Contains(err.String(), "private-input") {
			t.Fatal("invalid command not bounded/redacted")
		}
	}
	for _, command := range []string{"compile-token", "verify-token"} {
		var out, err bytes.Buffer
		if Run([]string{command, "--help"}, &out, &err) != 0 || out.Len() == 0 || err.Len() != 0 {
			t.Fatal("help command failed")
		}
	}
	var out, err bytes.Buffer
	args := []string{"compile-token", "--model-root", "private-input", "--output-parent", "private-input", "--secret-root", "private-input", "--stage", ".stage-test", "--target", "visible"}
	if Run(args, &out, &err) != 1 || out.Len() != 0 || strings.Contains(err.String(), "private-input") {
		t.Fatal("compiler I/O failure leaked input")
	}
	for _, values := range [][]string{{"--one", "value", "--one", "other"}, {"--one", ""}, {"--unknown", "private-input"}} {
		if _, ok := parseOptions(values, []string{"--one"}, false); ok {
			t.Fatal("duplicate/invalid option accepted")
		}
	}
}
