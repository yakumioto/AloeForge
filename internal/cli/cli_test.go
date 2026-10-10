package cli

import (
	"bytes"
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	tests := []struct {
		name string
		args []string
		code int
		out  string
		err  string
	}{
		{"version", []string{"version"}, 0, "aloeforge 0.1.0-dev\n", ""},
		{"help", []string{"--help"}, 0, usage, ""},
		{"inspect help", []string{"inspect", "--help"}, 0, inspectHelp, ""},
		{"inspect short help", []string{"inspect", "-h"}, 0, inspectHelp, ""},
		{"missing command", nil, 2, "", usage},
		{"unknown command", []string{"private-prompt"}, 2, "", usage},
		{"version arguments", []string{"version", "private-prompt"}, 2, "", usage},
		{"help arguments", []string{"--help", "private-prompt"}, 2, "", usage},
		{"inspect missing root", []string{"inspect"}, 2, "", inspectHelp},
		{"inspect path", []string{"inspect", "private-prompt"}, 2, "", inspectHelp},
		{"inspect extra arguments", []string{"inspect", "--help", "private-prompt"}, 2, "", inspectHelp},
		{"verify help", []string{"verify", "--help"}, 0, verifyHelp, ""},
		{"verify missing root", []string{"verify"}, 2, "", verifyHelp},
		{"verify invalid root", []string{"verify", "--artifact-root", "private-prompt"}, 1, "", "verification failed: cannot open artifact root\n"},
		{"inspect invalid root", []string{"inspect", "--model-root", "private-prompt"}, 1, "", "inspection failed: cannot open model root\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, err bytes.Buffer
			if code := Run(tt.args, &out, &err); code != tt.code {
				t.Fatalf("exit = %d, want %d", code, tt.code)
			}
			if out.String() != tt.out || err.String() != tt.err {
				t.Fatalf("stdout = %q, stderr = %q", out.String(), err.String())
			}
			if strings.Contains(out.String()+err.String(), "private-prompt") {
				t.Fatal("user arguments leaked into output")
			}
		})
	}
}
