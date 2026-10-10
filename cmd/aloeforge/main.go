// Command aloeforge exposes the P0 CLI skeleton.
package main

import (
	"os"

	"github.com/yakumioto/AloeForge/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
