# Codex task 00 — initialize AloeForge foundation

Repo: `yakumioto/AloeForge`. Read root `AGENTS.md` and docs first.

Goal: initialize a real, minimal, testable **Go + Python** monorepo on the existing branch. Do not write AloePri V2 math yet. Do not touch the original Rust repository.

Deliver:
1. `go.mod`, `cmd/aloeforge/main.go` with `version` and `inspect --help` CLI skeleton, `internal` package boundaries and Go tests.
2. `python/pyproject.toml`, importable `aloeforge_reference` and `aloeforge_worker` modules with tiny smoke tests; keep them separately testable.
3. `Makefile` (`make lint`, `make test`, `make check`) and GitHub Actions CPU CI running Go test/vet, Python pytest/Ruff with pinned tool versions.
4. Update README with local and Codex Cloud instructions, document cross-process IO and trustworthy/remote-server secret boundary.
5. Define initial `ModelIR`, `MethodContract`, `RuntimeContract`, `SecretBinding` data structures in docs/tests only; no premature production math. Explicitly flag unimplemented fields.
6. Add `.gitignore` protecting `.venv/`, checkpoints, source models, secrets, caches and compiled artifact directories.

Acceptance: `make check` passes on Linux without a GPU, no model downloads, no Rust source/build, no credentials or secret output. Include exact commands/results in PR. Stop at this scope.
