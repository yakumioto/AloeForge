# Codex Cloud playbook

1. Select the already configured Codex Cloud environment for `yakumioto/AloeForge`.
2. Checkout this Starter Kit PR once it is merged (or use its branch deliberately).
3. Give Codex the contents of `codex/tasks/00-bootstrap.md`. Complete exactly one task in one PR, never directly on main.
4. Initial environment should provide pinned Go, Python 3.12, pytest and Ruff. P0 must be CPU-only without model downloads, CUDA or vLLM.
5. Require actual `make check` output in the PR, with skipped tests named explicitly.
6. Run real model, vLLM and CUDA integration on a separate explicitly configured GPU runner.
7. Never upload credentials, source-model IP or real Client Secrets to public code, CI logs, or cloud task prompts.
8. After P0 merge, continue P1–P8 in dependency order in `docs/ROADMAP.md`.
