# AloeForge

AloeForge is a **Go + Python** project for researching and implementing AloePri V2-style privacy-preserving LLM compilation and inference.

**Status:** Initial project research and Codex Cloud bootstrap. No complete AloePri V2 implementation or cryptographic privacy claim.

## Design

- **Go:** compiler orchestration, model discovery, transform planning, artifact verification, secret lifecycle, client SDK and HTTP gateway.
- **Python/PyTorch:** trusted offline tensor transform worker, independent numerical reference and vLLM inference adapter.
- **Protocol:** client-side tokenization and permutation; transport obfuscated token IDs; server executes an obfuscated model; client restores output.

## Start development in Codex Cloud

1. Read [`AGENTS.md`](AGENTS.md) and [`codex/tasks/00-bootstrap.md`](codex/tasks/00-bootstrap.md).
2. Extract the archived design documents using `unzip -o codex/AloeForge_Codex_Starter_Kit.zip -d .` in your P0 development branch; do not overwrite changes already made to these files without examining differences.
3. Complete P0 and submit a scoped PR to `main`, including commands and actual test results.
4. Continue with `codex/tasks/01-artifact.md` through `08-security-release.md` after P0 is merged.

**Security:** Do not upload production keys, private source-model weights or plaintext prompts to an untrusted server or public repo. Token permutation is not encryption.

## References

- AloePri V2: https://arxiv.org/abs/2603.01499v2
- Previous Rust research code (reference only): https://github.com/yakumioto/AloePri-Model-Compiler
- Community implementation (verify method revision): https://github.com/sheng1feng/Aloepri
