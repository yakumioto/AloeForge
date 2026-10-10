"""CPU test harness: scalar evidence only, no weights, prompts or Client Secret."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from tests.reference import test_dense_runtime as suite  # noqa: E402


def main():
    work = REPO / "work"
    work.mkdir(exist_ok=True)
    evidence = {
        "cases": 0,
        "max_logit_error": 0.0,
        "max_cache_error": 0.0,
        "max_mean_nll_delta": 0.0,
        "greedy_exact_cases": 0,
        "numpy": np.__version__,
        "profile": "diagnostic",
        "source_fixture_seed": 6,
        "logit_atol": suite.LOGIT_ATOL,
        "logit_rtol": suite.LOGIT_RTOL,
    }
    with tempfile.TemporaryDirectory(dir=work) as directory:
        root = Path(directory)
        cli = root / "aloeforge"
        subprocess.run(
            ["go", "build", "-o", str(cli), "./cmd/aloeforge"], cwd=REPO, check=True
        )
        for tied, kv, heads, dtype, compute in [
            (False, 1, 2, "F32", np.float64),
            (True, 2, 2, "F32", np.float32),
            (True, 1, 2, "BF16", np.float64),
            (False, 2, 2, "F16", np.float32),
            (False, 2, 4, "F32", np.float32),
            (True, 2, 4, "BF16", np.float64),
        ]:
            case = root / str(evidence["cases"])
            case.mkdir(mode=0o700)
            c, tensors, path, identity, secret, _ = suite.compile_fixture(
                cli, case, tied=tied, kv=kv, heads=heads, dtype=dtype
            )
            runtime = suite.load(path, identity, compute_dtype=compute)
            ids, p = [0, 4, 7, 1, 9], secret["permutation"]
            physical, _ = runtime.forward([p[i] for i in ids])
            source = suite.forward(c, tensors, ids)
            np.testing.assert_allclose(
                physical[:, p], source, atol=suite.LOGIT_ATOL, rtol=suite.LOGIT_RTOL
            )
            cache, chunks = None, []
            labels = np.asarray(ids[1:])

            def nll(x):
                x = x[:-1]
                maximum = np.max(x, -1)
                return float(
                    np.mean(
                        maximum
                        + np.log(np.exp(x - maximum[:, None]).sum(-1))
                        - x[np.arange(len(labels)), labels]
                    )
                )

            delta = abs(nll(physical[:, p]) - nll(source))
            assert delta <= suite.NLL_GATE
            evidence["max_mean_nll_delta"] = max(evidence["max_mean_nll_delta"], delta)
            for token in ids:
                logits, cache = runtime.forward([p[token]], cache)
                chunks.append(logits)
            np.testing.assert_allclose(
                np.concatenate(chunks), physical, atol=2e-6, rtol=2e-6
            )
            error = float(np.max(np.abs(np.concatenate(chunks) - physical)))
            evidence["max_cache_error"] = max(evidence["max_cache_error"], error)
            evidence["max_logit_error"] = max(
                evidence["max_logit_error"],
                float(np.max(np.abs(physical[:, p] - source))),
            )
            inv = np.argsort(p)
            generated = [int(inv[i]) for i in runtime.greedy([p[i] for i in ids], 5)]
            assert generated == suite.greedy(c, tensors, ids, 5)
            evidence["greedy_exact_cases"] += 1
            evidence["cases"] += 1
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
