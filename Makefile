GO ?= go
GOFMT ?= gofmt
PYTHON ?= .venv/bin/python

.PHONY: lint test check

lint:
	@files="$$($(GOFMT) -l cmd internal)" || exit $$?; test -z "$$files" || { echo "Go formatting differs; run gofmt and retry"; exit 1; }
	$(GO) vet ./...
	$(PYTHON) -m ruff check python
	$(PYTHON) -m ruff format --check python

test:
	$(GO) test ./...
	$(PYTHON) -m pytest -q -c python/pyproject.toml python/tests

check: lint test
