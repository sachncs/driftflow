#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  CHECK_PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  CHECK_PYTHON="$REPO_ROOT/.venv/bin/python"
else
  CHECK_PYTHON="python3"
fi

"$CHECK_PYTHON" -m ruff check driftflow tests examples scripts
"$CHECK_PYTHON" -m ruff format --check driftflow tests examples scripts
"$CHECK_PYTHON" -m mypy driftflow
"$CHECK_PYTHON" -m pytest tests/ --cov=driftflow --cov-report=term-missing --cov-fail-under=90 -q
"$CHECK_PYTHON" examples/demo.py --model GruM --dataset QM9 --solver Euler-Maruyama --use-approximation

if [[ -d site/node_modules ]]; then
  (cd site && npm run build)
else
  echo "site/node_modules is missing; run npm ci in site/ before site checks." >&2
  exit 1
fi
