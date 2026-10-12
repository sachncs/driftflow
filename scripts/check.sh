#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m ruff check driftflow tests examples
"$PYTHON" -m ruff format --check driftflow tests examples
"$PYTHON" -m mypy driftflow
"$PYTHON" -m pytest tests/ --cov=driftflow --cov-report=term-missing --cov-fail-under=90 -q
"$PYTHON" examples/demo.py --model GruM --dataset QM9 --solver Euler-Maruyama --use-approximation

if [[ -d site/node_modules ]]; then
  (cd site && npm run build)
else
  echo "site/node_modules is missing; run npm ci in site/ before site checks." >&2
  exit 1
fi
