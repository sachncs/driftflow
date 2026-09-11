#!/usr/bin/env bash
# Local CI runner mirroring .github/workflows/ci.yml.
#
# Usage:
#   bash scripts/ci.sh
#
# Exit code is non-zero if any step fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() {
    printf '\n\033[1;34m==> %s\033[0m\n' "$1"
}

step "Install package and dev dependencies"
python -m pip install --no-cache-dir -e ".[dev]"

step "Lint with ruff"
python -m ruff check driftflow/ tests/ examples/

python -m ruff format --check driftflow/ tests/ examples/
step "Type check with mypy"
python -m mypy driftflow

step "Run test suite with 100% coverage gate"
python -m pytest tests/ --cov=driftflow --cov-fail-under=100 -q

step "Run demo smoke test"
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation > /dev/null

step "Build sdist and wheel"
python -m build

step "Verify built artifacts are importable"
python -m pip install --force-reinstall --no-cache-dir dist/driftflow-*.whl
VERSION_CHECK="$(python -c 'import driftflow; print(driftflow.__version__)')"
if [ "$VERSION_CHECK" != "1.0.0" ]; then
    printf 'Expected version 1.0.0, got %s\n' "$VERSION_CHECK"
    exit 1
fi

step "Verify sdist is self-contained"
tar -tzf dist/driftflow-*.tar.gz | grep -E "docs|examples|tests|scripts" > /dev/null
printf 'sdist contains docs, examples, tests, and scripts.\n'

printf '\n\033[1;32mAll CI steps passed.\033[0m\n'