#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

./scripts/check.sh
"$PYTHON" -m build
"$PYTHON" -m pip_audit --skip-editable

(cd site && npm ci && npm audit --omit=dev --audit-level=high && npm run build)
