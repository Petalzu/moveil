#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=''
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo 'Python 3.11 not found. Install Python 3.11 and venv support for your distribution, or set PYTHON_BIN.' >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else "Python 3.11 is required")'
for env in .venv311 .venv-ner; do
    if [[ ! -e "$env" ]]; then
        "$PYTHON_BIN" -m venv "$env"
    fi
    [[ -x "$env/bin/python" ]] || { echo "Invalid Linux environment: $env; use a fresh Linux venv, not a copied Windows environment." >&2; exit 1; }
    "$env/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else "Existing environment must use Python 3.11")'
done
# Refuse to replace a platform-specific or incompatible existing torch installation.
.venv-ner/bin/python -c 'import importlib.metadata as m; import sys
try: version = m.version("torch")
except m.PackageNotFoundError: version = None
if version and version.split("+")[0] != "2.6.0": sys.exit("Existing torch " + version + ": use a fresh environment; setup will not replace it")'
.venv311/bin/python -m pip install -r requirements.txt
.venv311/bin/python -m pip check
.venv-ner/bin/python -m pip install -r requirements-ner.txt
.venv-ner/bin/python -m pip check
echo 'Dependencies ready. Review THIRD_PARTY_NOTICES.md, then run: .venv-ner/bin/python prepare_model.py'