#!/usr/bin/env bash
# Stage one SWE-rebench instance WITHOUT Docker.
#
# This is a MIMIC, not the real benchmark. The official harness runs each instance in its
# own pinned image (nebius/SWE-rebench ships `docker_image` per row); this box is an
# unprivileged container with no container runtime, so we reproduce the shape locally:
# clone at base_commit, install into a dedicated venv, apply the test patch, and grade by
# running the instance's own test_cmd. Differences from the official run — host toolchain,
# resolved dependency versions, no filesystem isolation — mean these numbers are a smoke
# test and are NOT comparable to published SWE-rebench figures.
set -euo pipefail

cd "$(dirname "$0")"
INSTANCE_JSON=${1:-instance.json}
WORK=work

read -r IID REPO BASE PY <<<"$(.venv/bin/python -c "
import json;r=json.load(open('$INSTANCE_JSON'))
print(r['instance_id'], r['repo'], r['base_commit'], r['install_config'].get('python','3.10'))
")"

REPO_DIR="$WORK/$IID/repo"
echo "instance=$IID repo=$REPO base=$BASE python=$PY"
mkdir -p "$WORK/$IID"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "== cloning $REPO =="
  git clone --quiet "https://github.com/$REPO" "$REPO_DIR"
fi
git -C "$REPO_DIR" checkout --quiet --force "$BASE"
git -C "$REPO_DIR" clean -qfd

VENV="$WORK/$IID/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv (python $PY) =="
  uv venv --python "$PY" "$VENV"
fi

echo "== installing (this is the slow part) =="
INSTALL=$(.venv/bin/python -c "import json;print(json.load(open('$INSTANCE_JSON'))['install_config'].get('install') or 'pip install -e .')")
PIPPKGS=$(.venv/bin/python -c "
import json;print(' '.join(json.load(open('$INSTANCE_JSON'))['install_config'].get('pip_packages') or []))")

# the instance's own install line, run with uv pip for speed
( cd "$REPO_DIR" && uv pip install --python "$OLDPWD/$VENV/bin/python" -e ".[complete]" 2>&1 | tail -3 ) || \
( cd "$REPO_DIR" && uv pip install --python "$OLDPWD/$VENV/bin/python" -e . 2>&1 | tail -3 )
# shellcheck disable=SC2086
uv pip install --python "$VENV/bin/python" $PIPPKGS 2>&1 | tail -2

echo "== applying test_patch =="
.venv/bin/python -c "
import json;r=json.load(open('$INSTANCE_JSON'));open('$WORK/$IID/test_patch.diff','w').write(r['test_patch'])
"
git -C "$REPO_DIR" apply -v "../test_patch.diff" 2>&1 | tail -3

echo "== env ready: $REPO_DIR =="
"$VENV/bin/python" -c "import dask, sys; print('dask', dask.__version__, '| python', sys.version.split()[0])"
