#!/usr/bin/env bash
# Prepare work/<iid>/{instance.json,repo,venv,test_patch.diff} for each harvest instance.
set -uo pipefail
cd /workspace/swe-mimic
SRC=/workspace/Quant-Tuner/out/external/swe-rebench/harvest_instances.jsonl
python3 - <<'PY'
import json, pathlib
for line in open("/workspace/Quant-Tuner/out/external/swe-rebench/harvest_instances.jsonl"):
    r = json.loads(line)
    d = pathlib.Path("work") / r["instance_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "instance.json").write_text(json.dumps(r))
    (d / "test_patch.diff").write_text(r["test_patch"])
    print(r["instance_id"], r["repo"], r["base_commit"][:10])
PY
while read -r iid repo commit; do
    w=work/$iid
    if [ ! -d "$w/repo/.git" ]; then
        echo "[harvest] cloning $repo"
        git clone -q "https://github.com/$repo" "$w/repo" || { echo "[harvest] CLONE FAIL $iid"; continue; }
    fi
    git -C "$w/repo" checkout -qf "$commit" || echo "[harvest] CHECKOUT FAIL $iid"
    if [ ! -d "$w/venv" ]; then
        python3 -m venv "$w/venv"
        timeout 600 "$w/venv/bin/pip" install -q -e "$w/repo" pytest 2>&1 | tail -1 || true
    fi
    echo "[harvest] ready $iid"
done < <(python3 -c "
import json
for line in open('$SRC'):
    r = json.loads(line)
    print(r['instance_id'], r['repo'], r['base_commit'])")
echo "[harvest] setup done"
