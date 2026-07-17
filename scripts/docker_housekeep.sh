#!/bin/bash
# SWE-rebench Docker housekeeping — removes ONLY SWE-rebench eval images (never a
# global prune, never anything else running on the daemon). The agentic eval pulls a
# multi-GB image per instance and its --rm containers remove themselves but leave the
# IMAGES behind; on a capped Docker Desktop VM disk they pile up until a pull's transient
# extraction space overflows the cap and `docker run` fails with exit 125.
#
# Two modes:
#   scripts/docker_housekeep.sh            # one-shot: remove all idle SWE-rebench images now
#   scripts/docker_housekeep.sh --guard    # loop every 3 min while a generation run is live
#
# Prefer `run_swebench_eval.py --cleanup-images` for new runs (removes each instance's
# image inline). This is the safety net for runs already in flight + ad-hoc cleanup.
#
# "SWE-rebench image" = repository matching swebench/sweb.eval or swerebench/sweb.eval.
# Images backing a running container are skipped by docker automatically.
set -u
SWE_RE='swe(re)?bench/sweb\.eval'

sweep() {
  local imgs n=0
  # 1) SWE-tagged images (swerebench/sweb.eval.*) with no container.
  imgs=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null \
         | grep -iE "$SWE_RE" | awk '{print $2}' | sort -u)
  for id in $imgs; do
    docker image rm -f "$id" >/dev/null 2>&1 && n=$((n+1))  # no-op if a container holds it
  done
  # 2) DANGLING (<none>, untagged) images -- run_swebench_eval's --cleanup-images UNTAGS
  #    each SWE image, leaving its layers as <none>; `prune -f` removes ONLY dangling
  #    images (never a tagged image, so deckdoctor etc. stay; never a global -a prune).
  local d
  d=$(docker image prune -f 2>/dev/null | grep -Eo 'Total reclaimed space:.*' || true)
  echo "[housekeep $(date +%H:%M:%S)] removed $n tagged SWE image(s); dangling: ${d:-none}; $(docker system df --format '{{.Type}} {{.Size}}' 2>/dev/null | grep -i '^Images')"
}

if [ "${1:-}" = "--guard" ]; then
  while pgrep -f run_swebench_eval >/dev/null; do
    sweep
    sleep 180
  done
  sweep  # final pass after the run exits
  echo "[housekeep $(date +%H:%M:%S)] generation run finished; exiting guard"
else
  sweep
fi
