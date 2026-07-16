#!/bin/bash
# Aggressive docker image prune during a long SWE-rebench generation run.
# The Docker Desktop VM disk is capped (~58GB here) and shared with other stacks;
# the eval's --rm containers never remove the pulled IMAGES, so a multi-GB image's
# transient extraction space can exceed the cap mid-pull -> docker run exits 125
# (the failed pull leaves no trace, so disk looks fine afterwards). Prune every 3
# min so at most ~1-2 images are ever resident. The running container's image is
# protected from prune, so this never disrupts the active instance.
while pgrep -f run_swebench_eval >/dev/null; do
  docker image prune -a -f >/dev/null 2>&1
  sleep 180
done
echo "[prune-guard $(date +%H:%M:%S)] run finished; exiting"
