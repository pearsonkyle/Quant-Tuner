#!/usr/bin/env bash
set -uo pipefail
cd /workspace/Quant-Tuner
RUN=out/exp-059/kd32b-full-coder3
until [ -f "$RUN/train.log" ]; do sleep 60; done
sleep 120
exec bash scripts/qat_report_watch.sh "$RUN" 600
