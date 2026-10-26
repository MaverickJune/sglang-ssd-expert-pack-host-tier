#!/usr/bin/env bash
# Sweep 2: same configs as chain.sh but with per-source (RAM vs SSD) prefetch readiness counters
# (code revision with prefetch_used_{ready,waited}_{ram,ssd}) and k=3 / k=6 added.
set -uo pipefail
PF=/workspace/bench/prefetch
for k in 1 2 3 4 6; do bash $PF/run_pf.sh SSDv2_host4g_k$k 4096 21504 $k 1; done
for k in 1 2 3 4; do bash $PF/run_pf.sh RAMv2_host33g_k$k 33792 21504 $k 2; done
echo "CHAIN2 DONE"
