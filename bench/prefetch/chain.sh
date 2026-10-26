#!/usr/bin/env bash
# Whole matrix. RAM-source runs use 2 passes (pass 2 = warm RAM); SSD-source runs 1 pass with a small (transit-only) host cache.
set -uo pipefail
PF=/workspace/bench/prefetch
bash $PF/run_pf.sh REC_ram33g_k0 33792 21504 0 2 1          # records route.jsonl (2 passes) and is the RAM k=0 point
for k in 0 1 2 4 8; do bash $PF/run_pf.sh SSD_host4g_k$k 4096 21504 $k 1; done
for k in 1 2 4 8; do bash $PF/run_pf.sh RAM_host33g_k$k 33792 21504 $k 2; done
echo "CHAIN DONE"
