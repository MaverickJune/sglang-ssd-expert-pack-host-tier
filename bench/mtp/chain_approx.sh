#!/usr/bin/env bash
# Approximate expert fetching for draft rows + draft layer (RAM 33 GiB regime, MTP 1 step / 2 verify tokens).
set -uo pipefail
MT=/workspace/bench/mtp
APPROX=vram bash $MT/run_mtp.sh B1_spec_s1d2_ram33g_approx_vram 1 1 1 2 33792 19456 1
APPROX=ram  bash $MT/run_mtp.sh B2_spec_s1d2_ram33g_approx_ram  1 1 1 2 33792 19456 1
echo "CHAIN_APPROX DONE"
