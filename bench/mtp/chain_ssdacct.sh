#!/usr/bin/env bash
# SSD-miss accounting runs (RAM 33 GiB regime): baseline vs MTP 1-step, with blk_ssd_ns/rd_ssd_ns in the update trace.
set -uo pipefail
MT=/workspace/bench/mtp
bash $MT/run_mtp.sh A0_base_ram33g_acct      0 1 1 1 33792 19456 1
bash $MT/run_mtp.sh A1_spec_s1d2_ram33g_acct 1 1 1 2 33792 19456 1
echo "CHAIN_ACCT DONE"
