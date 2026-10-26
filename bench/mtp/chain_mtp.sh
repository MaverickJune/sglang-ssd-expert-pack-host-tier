#!/usr/bin/env bash
# MTP matrix: baseline vs speculative (steps/topk/draft) in the RAM regime (33 GiB host) and the SSD regime (4 GiB host).
set -uo pipefail
MT=/workspace/bench/mtp
bash $MT/run_mtp.sh M0_base_ram33g      0 1 1 1 33792 19456 1
bash $MT/run_mtp.sh M1_spec_s1d2_ram33g 1 1 1 2 33792 19456 1
bash $MT/run_mtp.sh M2_spec_s2d3_ram33g 1 2 1 3 33792 19456 1
bash $MT/run_mtp.sh M3_spec_s3d4_ram33g 1 3 1 4 33792 19456 1
bash $MT/run_mtp.sh S0_base_ssd4g       0 1 1 1 4096  19456 1
bash $MT/run_mtp.sh S1_spec_s1d2_ssd4g  1 1 1 2 4096  19456 1
bash $MT/run_mtp.sh S3_spec_s3d4_ssd4g  1 3 1 4 4096  19456 1
echo "CHAIN_MTP DONE"
