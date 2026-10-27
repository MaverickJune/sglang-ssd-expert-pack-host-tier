#!/usr/bin/env bash
# After chain_approx: lossless variant (draft layer 43 only; all verify rows exact), then teacher-forced agreement scoring.
set -uo pipefail
MT=/workspace/bench/mtp
until grep -q "CHAIN_APPROX DONE" /opt/ep/logs/chain_approx.log; do sleep 10; done
# approx_rows_period=1 -> no verify row is approximated; only the MTP draft layer is
sed 's/"approx_draft_experts\\":\\"$APPROX\\",/"approx_draft_experts\\":\\"$APPROX\\",\\"approx_rows_period\\":1,/' $MT/run_mtp.sh > $MT/run_mtp_draftonly.sh
APPROX=vram bash $MT/run_mtp_draftonly.sh B3_spec_s1d2_ram33g_approx_draftlayer 1 1 1 2 33792 19456 1
R=$MT/results
bash $MT/agree.sh $R/agreement.json $R/A0_base_ram33g_acct=A0_exact_greedy $R/A1_spec_s1d2_ram33g_acct=A1_exact_spec $R/B1_spec_s1d2_ram33g_approx_vram=B1_approx_vram $R/B2_spec_s1d2_ram33g_approx_ram=B2_approx_ram $R/B3_spec_s1d2_ram33g_approx_draftlayer=B3_approx_draftlayer
echo "CHAIN_APPROX2 DONE"
