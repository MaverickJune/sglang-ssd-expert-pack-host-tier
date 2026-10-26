#!/usr/bin/env bash
# Reproduce E1..E4 from a clean shell (after bench/run_all.sh steps 1-2 and the branch patch are in place).
set -uo pipefail
ART=$(ls -d $HOME/.cache/sglang-expert-pack/deepseek-v4-flash/*/ | head -1); ART=${ART%/}
COMMON='"miss_tier_probe":true,"admission_min_accesses":2,"admission_window_steps":8'
run () {  # name host_cache_mib policy cache_vram_mib
  local N=$1 HOST=$2 POL=$3 VRAM=$4
  rm -f $ART/stats_$N.json
  (cd /opt/ep && CACHE_VRAM_MIB=$VRAM nohup bash /workspace/bench/launch_server.sh /workspace/bench/launch_inputs.json true $ART/stats_$N.json 30001 "$COMMON,\"host_cache_mib\":$HOST,\"vram_admission_policy\":\"$POL\"" > /opt/ep/logs/server_$N.log 2>&1 & echo $! > /opt/ep/logs/server_$N.pid)
  sleep 5; bash /workspace/bench/run_experiment.sh $N $(cat /opt/ep/logs/server_$N.pid) $ART/stats_$N.json
}
run E1_lru_host0 0 lru 21504
run E2_paper_host0 0 paper 21504
run E3_paper_host33g 33792 paper 21504
run E4_paper_host33g_vram_half 33792 paper 10752
python3 -I /workspace/bench/make_experiment_table.py /workspace/bench/results/{A_directio_true,E1_lru_host0,E2_paper_host0,E3_paper_host33g,E4_paper_host33g_vram_half}/requests.jsonl
