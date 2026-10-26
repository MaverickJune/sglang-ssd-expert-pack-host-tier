#!/usr/bin/env bash
# E2-E4 re-run with heap victim selection + 24-slot FIFO transient pool (commit after 78c8597).
set -uo pipefail
ART=/root/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d
COMMON='"miss_tier_probe":true,"admission_min_accesses":2,"admission_window_steps":8,"transient_slots":24,"vram_admission_policy":"paper"'
run () {  # name host_cache_mib cache_vram_mib
  local N=$1 HOST=$2 VRAM=$3
  rm -f $ART/stats_$N.json
  (cd /opt/ep && CACHE_VRAM_MIB=$VRAM nohup bash /workspace/bench/launch_server.sh /workspace/bench/launch_inputs.json true $ART/stats_$N.json 30001 "$COMMON,\"host_cache_mib\":$HOST" > /opt/ep/logs/server_$N.log 2>&1 & echo $! > /opt/ep/logs/server_$N.pid)
  sleep 5; bash /workspace/bench/run_experiment.sh $N $(cat /opt/ep/logs/server_$N.pid) $ART/stats_$N.json
}
run E2v2_paper_host0_pool24 0 21504
run E3v2_paper_host33g_pool24 33792 21504
# E4v2 already completed in the first chain
# run E4v2_paper_host33g_vram_half_pool24 33792 10752
echo "CHAIN DONE"
