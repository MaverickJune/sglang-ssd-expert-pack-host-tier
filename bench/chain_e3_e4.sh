#!/usr/bin/env bash
# Run E3 then E4 after E2's driver prints DONE.
set -uo pipefail
ART=/root/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d
E2OUT=/tmp/claude-0/-workspace/9196c7f6-43f1-4e1c-bf49-223e23814632/tasks/bq921osp2.output
until grep -q 'DONE E2_paper_host0' $E2OUT 2>/dev/null; do sleep 15; done
run () {  # name host_cache_mib cache_vram_mib
  local N=$1 HOST=$2 VRAM=$3
  rm -f $ART/stats_$N.json
  (cd /opt/ep && CACHE_VRAM_MIB=$VRAM nohup bash /workspace/bench/launch_server.sh /workspace/bench/launch_inputs.json true $ART/stats_$N.json 30001 "\"miss_tier_probe\":true,\"host_cache_mib\":$HOST,\"vram_admission_policy\":\"paper\",\"admission_min_accesses\":2,\"admission_window_steps\":8" > /opt/ep/logs/server_$N.log 2>&1 &
   echo $! > /opt/ep/logs/server_$N.pid)
  sleep 5
  bash /workspace/bench/run_experiment.sh $N $(cat /opt/ep/logs/server_$N.pid) $ART/stats_$N.json
}
run E3_paper_host33g 33792 21504
run E4_paper_host33g_vram_half 33792 10752
echo "CHAIN DONE"
