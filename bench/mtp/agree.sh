#!/usr/bin/env bash
# Start an exact (non-speculative) server with the A0 config and score recorded outputs: agree.sh out.json run_dir[=label] ...
set -uo pipefail
OUT=$1; shift
ART=/root/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d
MT=/workspace/bench/mtp; STATS=$ART/stats_agree.json; rm -f $STATS
EXTRA="\"miss_tier_probe\":false,\"admission_min_accesses\":2,\"admission_window_steps\":8,\"transient_slots\":24,\"vram_admission_policy\":\"paper\",\"host_cache_mib\":33792"
LOG=/opt/ep/logs/server_agree.log
(cd /opt/ep && SPEC=0 CACHE_VRAM_MIB=19456 nohup bash $MT/launch_spec.sh /workspace/bench/launch_inputs.json true $STATS 30001 "$EXTRA" > $LOG 2>&1 & echo $! > /opt/ep/logs/server_agree.pid)
PID=$(cat /opt/ep/logs/server_agree.pid); sleep 5
until grep -q 'The server is fired up' $LOG || ! kill -0 $PID 2>/dev/null || grep -q -E 'OutOfMemoryError|MemoryError|Traceback .*expert_pack' $LOG; do sleep 3; done
if ! grep -q 'The server is fired up' $LOG; then echo "SERVER FAILED agree"; tail -30 $LOG | cut -c1-300; exit 1; fi
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 900 http://127.0.0.1:30001/health
/opt/ep/venv/bin/python $MT/agree.py http://127.0.0.1:30001 $OUT "$@" 2>&1 | tee ${OUT%.json}.log
CH=$(pgrep -P $PID); S=$(pgrep -f '^sglang::scheduler'); D=$(pgrep -f '^sglang::detokenizer'); kill -TERM $CH $PID $S $D 2>/dev/null; sleep 3; kill -9 $CH $PID $S $D 2>/dev/null
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1000 ] && break; sleep 2; done
echo "AGREE DONE gpu_used_mib=$u"
