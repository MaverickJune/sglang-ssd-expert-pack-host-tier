#!/usr/bin/env bash
# One MTP experiment: name spec(0|1) steps topk draft [host_mib] [vram_mib] [passes]
set -uo pipefail
NAME=$1; SPECON=$2; STEPS=$3; TOPK=$4; DRAFT=$5; HOST=${6:-33792}; VRAM=${7:-19456}; PASSES=${8:-1}
ART=/root/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d
MT=/workspace/bench/mtp; OUT=$MT/results/$NAME; mkdir -p $OUT /opt/ep/logs
STATS=$ART/stats_$NAME.json; rm -f $STATS
APPROX=${APPROX:-off}
EXTRA="\"approx_draft_experts\":\"$APPROX\",\"approx_rows_period\":1,\"decode_token_threshold\":$DRAFT,\"miss_tier_probe\":false,\"admission_min_accesses\":2,\"admission_window_steps\":8,\"transient_slots\":24,\"vram_admission_policy\":\"paper\",\"host_cache_mib\":$HOST,\"update_trace_path\":\"$OUT/update_trace.jsonl\""
LOG=/opt/ep/logs/server_$NAME.log
(cd /opt/ep && SPEC=$SPECON SPEC_STEPS=$STEPS SPEC_TOPK=$TOPK SPEC_DRAFT=$DRAFT CACHE_VRAM_MIB=$VRAM nohup bash $MT/launch_spec.sh /workspace/bench/launch_inputs.json true $STATS 30001 "$EXTRA" > $LOG 2>&1 & echo $! > /opt/ep/logs/server_$NAME.pid)
PID=$(cat /opt/ep/logs/server_$NAME.pid); sleep 5
until grep -q 'The server is fired up' $LOG || ! kill -0 $PID 2>/dev/null || grep -q -E 'OutOfMemoryError|MemoryError|Traceback .*expert_pack' $LOG; do sleep 3; done
if ! grep -q 'The server is fired up' $LOG; then echo "SERVER FAILED $NAME"; tail -30 $LOG | cut -c1-300; cp $LOG $OUT/server.log; exit 1; fi
grep -E 'host tier|Expert pack ready|NextN draft' $LOG | cut -c1-250
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 900 http://127.0.0.1:30001/health
curl -s --max-time 900 http://127.0.0.1:30001/generate -H 'Content-Type: application/json' -d '{"text":"<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜Say hi.<｜Assistant｜><think>","sampling_params":{"temperature":0,"max_new_tokens":2}}' | cut -c1-120; echo
for p in $(seq 1 $PASSES); do
  /opt/venv/bin/python3 -I /workspace/bench/run_requests.py --url http://127.0.0.1:30001 --stats-path $STATS --out-dir $OUT/pass$p --device nvme0n1 --label ${NAME}_pass$p --only 5 2>&1 | grep -E '^\['
done
curl -s --max-time 60 http://127.0.0.1:30001/metrics > $OUT/metrics.txt 2>/dev/null || true
cp $STATS $OUT/stats_final.json; cp $LOG $OUT/server.log
CH=$(pgrep -P $PID); S=$(pgrep -f '^sglang::scheduler'); D=$(pgrep -f '^sglang::detokenizer'); kill -TERM $CH $PID $S $D 2>/dev/null; sleep 3; kill -9 $CH $PID $S $D 2>/dev/null
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1000 ] && break; sleep 2; done
echo "DONE $NAME gpu_used_mib=$u"
