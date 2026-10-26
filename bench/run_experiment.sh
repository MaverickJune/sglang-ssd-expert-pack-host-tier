#!/usr/bin/env bash
# Drive one experiment against an already-launched server: wait for readiness,
# replay the same warm-up as E0 (one /health probe + one 2-token smoke request),
# run the 10-request pass, archive stats/logs, stop the server.
#   usage: run_experiment.sh <name> <server_pid> <stats_path>
set -uo pipefail
NAME=$1; PID=$2; STATS=$3; BENCH=/workspace/bench; OUT=$BENCH/results/$NAME; LOG=/opt/ep/logs/server_$NAME.log
mkdir -p $OUT
until grep -q 'The server is fired up' $LOG || ! kill -0 $PID 2>/dev/null || grep -q -E 'OutOfMemoryError|MemoryError|Traceback .*expert_pack' $LOG; do sleep 3; done
if ! grep -q 'The server is fired up' $LOG; then echo "SERVER FAILED"; tail -40 $LOG | cut -c1-300; exit 1; fi
grep -E 'Expert pack ready|Expert pack host tier|prewarmed' $LOG | cut -c1-250
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 600 http://127.0.0.1:30001/health
curl -s --max-time 900 http://127.0.0.1:30001/generate -H 'Content-Type: application/json' \
  -d '{"text":"<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜Say hi.<｜Assistant｜><think>","sampling_params":{"temperature":0,"max_new_tokens":2}}' | cut -c1-200; echo
sleep 2; cp $STATS $OUT/stats_after_warmup.json 2>/dev/null
/opt/venv/bin/python3 -I $BENCH/run_requests.py --url http://127.0.0.1:30001 --stats-path $STATS --out-dir $OUT --device nvme0n1 --label $NAME 2>&1 | tee $BENCH/results/$NAME.log | grep -E '^\[|hit_rate"'
cp $STATS $OUT/stats_final.json; cp $LOG $OUT/server.log
CH=$(pgrep -P $PID); S=$(pgrep -f '^sglang::scheduler'); D=$(pgrep -f '^sglang::detokenizer'); kill -9 $CH $PID $S $D 2>/dev/null
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1000 ] && break; sleep 2; done
echo "DONE $NAME gpu_used_mib=$u"
