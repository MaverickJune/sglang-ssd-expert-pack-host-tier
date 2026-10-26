#!/usr/bin/env bash
# End-to-end reproduction from a clean shell (paths as used in report.md).
# Steps already done are idempotent (download resumes, pack build is reused when valid).
set -euo pipefail
EP=/opt/ep; VENV=$EP/venv; REPO=$EP/sglang; BENCH=$(cd "$(dirname "$0")" && pwd)
GGUF=$EP/models/deepseek-v4-flash-0731/deepseek-v4-flash-0731.gguf

# 1. source + venv (upstream main, commit recorded in bench/env/sglang-commit.txt)
[ -d $REPO ] || git clone --depth 1 --branch main https://github.com/sgl-project/sglang.git $REPO
[ -x $VENV/bin/python ] || uv venv $VENV --python 3.12
export UV_CACHE_DIR=$EP/uv-cache SGLANG_BUILD_RUST_EXTS=none
$VENV/bin/python -c 'import sglang' 2>/dev/null || uv pip install --python $VENV/bin/python --prerelease=allow \
   --index-strategy unsafe-best-match --extra-index-url https://docs.sglang.ai/whl/cu130/ -e $REPO/python
# 1b. instrumentation patch (flag-gated; harmless when miss_tier_probe is not set)
(cd $REPO && git apply --check $BENCH/patches/0001-expert-pack-miss-tier-probe.patch 2>/dev/null && git apply $BENCH/patches/0001-expert-pack-miss-tier-probe.patch || true)

# 2. model (155.10 GB) + pack (147.18 GB) on the local NVMe-backed container root
[ -f $GGUF ] && [ ! -f $GGUF.chunks.json ] || $VENV/bin/python -I $BENCH/download_gguf.py --streams 16 --out $GGUF
PYTHONPATH=$REPO/python $VENV/bin/python $BENCH/prepare_pack.py --gguf $GGUF --safety-margin-gib 16 | tee $EP/logs/prepare_pack.log
ART=$(grep -m1 '^artifact_dir=' $EP/logs/prepare_pack.log | cut -d= -f2)
INPUTS=$ART/launch_inputs.json

run_config () {  # name direct_io
  local name=$1 dio=$2 stats=$ART/stats_$1.json
  rm -f $stats
  bash $BENCH/launch_server.sh $INPUTS $dio $stats 30001 '"miss_tier_probe":true' > $EP/logs/server_$name.log 2>&1 &
  local pid=$!
  until curl -s -o /dev/null http://127.0.0.1:30001/get_server_info; do sleep 5; kill -0 $pid; done
  # pass 1: cold VRAM cache; pass 2: warm VRAM cache (and, for direct_io=false, warm page cache)
  for pass in 1 2; do
    $VENV/bin/python -I $BENCH/run_requests.py --url http://127.0.0.1:30001 --stats-path $stats \
       --out-dir $BENCH/results/${name}_pass$pass --device nvme0n1 --label ${name}_pass$pass | tee $BENCH/results/${name}_pass$pass.log
  done
  kill $pid; wait $pid || true
}
mkdir -p $EP/logs $BENCH/results
run_config A_directio_true true
run_config B_directio_false false
for r in $BENCH/results/*/requests.jsonl; do echo "== $r"; $VENV/bin/python -I $BENCH/make_tables.py $r; done
