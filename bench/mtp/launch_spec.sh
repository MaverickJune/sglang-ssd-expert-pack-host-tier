#!/usr/bin/env bash
# Launch the Expert Pack server with the MTP (NextN) draft and EAGLE-style speculative decoding.
#   usage: launch_spec.sh <launch_inputs.json> <direct_io> <stats_path> [port] [extra json] 
#   env: CACHE_VRAM_MIB (default 19456), SPEC_STEPS (1), SPEC_TOPK (1), SPEC_DRAFT (2), SPEC=0 to disable
set -euo pipefail
INPUTS=$1; DIRECT_IO=$2; STATS_PATH=$3; PORT=${4:-30001}; EXTRA=${5:-}
VENV=${VENV:-/opt/ep/venv}
DRAFT_DIR=${DRAFT_DIR:-/opt/ep/models/deepseek-v4-flash-mtp}
j() { python3 -c "import json,sys; print(json.load(open('$INPUTS'))['$1'])"; }
MODEL_META=$(j model_meta); PACK_PATH=$(j pack_path); MANIFEST_PATH=$(j manifest_path)
GGUF=$(j source_path); SOURCE_SHA256=$(j source_sha256)
MODEL_IDENTITY_SHA256=$(j model_identity_sha256); CONFIG_SHA256=$(j config_sha256)
CACHE_VRAM_MIB=${CACHE_VRAM_MIB:-19456}
MTP="\"mtp_pack_path\":\"$DRAFT_DIR/DeepSeek-V4-Flash-MTP.expert-pack\",\"mtp_manifest_path\":\"$DRAFT_DIR/DeepSeek-V4-Flash-MTP.expert-pack.manifest.json\""
CFG="{\"pack_path\":\"$PACK_PATH\",\"manifest_path\":\"$MANIFEST_PATH\",\"source_path\":\"$GGUF\",\"source_sha256\":\"$SOURCE_SHA256\",\"model_identity_sha256\":\"$MODEL_IDENTITY_SHA256\",\"config_sha256\":\"$CONFIG_SHA256\",\"cache_vram_mib\":$CACHE_VRAM_MIB,\"cache_vram_reserve_mib\":2048,\"stage_slots\":12,\"read_splits\":4,\"direct_io\":$DIRECT_IO,\"stats_flush_interval\":43,\"stats_path\":\"$STATS_PATH\",$MTP${EXTRA:+,$EXTRA}}"
SPEC_ARGS=()
if [ "${SPEC:-1}" = 1 ]; then
  SPEC_ARGS=(--speculative-algorithm EAGLE --speculative-draft-model-path "$DRAFT_DIR"
             --speculative-num-steps "${SPEC_STEPS:-1}" --speculative-eagle-topk "${SPEC_TOPK:-1}"
             --speculative-num-draft-tokens "${SPEC_DRAFT:-2}")
fi
echo "model-loader-extra-config: $CFG"; echo "spec args: ${SPEC_ARGS[*]:-none}"
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export SGLANG_OPT_USE_TILELANG_INDEXER=${SGLANG_OPT_USE_TILELANG_INDEXER:-1}
exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_META" --tokenizer-path "$MODEL_META" --trust-remote-code \
  --load-format expert_pack --model-loader-extra-config "$CFG" \
  --attention-backend dsv4 --tp-size 1 --ep-size 1 \
  --disable-cuda-graph --disable-flashinfer-autotune \
  --disable-shared-experts-fusion --skip-server-warmup \
  --max-running-requests 1 --mem-fraction-static 0.96 \
  --watchdog-timeout 1800 --host 127.0.0.1 --port "$PORT" "${SPEC_ARGS[@]}"
