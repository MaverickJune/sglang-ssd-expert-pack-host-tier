# SGLang SSD Expert Pack on one RTX 5090: install, tier-model audit, and 10-request benchmark

Date: 2026-10-24. Machine: single RTX 5090 (32 GB, sm_120) in a RunPod-style Docker container.

## 1. Summary

- SGLang's Expert Pack path (`--load-format expert_pack`) is **already merged into upstream `main`**; this report uses
  upstream main at commit `773cb938fe9484a8f7e7e46fb1b1fe96412c59ed` (2026-10-25 00:27 +0800). The blog's branch
  `support_deepseek-v4_and_kimi-k3_on_ssd` no longer exists on the remote; its commit `81c9f837` is an ancestor of main
  (GitHub compare: main is 3697 commits ahead, 17 behind that commit, i.e. the branch head had 17 commits never merged).
- Installed from source into a fresh uv venv (torch 2.14.1+cu130, sgl-kernel 0.4.10, flashinfer 0.7.2), JIT MXFP4
  expert kernels compiled for sm_120 on first start. One build issue fixed (Rust toolchain requirement, see §8).
- DeepSeek-V4-Flash MXFP4 GGUF (155.10 GB, Ollama blob, SHA-256 verified against the blog) and the generated Expert
  Pack (147.18 GB, 43 layers × 256 experts, top-k 6, 12.75 MiB per expert object) live on the local Samsung 990 PRO NVMe.
- **Tier model conclusion: there is no DRAM expert tier in SGLang's implementation.** The cache is two-level:
  VRAM hit, or SSD miss. Pinned host memory is 12 round-robin staging buffers (160 MB total) with no key→buffer
  mapping, so a "small miss served from host RAM" cannot occur by design. With `direct_io=false` the OS page cache
  *can* serve a miss from DRAM, but the stock counters cannot see it. I added a 79-line, flag-gated probe
  (`miss_tier_probe`) that classifies each miss by page-cache residency via `mincore(2)` immediately before the read,
  and measured both `direct_io=true` and `direct_io=false`. With `direct_io=true` (the blog default) every miss is SSD-served
  (device bytes = `pack_read_bytes`, equal latency for page-cache-resident and absent objects); with `direct_io=false` the page cache
  served 94% of misses at ~1 ms instead of ~9 ms and decode rose from 1.76 to 2.87 tok/s (§7.3). Part 1 results: §7.
- **Part 2 (follow-up task)**: I implemented the SSD-LLaMA paper's host tier in SGLang on branch `expert-pack-host-tier` (unified
  design: pinned, aligned RAM-cache slots double as the SSD read pool; VRAM → RAM → SSD lookup; transient VRAM slots; `(access_count,
  last_access)` eviction with demotion to RAM; RAM LRU with in-flight protection). With a 33 GiB host cache, 70% of VRAM misses became
  RAM hits, SSD traffic fell from 3.36 TB to 0.99 TB over the ten requests, and decode went from 1.76 to 3.14 tok/s (TPOT 574 → 321 ms),
  with outputs token-identical to the original build in every configuration (§13). A follow-up commit replaced the O(n) victim scan
  with a heap (identical victims, slot-index tie-break) and added a 24-slot FIFO transient pool allocated on top of the VRAM budget; it recovered
  ~60% of the paper-policy overhead in the no-RAM case (TPOT 664 → 602 ms) and left the RAM-tier results unchanged within noise (§13.6).


## 2. Environment (all values measured on this machine)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 |
| Driver / CUDA | driver 595.91.07 (CUDA 13.2 runtime support); nvcc 13.0.88 at `/usr/local/cuda` |
| CPU | AMD EPYC 9354 32-core (64 threads visible), 1 NUMA node |
| DRAM | 251 GiB physical on host; **container memory cgroup limit 82,999,996,416 B (77.3 GiB)**, which bounds page cache available to the server |
| OS / kernel | Ubuntu 24.04.3 LTS, Linux 6.8.0-147-generic, Docker container (cgroup v1) |
| Python | 3.12.15; uv 0.x available |
| Local SSD | Samsung SSD 990 PRO 4TB (`nvme0n1`, PCIe 4.0 class). Container root is an overlayfs on the host's `ubuntu--vg-rp--data` XFS LV on that NVMe; 512 GB quota, 512 GB free at start |
| Measured local throughput | `dd bs=1M oflag/iflag=direct`: write 4.4 GB/s, read 4.1 GB/s on `/` |
| `/workspace` | **network FUSE mount** (`mfs#eur-is-1-2.runpod.net:9421`, 337 TB free), 0.6 GB/s write / 0.9 GB/s read. Not an SSD tier; only small deliverables are kept here |
| O_DIRECT | works on `/` (overlayfs on XFS) and, surprisingly, on the FUSE mount too (bench/env/odirect_result.txt) |
| `/proc/diskstats` | `nvme0n1` visible in the container; it is the **host-wide** device counter (shared with other containers), used only as a cross-check |
| Tools absent | ollama, jq, cargo/rustc, fio, iostat, vmtouch, fincore |

Raw survey files: `bench/env/` (nvidia-smi, lscpu, meminfo, df, lsblk, mounts, cgroup limit, pip freeze, O_DIRECT test).

## 3. Revision selection

```
git ls-remote https://github.com/sgl-project/sglang.git            # main = 773cb938f...; no support_deepseek*/ssd branch
GET api.github.com/repos/sgl-project/sglang/compare/81c9f837...main # status diverged, ahead_by 3697, behind_by 17
GET api.github.com/repos/sgl-project/sglang/git/trees/main?recursive=1 | grep expert_pack
```
Upstream main contains the full feature: `python/sglang/srt/model_loader/expert_pack_loader.py`,
`.../expert_pack_runtime.py`, `.../expert_pack/{build,format,validate,prepare_deepseek_pack}.py`,
`python/sglang/srt/layers/moe/expert_pack.py` (cache store), `python/sglang/srt/layers/quantization/expert_pack.py`
(MoE method), `python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cuh`, `examples/runtime/deepseek_v4/benchmark_deepseek_5090.py`,
`test/registered/expert_pack/`. Two deltas vs. the blog text matter for reproduction:

1. `tools/expert_pack/prepare_deepseek_pack.py` moved to `python/sglang/srt/model_loader/expert_pack/prepare_deepseek_pack.py`.
2. `benchmark_deepseek_5090.py` on main has **no `--validate-only`** and runs a single fixed prompt ("Please introduce Shenzhen"), not the
   ten Alpaca/MMLU requests. Model-meta generation is now done by `expert_pack_runtime._prepare_deepseek_model_metadata` (invoked
   automatically when `--model-path` is a raw GGUF). I drove those library functions explicitly (`bench/prepare_pack.py`) and wrote my
   own 10-request client (`bench/run_requests.py`).

## 4. Install (fresh venv, from source)

```bash
mkdir -p /opt/ep && cd /opt/ep
git clone --depth 1 --branch main https://github.com/sgl-project/sglang.git sglang   # HEAD 773cb938f
uv venv /opt/ep/venv --python 3.12
# first attempt failed: setup.py requires cargo to discover Rust extensions (see §8)
export UV_CACHE_DIR=/opt/ep/uv-cache SGLANG_BUILD_RUST_EXTS=none
uv pip install --python /opt/ep/venv/bin/python --prerelease=allow --index-strategy unsafe-best-match \
   --extra-index-url https://docs.sglang.ai/whl/cu130/ -e /opt/ep/sglang/python
```
Result: `sglang 0.0.0.dev1+g773cb938f`, `torch 2.14.1+cu130` (arch list includes sm_120), `sgl_kernel 0.4.10`,
`flashinfer 0.7.2`, `transformers 5.19.0`. The expert-pack MXFP4 kernel has no arch guards and is JIT-built by nvcc for the
running GPU (`load_jit(... --use_fast_math)`); it compiled and prewarmed in 13.1 s on the 5090. `sm_120` was not a blocker.

## 5. Model, pack preparation, validation

```bash
# 155.10 GB blob pulled straight from the Ollama OCI registry (no ollama binary needed), 16 parallel ranges, ~780 MB/s
python3 bench/download_gguf.py --streams 16     # -> /opt/ep/models/deepseek-v4-flash-0731/deepseek-v4-flash-0731.gguf
#   sha256 947ac34c08c0e5c5752ac76398f934b3b6b4075cfe915ba43dd5ac754900a4cd  == blog blob digest   (OK)
#   Ollama manifest sha256 882b1398c0ca4e7ec8ca0a501fd8c4372f780f690536a3ec17ffc75306569ed3 == blog   (bench/env/ollama_manifest.json)
PYTHONPATH=/opt/ep/sglang/python /opt/ep/venv/bin/python bench/prepare_pack.py \
   --gguf /opt/ep/models/deepseek-v4-flash-0731/deepseek-v4-flash-0731.gguf --safety-margin-gib 16
```
`prepare_pack.py` (a) writes model-meta (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`metadata.json`) to `~/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d/model-meta/` (fingerprint
`87d1fbae3308fd145a1d`), (b) runs `prepare_deepseek_pack.py` which built `DeepSeek-V4-Flash.expert-pack` + manifest beside the GGUF
in ~11 min, (c) the same tool re-validated it: `EXPERT_PACK_READY ... header, index, source path and sizes are valid`. Full log:
`bench/prepare_pack.log`. Key manifest values (`bench/launch_inputs.json`):

| field | value |
|---|---|
| pack_size | 147,180,572,672 B (147.18 GB) |
| objects / index entries | 11,008 (43 layers × 256 experts) / 33,024 |
| object_stride (= expert payload, gate+up+down MXFP4) | 13,369,344 B = 12.75 MiB |
| top_k | 6 |
| pack_sha256 | 34b0b5b3867674efde0a4c4e5d7d6524764f728e3b067bc09769bbb96ceed3fa |
| model_identity_sha256 | ea2a5c416392fc8226337db0393de74618fb23c0e9210e6ed66eb24e21bb6810 |
| config_sha256 | 352c1bbf40d9e535a4e26c1e12cc79c3107f47a9253d13ed73386f038420803e |

Disk after preparation: 302 GB used of the 512 GB container root.

## 6. Tier model: what the code actually does (file:line against commit 773cb938f, unpatched line numbers)

Core file: `python/sglang/srt/layers/moe/expert_pack.py`.

- **Open mode** – `expert_pack.py:191` `open_flags = os.O_RDONLY | (os.O_DIRECT if store.direct_io else 0)`; the loader defaults
  `direct_io=True` (`expert_pack_loader.py:253`).
- **Host memory is staging only** – `expert_pack.py:388-391` allocates `stage_slots` pinned buffers of one expert payload each
  (`torch.empty(object_payload_bytes, pin_memory=True)`); 12 slots × 12.75 MiB = 160,432,128 B (`staged_bytes` in stats).
  `expert_pack.py:580` hands them out round-robin (`_stage_cursor = (_stage_cursor + 1) % len(self._staging)`), and `:583` only
  waits for the previous H2D from that slot before overwriting it. There is no dict from (layer, expert) to a host buffer, no host
  lookup before a read, and no host-side eviction counter. Any claim of a host expert cache is false for this code; the class
  docstring ("GPU and host caches") is stale.
- **Hit / miss decision is VRAM-only** – `acquire()` `expert_pack.py:507-638`: `:538` de-duplicates the routed ids of the current
  update into `requested_keys`; `:541-554` a key whose slot is in `_key_to_slot` with a matching generation is a `cache_hits`;
  everything else is `cache_misses` (`:556`), takes a victim slot (`_victim_slot`, `:464`, reuse-LFU then LRU, active working set
  protected), bumps `cache_evictions` if the slot was occupied (`:562-567`), and is appended to `pending`.
- **Every miss is a file read** – `:575-596` every pending key is read with `read_splits` (=4) concurrent `os.preadv` calls on the
  pack fd (`_read_object_range`, `:427-450`) straight into the pinned staging buffer, then `:613` `copy_(non_blocking=True)` on a
  dedicated transfer stream (`h2d_bytes += payload`, `:622`). `pack_read_bytes` (`_record_read`, `:502`) is therefore always exactly
  `cache_misses × 13,369,344` and `h2d_bytes == pack_read_bytes`; the counters cannot tell page-cache-served from device-served reads.
- **`io_errors`** – `:600-604` increments only when a read future raises `OSError`, then re-raises (the request fails). `fallback_count`
  is never incremented anywhere (telemetry placeholder, as the blog says).
- **Stats flushing** – `mark_used()` `:640-658` flushes the JSON after every `stats_flush_interval` route calls; one forward pass
  issues 43 route calls (one per MoE layer), so `stats_flush_interval=43` flushes once per token step. `route_tokens_by_layer`
  lets a client verify a request's stats are complete (prompt+completion tokens per layer).
- **Counting unit** – hits/misses count **unique (layer, expert) keys per update** after the `:538` de-dup, not per-token router
  edges. For decode (1 token) that is ≤6 keys per layer per step; for prefill it is the union of experts the prompt's tokens route to
  in that layer (≤256). This is why per-request "hits+misses" is much smaller than `tokens × 43 × 6`.

**Does `direct_io=false` make the page cache a DRAM tier?** Yes, functionally: `preadv` without `O_DIRECT` is served from the page
cache when the pages are resident, and the only cost difference is the kernel copy into the pinned buffer (the blog's §4). It is
invisible to the stock counters (`pack_read_bytes` and `h2d_bytes` are identical either way) and only weakly visible in the aggregate
`pack_read_ns`/`mean_read_ms`. It is also bounded here by the 77 GiB cgroup limit (the pack is 147 GB, so at most ~half can be resident).
Ways to observe it: (1) per-miss read latency (page-cache hits are ~an order of magnitude faster than NVMe reads of 12.75 MiB),
(2) `/proc/diskstats` device bytes vs `pack_read_bytes` (host-wide and noisy here), (3) `mincore(2)` on the pack mapping before each
read, which is exact and cheap (3,264 pages per object). I implemented (1)+(3) behind a flag.

### 6.1 Instrumentation added (patch: `bench/patches/0001-expert-pack-miss-tier-probe.patch`, 79 insertions, 2 files)

Loader config key `"miss_tier_probe": true` (plumbed at `expert_pack_loader.py`, 1 line). When set, `_initialize_runtime_state`
opens a second non-O_DIRECT fd, `mmap`s the whole pack `PROT_READ|MAP_SHARED` via ctypes (virtual only, nothing is touched), and
adds counters: `miss_pagecache_resident` (all pages of the object resident before the read), `miss_pagecache_absent` (none),
`miss_pagecache_partial`, per-class latency sums `miss_pagecache_{resident,absent}_ns`, `miss_pagecache_resident_bytes`, and a
log2 per-object read-latency histogram `read_latency_hist_us`. In `acquire()` the probe (`_probe_residency`, mincore over the object's
page range) runs right before the read futures are submitted, and `_record_miss_tier` classifies the completed read. Interpretation:
with `direct_io=false`, "resident" ⇒ served from DRAM (page cache), "absent" ⇒ served from SSD; with `direct_io=true` O_DIRECT bypasses the
page cache, so every miss is device-served and the residency split is informational only (the latency data in §7 confirms this).
Overhead: one `mincore` syscall per miss (~tens of µs vs ~10 ms per read). Unit tests: `bench/test_miss_tier_probe.py` (2 tests, pass)
and upstream `test/registered/expert_pack/test_expert_pack_runtime.py` (11 tests, still pass with the patch).

## 7. Benchmark results

Common setup: ten fixed requests from the blog (5 Alpaca, 5 MMLU) sent serially (concurrency 1) to `/generate`, prompt wrapped in
the DeepSeek chat template used by the bundled benchmark (`<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>…<｜Assistant｜><think>`),
temperature 0, `max_new_tokens=200`, default EOS handling (two MMLU/Alpaca answers stopped early at EOS: #4 at 95 tokens, #5 at 122).
`stats_flush_interval=43` = one flush per forward step. Per request the client snapshots the stats JSON before/after (waits until
`route_tokens_by_layer[0]` reflects all of the request's forwards) and `/proc/diskstats` for `nvme0n1`. Unit of hits/misses:
**unique (layer, expert) keys per `acquire()` update** (§6), not per-token router edges. "mean read ms" is the per-object read
latency (max over the 4 split reads), from `pack_read_ns / pack_reads`. Before pass 1 of config A the server had served one
1-token health probe and one 2-token smoke test (`bench/results/stats_A_after_smoketest.json`).

Final server command (config A; config B differs only in `"direct_io":false` and the stats path; `"miss_tier_probe":true` is my flag):

```bash
/opt/ep/venv/bin/python -m sglang.launch_server \
  --model-path  ~/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d/model-meta \
  --tokenizer-path ~/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d/model-meta \
  --trust-remote-code --load-format expert_pack \
  --model-loader-extra-config '{"pack_path":"/opt/ep/models/deepseek-v4-flash-0731/DeepSeek-V4-Flash.expert-pack","manifest_path":"/opt/ep/models/deepseek-v4-flash-0731/DeepSeek-V4-Flash.expert-pack.manifest.json","source_path":"/opt/ep/models/deepseek-v4-flash-0731/deepseek-v4-flash-0731.gguf","source_sha256":"947ac34c08c0e5c5752ac76398f934b3b6b4075cfe915ba43dd5ac754900a4cd","model_identity_sha256":"ea2a5c416392fc8226337db0393de74618fb23c0e9210e6ed66eb24e21bb6810","config_sha256":"352c1bbf40d9e535a4e26c1e12cc79c3107f47a9253d13ed73386f038420803e","cache_vram_mib":21504,"cache_vram_reserve_mib":2048,"stage_slots":12,"read_splits":4,"direct_io":true,"stats_flush_interval":43,"stats_path":"~/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d/stats_A_directio.json","miss_tier_probe":true}' \
  --attention-backend dsv4 --tp-size 1 --ep-size 1 --disable-cuda-graph --disable-flashinfer-autotune \
  --disable-shared-experts-fusion --skip-server-warmup --max-running-requests 1 --mem-fraction-static 0.96 \
  --watchdog-timeout 1800 --host 127.0.0.1 --port 30001
```
(`CUDA_VISIBLE_DEVICES=0`, `SGLANG_OPT_USE_TILELANG_INDEXER=1`, `CUDA_HOME=/usr/local/cuda`; see `bench/launch_server.sh`.) The
blog's `cache_vram_mib=21504` worked unchanged: `cache_capacity_experts=1614` (21,578,121,216 B), `staged_bytes=160,432,128`,
dense (non-expert) weights 8,993,054,451 B; `nvidia-smi` showed 30.9 GiB used while serving.

### 7.1 Config A: `direct_io=true` (the blog's configuration), pass 1 (cold VRAM cache)

| # | prompt id | prompt tok | compl tok | prefill tok/s | decode tok/s | hits | misses | miss: pc-resident | miss: pc-absent | miss: partial | hit rate | evictions | pack_read GB | h2d GB | diskstats read GB | io_errors | mean read ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 3.32 | 2.190 | 36564 | 17037 | 6538 | 10489 | 10 | 0.682 | 17037 | 227.77 | 227.77 | 227.78 | 0 | 3.9 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 1.70 | 1.829 | 28931 | 26698 | 11446 | 15243 | 9 | 0.520 | 26698 | 356.93 | 356.93 | 356.93 | 0 | 4.8 |
| 3 | alpaca-50812 | 28 | 200 | 3.00 | 1.868 | 30065 | 24088 | 9211 | 14870 | 7 | 0.555 | 24088 | 322.04 | 322.04 | 322.04 | 0 | 4.5 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.39 | 1.526 | 10517 | 19375 | 7862 | 11513 | 0 | 0.352 | 19375 | 259.03 | 259.03 | 259.03 | 0 | 5.9 |
| 5 | alpaca-9907 | 25 | 122 | 2.88 | 1.689 | 15839 | 18298 | 7328 | 10969 | 1 | 0.464 | 18298 | 244.63 | 244.63 | 244.63 | 0 | 4.9 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.65 | 1.685 | 25780 | 31090 | 12477 | 18612 | 1 | 0.453 | 31090 | 415.65 | 415.65 | 415.65 | 0 | 5.1 |
| 7 | alpaca-40699 | 24 | 200 | 2.96 | 1.867 | 29950 | 23954 | 9203 | 14748 | 3 | 0.556 | 23954 | 320.25 | 320.25 | 320.25 | 0 | 4.4 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 5.56 | 1.639 | 24758 | 33542 | 12607 | 20929 | 6 | 0.425 | 33542 | 448.43 | 448.43 | 448.43 | 0 | 5.3 |
| 9 | alpaca-34440 | 30 | 200 | 3.07 | 1.678 | 28347 | 25967 | 10294 | 15658 | 15 | 0.522 | 25967 | 347.16 | 347.16 | 347.61 | 0 | 5.1 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 3.85 | 1.621 | 24691 | 31177 | 13446 | 17727 | 4 | 0.442 | 31177 | 416.82 | 416.82 | 416.84 | 0 | 5.0 |

| aggregate | value |
|---|---|
| prompt / completion tokens | 792 / 1817 |
| mean prefill tok/s (all / alpaca / mmlu) | 3.54 / 3.05 / 4.03 |
| mean decode tok/s (all / alpaca / mmlu) | 1.759 / 1.858 / 1.660 |
| unique-key hits / misses / **hit rate** | 255,442 / 251,226 / **50.4%** |
| evictions | 251,226 |
| pack_read_bytes = h2d_bytes | 3.359 TB |
| /proc/diskstats nvme0n1 read bytes | 3.359 TB (+0.014%) |
| io_errors | 0 |
| misses: page-cache resident / absent / partial (probe) | 100,412 / 150,758 / 56 |
| mean per-object read latency, resident vs absent | 4.83 ms vs 5.00 ms |

### 7.2 Config A, pass 2 (same server, VRAM cache warm from pass 1)

| # | prompt id | prompt tok | compl tok | prefill tok/s | decode tok/s | hits | misses | miss: pc-resident | miss: pc-absent | miss: partial | hit rate | evictions | pack_read GB | h2d GB | diskstats read GB | io_errors | mean read ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 2.63 | 1.879 | 30199 | 23402 | 9187 | 14205 | 10 | 0.563 | 23402 | 312.87 | 312.87 | 312.87 | 0 | 4.4 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 3.70 | 1.637 | 23995 | 31634 | 13584 | 18022 | 28 | 0.431 | 31634 | 422.93 | 422.93 | 422.93 | 0 | 5.0 |
| 3 | alpaca-50812 | 28 | 200 | 3.05 | 1.965 | 31893 | 22260 | 8531 | 13719 | 10 | 0.589 | 22260 | 297.60 | 297.60 | 297.60 | 0 | 4.5 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.43 | 1.723 | 13269 | 16623 | 6652 | 9970 | 1 | 0.444 | 16623 | 222.24 | 222.24 | 222.24 | 0 | 6.1 |
| 5 | alpaca-9907 | 25 | 122 | 2.89 | 1.732 | 16449 | 17688 | 7338 | 10349 | 1 | 0.482 | 17688 | 236.48 | 236.48 | 236.48 | 0 | 5.0 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.66 | 1.782 | 28289 | 28581 | 11395 | 17185 | 1 | 0.497 | 28581 | 382.11 | 382.11 | 382.11 | 0 | 5.2 |
| 7 | alpaca-40699 | 24 | 200 | 2.97 | 1.934 | 31320 | 22584 | 8730 | 13851 | 3 | 0.581 | 22584 | 301.93 | 301.93 | 301.93 | 0 | 4.4 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 22.21 | 1.757 | 26978 | 27111 | 9700 | 17406 | 5 | 0.499 | 27111 | 362.46 | 362.46 | 362.46 | 0 | 4.6 |
| 9 | alpaca-34440 | 30 | 200 | 3.09 | 1.872 | 29899 | 24415 | 9471 | 14929 | 15 | 0.550 | 24415 | 326.41 | 326.41 | 326.41 | 0 | 4.6 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 3.85 | 1.715 | 27045 | 28823 | 12071 | 16748 | 4 | 0.484 | 28823 | 385.34 | 385.34 | 385.43 | 0 | 5.0 |

| aggregate | value |
|---|---|
| prompt / completion tokens | 792 / 1817 |
| mean prefill tok/s (all / alpaca / mmlu) | 5.35 / 2.93 / 7.77 |
| mean decode tok/s (all / alpaca / mmlu) | 1.800 / 1.876 / 1.723 |
| unique-key hits / misses / **hit rate** | 259,336 / 243,121 / **51.6%** |
| evictions | 243,121 |
| pack_read_bytes = h2d_bytes | 3.250 TB |
| /proc/diskstats nvme0n1 read bytes | 3.250 TB (+0.003%) |
| io_errors | 0 |
| misses: page-cache resident / absent / partial (probe) | 96,659 / 146,384 / 78 |
| mean per-object read latency, resident vs absent | 4.68 ms vs 4.99 ms |

Notes on config A: (i) the probe's residency split is informational here: O_DIRECT bypasses the page cache, and the identical
latencies for resident vs absent objects prove that every miss was device-served (device bytes also equal `pack_read_bytes`
to within 0.03%). The ~40% "resident" fraction is simply what the pack build left in the (cgroup-limited) page cache.
(ii) Warming the VRAM cache barely moves the hit rate (50.4% → 51.6%): the 10-request working
set is far larger than 1614 slots, so most hits are intra-request reuse (decode steps re-selecting recent experts), not cross-request.
(iii) Pass-2 request 8 (287-token prompt) got a radix prefix-cache hit: only 230 instead of 486 forwards, so its 22.2 tok/s "prefill" is
not comparable and its output differed from pass 1 (all other nine outputs were byte-identical across passes). Pass-1 numbers are
the ones to compare with the blog. (iv) `stats_complete=false` flags in pass-1 rows are an off-by-one in my client's expectation
(the last sampled token needs no forward); deltas are unaffected. Fixed before pass 2.

B_SECTIONS_PLACEHOLDER

### 7.3 Config B: `direct_io=false` (page cache allowed), pass 1 (cold VRAM; page cache ~40% warm from the pack build)

This is the only configuration in which a DRAM tier exists at all, and it is the OS page cache, not SGLang. The probe classifies
each miss by `mincore` residency before the read.

| # | prompt id | prompt tok | compl tok | prefill tok/s | decode tok/s | hits | misses | miss: pc-resident | miss: pc-absent | miss: partial | hit rate | evictions | pack_read GB | h2d GB | diskstats read GB | io_errors | mean read ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 1.45 | 3.294 | 35974 | 17627 | 13779 | 2711 | 1137 | 0.671 | 16013 | 235.66 | 235.66 | 51.61 | 0 | 2.7 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 4.71 | 2.927 | 29040 | 26589 | 24551 | 454 | 1584 | 0.522 | 26589 | 355.48 | 355.48 | 26.48 | 0 | 1.5 |
| 3 | alpaca-50812 | 28 | 200 | 4.28 | 2.969 | 30042 | 24111 | 23178 | 119 | 814 | 0.555 | 24111 | 322.35 | 322.35 | 10.73 | 0 | 1.2 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 5.44 | 2.549 | 10513 | 19379 | 18008 | 198 | 1173 | 0.352 | 19379 | 259.08 | 259.08 | 15.62 | 0 | 1.5 |
| 5 | alpaca-9907 | 25 | 122 | 4.09 | 2.779 | 15777 | 18360 | 17565 | 90 | 705 | 0.462 | 18360 | 245.46 | 245.46 | 7.79 | 0 | 1.2 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 5.68 | 2.780 | 25802 | 31068 | 29383 | 205 | 1480 | 0.454 | 31068 | 415.36 | 415.36 | 17.04 | 0 | 1.3 |
| 7 | alpaca-40699 | 24 | 200 | 4.26 | 3.000 | 29966 | 23938 | 23290 | 102 | 546 | 0.556 | 23938 | 320.04 | 320.04 | 7.07 | 0 | 1.2 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 6.39 | 2.767 | 24788 | 33512 | 31843 | 246 | 1423 | 0.425 | 33512 | 448.03 | 448.03 | 17.52 | 0 | 1.3 |
| 9 | alpaca-34440 | 30 | 200 | 4.36 | 2.938 | 28337 | 25977 | 24833 | 137 | 1007 | 0.522 | 25977 | 347.30 | 347.30 | 8.18 | 0 | 1.2 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 5.12 | 2.705 | 24699 | 31169 | 29968 | 124 | 1077 | 0.442 | 31169 | 416.71 | 416.71 | 12.09 | 0 | 1.2 |

| aggregate (config B pass 1) | value |
|---|---|
| mean prefill / decode tok/s | 4.58 / 2.871 (config A pass 1: 3.54 / 1.759) |
| VRAM hits / misses / hit rate | 254,938 / 251,730 / 50.3% (unchanged, as expected) |
| misses served from page cache (DRAM) / SSD / partial | **236,398 / 4,386 / 10,946** |
| mean per-object read latency: page-cache resident vs absent | **1.03 ms vs 9.26 ms** |
| pack_read_bytes = h2d_bytes | 3.365 TB |
| NVMe device read bytes | **0.174 TB** (5.2% of pack_read_bytes) |
| io_errors | 0 |

So with buffered I/O, 94% of the "misses" were really DRAM-served (page-cache) misses, the NVMe saw only 5% of the bytes the
counters call `pack_read_bytes`, each such miss cost ~1 ms instead of ~9 ms, and decode throughput rose 63%. None of this is visible in
the stock counters; it is exactly the "small miss" tier the task asked about, but it is accidental (kernel page cache, bounded by the
77 GiB cgroup, no pinning, extra page-cache→pinned copy) rather than designed. A second pass was started and then abandoned when the
task changed (`bench/results/B_directio_false_pass2*` is incomplete and not used).

### 7.4 Three-way split, Part 1 conclusion

| configuration | VRAM hit | DRAM-served miss | SSD-served miss |
|---|---|---|---|
| A `direct_io=true` (blog default), pass 1 | 255,442 (50.4%) | **0 by construction** (O_DIRECT; probe: equal latency for resident/absent) | 251,226 (49.6%) |
| B `direct_io=false`, pass 1 | 254,938 (50.3%) | 236,398 (46.7%, page cache; +10,946 partial) | 4,386 (0.9%) |

## 8. Problems hit and how they were fixed

| # | Problem | Fix / outcome |
|---|---|---|
| 1 | Blog's branch `support_deepseek-v4_and_kimi-k3_on_ssd` no longer exists on the remote. | Verified via GitHub compare that its commit is an ancestor of `main` and that all Expert Pack files are on `main`; used `main` @ `773cb938f` as the task asked. |
| 2 | `/workspace` is a network FUSE mount (0.9 GB/s), not the NVMe. | Put the GGUF, pack, venv and repo on the overlay container root (backed by the host's XFS LV on the Samsung 990 PRO; 4.1 GB/s O_DIRECT reads). Only small deliverables are in `/workspace`. Caveat: the container root is ephemeral. |
| 3 | `pip install -e python` failed: `setup.py` needs `cargo` to discover Rust extensions (none installed). | Rebuilt with `SGLANG_BUILD_RUST_EXTS=none` (the option the error message offers). The Rust radix-tree core is optional: `tree_core_registry._rust_fallback_reason()` falls back to the Python TreeCore when the toolchain is missing. Also moved `UV_CACHE_DIR` off the FUSE mount. |
| 4 | No `ollama` binary to pull the validated blob. | Pulled the blob directly from the Ollama OCI registry (`registry.ollama.ai/v2/frob/deepseek-v4-flash-0731/blobs/sha256:…`) with a 16-stream resumable range downloader (`bench/download_gguf.py`, ~780 MB/s); manifest and blob SHA-256 both match the blog. |
| 5 | Blog's `benchmark_deepseek_5090.py --validate-only` no longer exists; `tools/expert_pack/` moved. | `bench/prepare_pack.py` calls the same library functions (`_prepare_deepseek_model_metadata`, `prepare_deepseek_pack.py` incl. `--check-only` validation). |
| 6 | `/health` returned 503 for the first ~minute after "ready": the health probe is itself a generate request and a fully cold forward takes >20 s. | Harmless; the probe counts as 1 routed token before pass A1 (recorded). The `/freeze_gc` tracebacks in the server log come from `--skip-server-warmup` and are benign. |
| 7 | Client expected `prompt+completion` forwards per request; actual is `prompt+completion-1` (last token sampled from the final forward) and minus radix-cached prefix tokens. | Fixed in `run_requests.py` (`stats_complete` flag and `cached_tokens` recorded); deltas were never affected. |
| 8 | Server A's scheduler/detokenizer ignored SIGTERM; the first config-B launch hit CUDA OOM. | Killed by PID with SIGKILL, relaunched B (`bench/results/server_B_failed_oom.log` kept). |
| 9 | Config B's stats file does not exist until the first flush, so the client's first "before" snapshot failed. | Client treats a missing file as an all-zero baseline for request 1. |

## 9. Deliverables and reproduction

- `report.md` (this file).
- `bench/run_all.sh` – end-to-end from a clean shell (clone, venv, patch, download, pack, both configs × 2 passes, tables).
- `bench/download_gguf.py`, `bench/prepare_pack.py`, `bench/launch_server.sh`, `bench/run_requests.py`, `bench/make_tables.py`,
  `bench/test_miss_tier_probe.py`, `bench/launch_inputs.json`, `bench/prepare_pack.log`.
- `bench/results/<config>_pass<n>/requests.jsonl` (per-request rows incl. full outputs), `stats_NN_{before,after}.json`
  (raw server stats snapshots), `summary.json`; `bench/results/*.log` (client logs), `bench/results/server_*.log`,
  `bench/results/stats_*_final.json`, `bench/results/outputs_*.md` (all generations for the coherence check).
- `bench/env/` – environment survey; `bench/patches/0001-expert-pack-miss-tier-probe.patch` – the only code change (flag-gated).

Code changes: exactly one patch (79 insertions / 2 deletions in `python/sglang/srt/layers/moe/expert_pack.py` and
`python/sglang/srt/model_loader/expert_pack_loader.py`), inactive unless `"miss_tier_probe": true` is passed. No change to the
hit/miss/eviction/read/H2D logic. Nothing was deleted and no system setting was changed (page cache was never dropped).


---

# Part 2 — Follow-up: SSD-LLaMA host (RAM) expert tier added to SGLang Expert Pack

Branch `expert-pack-host-tier` off `773cb938f` (the benchmarked commit). Commits: `9cfc622` (flag-gated `miss_tier_probe`, Part 1)
and `78c8597` (three-tier cache). Patch: `bench/patches/0002-expert-pack-host-tier.patch` (branch vs `773cb938f`). Paper:
SSD-LLaMA, arXiv 2609.18110, §3.3 "Inference Workflow" (Decode; Correctness and bounded resource use), §4.1 "Asynchronous
expert delivery with continuous read submission", §4.2 "Managing Expert Placement across Three-Tier Storage".

## 10. Design decision: (A) unified RAM cache + transit pool

I chose **(A)**. In the stock code every SSD read already lands in a pinned, 4096-aligned, one-expert buffer and the H2D copies from
it (`expert_pack.py:388-391, 443, 497` in the unpatched file); making those buffers *the* RAM-cache slots means an SSD miss costs exactly
the same copies as before (SSD→pinned RAM→VRAM, zero extra memcpy), a RAM hit is a single H2D from a slot that already holds the bytes,
and a VRAM eviction demotes to RAM for free whenever the bytes are still resident. The price is that "transit buffer" and "retained in
RAM" collapse into one LRU, so an in-flight slot must be protected by state rather than by a separate pool (done with a per-slot read
refcount plus the slot's last CUDA event), and the read concurrency is bounded by `min(stage_slots, host slots)` instead of a dedicated
pool. Option (B) would have kept the paper's two pools literally but added a D2H or host memcpy of 12.75 MiB on every demotion and a second
copy of hot experts in host memory; nothing in the existing structure made (A) unreasonable, so (B) was not implemented.

Two refinements beyond the pure (A) description, both to stay closer to the paper's §4.2 wording:
- If a VRAM entry is evicted and its RAM copy has *already* been LRU-evicted, but a RAM slot is **free**, the block is copied D2H into that free
  slot (`_evict_vram_slot`, `expert_pack.py:850-872`); this never evicts a RAM entry just to demote. Otherwise the entry is discarded.
- With `host_cache_mib=0` the RAM tier is empty and the legacy round-robin staging buffers are used as the transit pool (`expert_pack.py:944-957`),
  so the `paper` admission policy can be evaluated on its own (E2).

Lines in §11 refer to commit `78c8597` (`git show 78c8597:<file>`); §13.6 cites the final tip `1582621`. Part 1 line numbers referred to the unpatched file.

## 11. Paper spec → code (file:line)

| # | Spec item | Where |
|---|---|---|
| 1 | Lookup order VRAM → RAM → SSD; only experts in neither tier become O_DIRECT reads | `expert_pack_tier.py:116-133` `TierPolicy.classify()`; called per de-duplicated key in `expert_pack.py:898`; SSD read is submitted only in the non-RAM branches `expert_pack.py:933-957` |
| 2 | Entry = whole gate+up+down block; `access_count` / `last_access` per entry, access = router-selected in a forward step, de-dup per update | one slot = `object_payload_bytes` (gate+up+down, `expert_pack.py:390`); `History` `expert_pack_tier.py:50-53`; `record_access()` `:109-113` called once per key of the de-duplicated `requested` set (`expert_pack.py:890-891`); the step counter advances once per forward pass at the first active MoE layer (`begin_step()` `:105`, `expert_pack.py:887-888`) |
| 3 | RAM hit → H2D only, no SSD read; RAM cache pinned + O_DIRECT aligned so reads land in it directly | RAM-hit branch `expert_pack.py:929-932` (empty `futures`, buffer = RAM slot); RAM slots are one `torch.empty(..., pin_memory=True)` tensor (cudaHostAlloc) `expert_pack.py:775-784`, 4096-alignment asserted `:780-783`; SSD reads `preadv` directly into a RAM slot `:933-942`; H2D `cache[slot].copy_(buffer, non_blocking=True)` on the transfer stream `:985` |
| 4 | VRAM admission: transient slot unless history justifies retention | rule `should_retain()` `expert_pack_tier.py:145-153`: retain iff a never-used VRAM slot exists, **or** `access_count ≥ admission_min_accesses` (default 2) **and** previous access ≤ `admission_window_steps` (default 8 forward steps) ago; evaluated *before* a slot is freed (`expert_pack.py:911`); `install_vram(retained=False)` counts `transient_uses` `:228-236`; a transient entry hit again that now satisfies the rule is promoted (`:122-125`) |
| 5 | VRAM eviction by lower frequency + older use; demote to RAM if space else discard | `pick_vram_victim()` `expert_pack_tier.py:159-194`: free slot → oldest transient slot → retained entry with min lexicographic `(access_count, last_access)`; protected working set never chosen (`:179, :185`); `evict_vram()` `:196-226` returns `resident` (bytes still in RAM, LRU touched) / `d2h` (free RAM slot reserved) / `discard`; store side `_evict_vram_slot` `expert_pack.py:850-872` |
| 6 | RAM eviction LRU, never an in-flight entry | `ram_alloc()` `expert_pack_tier.py:239-261`: free slot, else first LRU entry for which `evictable(slot)` holds; `_ram_slot_evictable` `expert_pack.py:826-828` = no read in flight (`_ram_reading` refcount `:935, :968-974`) and last H2D/D2H event complete (`_ram_events`) |
| 7 | Bounded VRAM / RAM / pinned pool; backpressure instead of allocation | VRAM slots fixed at init (`expert_pack.py:458-463`); RAM slots fixed (`:775-784`); pending misses are processed in batches of `stage_slots` (`:923-925`); when every RAM slot is busy `_ram_slot_for_read` waits on the oldest slot's event rather than allocating (`:830-848`) |
| 8 | No pruning / substitution / routing change; token-identical output | routed ids are only read (`:879-882`); every key is delivered before the kernel runs (`current_stream.wait_event` `:1000-1001`); `slots` is built from `_key_to_slot` for the original `route_ids` (`:1002`); the MoE kernels are untouched. Verified empirically in §13 (E1–E4 outputs vs E0) |

Stats (`snapshot()`, `expert_pack.py:1034-1068`): `vram_hits`, `ram_hits`, `ssd_misses`, `vram_evictions`, `demotions_to_ram`
(+`demotions_resident`/`demotions_d2h`/`demotions_discarded`), `ram_evictions`, `transient_uses`, `promotions`, `ram_cache_bytes_used`,
`ram_resident_experts`, `vram_retained_experts`, `vram_transient_experts`, `forward_steps`, `tier_h2d_latency_hist_us` (GPU-event-timed
H2D per expert, keyed `ram` / `ssd`; for `ssd` the host-measured read latency is added so the bucket is "SSD read + H2D"), and
`acquire_latency_hist_us` (host wall time of each `acquire()` call keyed by the worst tier it touched: `vram` = all hits). The legacy
names keep their meaning: in the tiered path `cache_hits` is incremented exactly where `vram_hits` is (`:898-901`) and `cache_misses`
where `ram_hits`/`ssd_misses` are (`:909`), so `cache_hits == vram_hits` and `cache_misses == ram_hits + ssd_misses` by construction;
the legacy path exposes the new names as aliases (`:1060-1067`). Tests: `test/registered/expert_pack/test_expert_pack_tier_policy.py`
(9 CPU-only tests on a fake backend: lookup order, counter identities incl. the two equalities above, transient admission protecting hot
experts, eviction order, demotion resident/d2h/discard, RAM LRU + in-flight protection + backpressure, `lru` policy = legacy rule) and
`test_expert_pack_tier_gpu.py` (3 tests on the synthetic pack with a real 1-slot VRAM cache: byte-exact slot contents through RAM hits,
counter identities, `h2d_bytes == (ram_hits+ssd_misses)·payload`, legacy path untouched when disabled). All 12 pass; the 11 upstream
runtime tests still pass.

## 12. Deliberately not implemented (paper features out of scope per the task)

CPU expert execution and the CPU–GPU work partition (§4.3; the paper's own RTX 5090 setting is all-GPU), rANS compression and CUDA
decompression (§4.1), io_uring continuous submission with timeout-based batching (§4.1; SGLang keeps its `preadv` thread pool with
`read_splits`×`stage_slots` concurrency), prefill expert-task grouping (§3.3; SGLang already de-duplicates experts per update), and any
cross-layer prefetch/overlap of delivery with compute (not in the paper's policy sections either).

## 13. Experiments E0–E4 (same prompts, same flags, concurrency 1, temperature 0, `direct_io=true`)

Configurations (all with `cache_vram_reserve_mib=2048, stage_slots=12, read_splits=4, stats_flush_interval=43, miss_tier_probe=true`;
E1–E4 add `admission_min_accesses=2, admission_window_steps=8`):

| run | code / config | VRAM slots | RAM slots |
|---|---|---|---|
| E0 | original code (Part 1 config A pass 1), `miss_tier_probe` on | 1614 | 0 |
| E1 | new code, `host_cache_mib=0`, `vram_admission_policy=lru` (legacy path) | 1614 | 0 |
| E2 | new code, `host_cache_mib=0`, `paper` | 1614 | 0 |
| E3 | new code, `host_cache_mib=33792` (33 GiB → 2650 RAM slots), `paper` | 1614 | 2650 (35.4 GB pinned) |
| E4 | new code, `host_cache_mib=33792`, `paper`, `cache_vram_mib=10752` (843 VRAM slots) | 843 | 2650 (35.4 GB pinned) |

`host_cache_mib=33792` is ~50% of the memory available to the server: cgroup limit 77.3 GiB minus the 10.6 GiB peak RSS measured in E1.
Each run: fresh server, one `/health` probe + one 2-token smoke request (as in E0), then the 10 requests. Raw data:
`bench/results/<run>/requests.jsonl`, `stats_NN_{before,after}.json`, `stats_final.json`, `server.log`; aggregates in
`bench/results/experiments_aggregate.json`.

### 13.1 Aggregate table

| metric (10 requests, 792 prompt + 1817 completion tokens) | E0 | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|
| VRAM slots / RAM slots | 1614 / 0 | 1614 / 0 | 1614 / 0 | 1614 / 2650 | 843 / 2650 |
| vram_hits | 255,442 | 255,442 | 255,613 | 255,613 | 181,726 |
| ram_hits ("small miss") | 0 | 0 | 0 | 176,793 | 231,488 |
| ssd_misses ("big miss") | 251,226 | 251,226 | 251,055 | 74,262 | 93,454 |
| VRAM hit rate (unique keys) | 50.4% | 50.4% | 50.4% | 50.4% | 35.9% |
| RAM hit rate among VRAM misses | 0.0% | 0.0% | 0.0% | 70.4% | 71.2% |
| served without SSD (VRAM+RAM) | 50.4% | 50.4% | 50.4% | 85.3% | 81.6% |
| vram_evictions | 251,226 | 251,226 | 251,055 | 251,055 | 324,942 |
| demotions_to_ram | 0 | 0 | 0 | 246,730 | 321,886 |
| ram_evictions | 0 | 0 | 0 | 73,572 | 92,764 |
| transient_uses | 0 | 0 | 144,002 | 144,002 | 170,299 |
| mean prefill tok/s | 3.54 | 3.72 | 3.73 | 4.05 | 3.89 |
| mean decode tok/s | 1.759 | 1.784 | 1.526 | 3.140 | 3.137 |
| mean TPOT ms | 574 | 566 | 664 | 321 | 321 |
| pack_read_bytes (TB) | 3.359 | 3.359 | 3.356 | 0.993 | 1.249 |
| NVMe read bytes, /proc/diskstats (TB) | 3.359 | 3.359 | 3.357 | 0.993 | 1.249 |
| NVMe / pack_read | 1.00014 | 1.00011 | 1.00014 | 1.00008 | 1.00000 |
| h2d_bytes (TB) | 3.359 | 3.359 | 3.356 | 3.356 | 4.344 |
| (ram_hits+ssd_misses)×12.75 MiB (TB) | 3.359 | 3.359 | 3.356 | 3.356 | 4.344 |
| io_errors | 0 | 0 | 0 | 0 | 0 |
| peak scheduler RSS (GiB) | not recorded | 10.6 | 10.6 | 67.1 | 67.0 |
| total wall time of the 10 requests (s) | 1230 | 1195 | 1367 | 744 | 750 |
| outputs token-identical to E0 | – | yes (10/10) | yes (10/10) | yes (10/10) | yes (10/10) |

### 13.2 Sanity checks (all pass)

- **E1 vs E0**: all 10 outputs byte-identical; `cache_hits`, `cache_misses`, `cache_evictions`, `pack_read_bytes`, `h2d_bytes` identical to
  the unit (255,442 / 251,226 / 251,226 / 3,358,726,815,744 / 3,358,726,815,744), as expected since the legacy path is untouched.
  Throughput differs only by noise (decode 1.784 vs 1.759 tok/s).
- **NVMe bytes ≈ pack_read_bytes**: E3 ratio 1.00008, E4 ratio 1.00000 (E0 1.00014). The host device counter includes nothing else of note.
- **h2d_bytes = (ram_hits + ssd_misses) × 13,369,344**: exact in every run (E3: 3,356,440,657,920 = (176,793+74,262)×13,369,344;
  E4: 4,344,199,827,456 = (231,488+93,454)×13,369,344). `pack_read_bytes = ssd_misses × 13,369,344` exactly as well.
- **RAM hit latency**: per RAM hit the H2D is timed with two CUDA events recorded on the transfer stream immediately before and after the
  `copy_` (`expert_pack.py:983-989` at `78c8597`; `:1015-1018` at the final tip), i.e. **GPU-side DMA duration**, not host enqueue time (which is ~16 µs) and not time-to-sync. 82-88% of
  RAM-hit copies fall in the (128, 256] µs log2 bucket and the rest in (256, 512] µs; the "≈0.23 ms mean" quoted elsewhere is a
  bucket-midpoint *estimate*, not a measured mean. An independent micro-benchmark on the idle GPU (`torch` pinned `copy_` of one 12.75 MiB
  expert, 50 repetitions, same event method) gives **median 0.268 ms = 49.9 GB/s** (min 0.254, max 0.291 ms); the link is PCIe Gen 5 x16
  (`nvidia-smi pcie.link.gen.max=5, width=16`), theoretical 64 GB/s, so ~50 GB/s is a realistic pinned-DMA figure. The same copy from
  pageable memory takes 0.52 ms, so the in-run numbers confirm the pinned path. Either way a RAM hit (~0.25-0.27 ms) is ~30× cheaper than an
  SSD miss (≈8 ms read + H2D). `acquire()` calls whose worst tier was a RAM hit take ≈1.3-1.6 ms wall (several experts per update, Python
  overhead included) versus ≈10-13 ms for updates with an SSD miss.
- **Outputs E2–E4 token-identical to E0**: 10/10 each (`bench/results/outputs_*.md`).
- `io_errors = 0` in all runs; `fallback_count = 0`.

### 13.3 Per-tier latency histograms (final snapshot of each run; log2 buckets, upper bound in µs)

- **E2** — per-expert GPU-timed H2D (RAM hit) : n/a (no RAM tier)
- **E2** — per-expert SSD read + H2D (SSD miss): n=253,067, approx. mean 7.54 ms; buckets (≤µs: count) ≤2048: 23, ≤4096: 73,006, ≤8192: 118,945, ≤16384: 44,711, ≤32768: 16,382
- **E2** — `acquire()` wall time, updates whose worst tier was vram: n=4,103, approx. mean 0.24 ms; buckets (≤µs: count) ≤128: 95, ≤256: 3,184, ≤512: 735, ≤1024: 87, ≤2048: 2
- **E2** — `acquire()` wall time, updates whose worst tier was ssd: n=74,243, approx. mean 12.85 ms; buckets (≤µs: count) ≤4096: 10,222, ≤8192: 22,827, ≤16384: 35,226, ≤32768: 5,485, ≤65536: 1, ≤131072: 71, ≤262144: 186, ≤524288: 98, ≤1048576: 85, ≤2097152: 42
- **E3** — per-expert GPU-timed H2D (RAM hit) : n=176,845, approx. mean 0.23 ms; buckets (≤µs: count) ≤256: 144,393, ≤512: 32,434, ≤1024: 18
- **E3** — per-expert SSD read + H2D (SSD miss): n=76,222, approx. mean 7.99 ms; buckets (≤µs: count) ≤2048: 338, ≤4096: 39,463, ≤8192: 14,738, ≤16384: 11,107, ≤32768: 10,576
- **E3** — `acquire()` wall time, updates whose worst tier was vram: n=4,103, approx. mean 0.27 ms; buckets (≤µs: count) ≤128: 976, ≤256: 1,901, ≤512: 773, ≤1024: 453
- **E3** — `acquire()` wall time, updates whose worst tier was ram: n=40,705, approx. mean 1.62 ms; buckets (≤µs: count) ≤512: 2,795, ≤1024: 11,871, ≤2048: 16,461, ≤4096: 9,225, ≤8192: 352, ≤65536: 1
- **E3** — `acquire()` wall time, updates whose worst tier was ssd: n=33,538, approx. mean 11.78 ms; buckets (≤µs: count) ≤4096: 11,284, ≤8192: 18,781, ≤16384: 2,962, ≤32768: 29, ≤65536: 3, ≤131072: 96, ≤262144: 168, ≤524288: 94, ≤1048576: 79, ≤2097152: 42
- **E4** — per-expert GPU-timed H2D (RAM hit) : n=231,761, approx. mean 0.22 ms; buckets (≤µs: count) ≤256: 203,027, ≤512: 28,720, ≤1024: 14
- **E4** — per-expert SSD read + H2D (SSD miss): n=95,414, approx. mean 7.80 ms; buckets (≤µs: count) ≤2048: 347, ≤4096: 48,282, ≤8192: 20,950, ≤16384: 13,779, ≤32768: 12,016, ≤65536: 40
- **E4** — `acquire()` wall time, updates whose worst tier was vram: n=869, approx. mean 0.25 ms; buckets (≤µs: count) ≤128: 147, ≤256: 527, ≤512: 121, ≤1024: 74
- **E4** — `acquire()` wall time, updates whose worst tier was ram: n=37,194, approx. mean 1.27 ms; buckets (≤µs: count) ≤256: 4, ≤512: 2,998, ≤1024: 11,710, ≤2048: 20,936, ≤4096: 1,544, ≤8192: 2
- **E4** — `acquire()` wall time, updates whose worst tier was ssd: n=40,283, approx. mean 10.29 ms; buckets (≤µs: count) ≤4096: 18,595, ≤8192: 18,261, ≤16384: 2,912, ≤32768: 32, ≤65536: 3, ≤131072: 87, ≤262144: 179, ≤524288: 93, ≤1048576: 79, ≤2097152: 42

### 13.4 Per-request tables

**E1** (`host_cache_mib=0`, `lru`)

| # | prompt id | ptok | ctok | prefill tok/s | decode tok/s | TPOT ms | vram_hits | ram_hits | ssd_misses | vram hit% | ram hit% of misses | vram_evict | demote→RAM | ram_evict | transient | pack_read GB | h2d GB | nvme GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 3.33 | 2.256 | 443 | 36564 | 0 | 17037 | 68.2 | 0.0 | 17037 | 0 | 0 | 0 | 227.8 | 227.8 | 227.8 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 3.65 | 1.839 | 544 | 28931 | 0 | 26698 | 52.0 | 0.0 | 26698 | 0 | 0 | 0 | 356.9 | 356.9 | 357.0 |
| 3 | alpaca-50812 | 28 | 200 | 2.88 | 1.856 | 539 | 30065 | 0 | 24088 | 55.5 | 0.0 | 24088 | 0 | 0 | 0 | 322.0 | 322.0 | 322.0 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.38 | 1.519 | 658 | 10517 | 0 | 19375 | 35.2 | 0.0 | 19375 | 0 | 0 | 0 | 259.0 | 259.0 | 259.1 |
| 5 | alpaca-9907 | 25 | 122 | 2.89 | 1.700 | 588 | 15839 | 0 | 18298 | 46.4 | 0.0 | 18298 | 0 | 0 | 0 | 244.6 | 244.6 | 244.6 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.64 | 1.689 | 592 | 25780 | 0 | 31090 | 45.3 | 0.0 | 31090 | 0 | 0 | 0 | 415.7 | 415.7 | 415.7 |
| 7 | alpaca-40699 | 24 | 200 | 2.95 | 1.868 | 535 | 29950 | 0 | 23954 | 55.6 | 0.0 | 23954 | 0 | 0 | 0 | 320.2 | 320.2 | 320.2 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 5.57 | 1.659 | 603 | 24758 | 0 | 33542 | 42.5 | 0.0 | 33542 | 0 | 0 | 0 | 448.4 | 448.4 | 448.4 |
| 9 | alpaca-34440 | 30 | 200 | 3.08 | 1.802 | 555 | 28347 | 0 | 25967 | 52.2 | 0.0 | 25967 | 0 | 0 | 0 | 347.2 | 347.2 | 347.3 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 3.85 | 1.649 | 606 | 24691 | 0 | 31177 | 44.2 | 0.0 | 31177 | 0 | 0 | 0 | 416.8 | 416.8 | 416.9 |

**E2** (`host_cache_mib=0`, `paper`)

| # | prompt id | ptok | ctok | prefill tok/s | decode tok/s | TPOT ms | vram_hits | ram_hits | ssd_misses | vram hit% | ram hit% of misses | vram_evict | demote→RAM | ram_evict | transient | pack_read GB | h2d GB | nvme GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 3.30 | 1.976 | 506 | 36078 | 0 | 17523 | 67.3 | 0.0 | 17523 | 0 | 0 | 12389 | 234.3 | 234.3 | 234.3 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 3.58 | 1.571 | 637 | 28909 | 0 | 26720 | 52.0 | 0.0 | 26720 | 0 | 0 | 15820 | 357.2 | 357.2 | 357.4 |
| 3 | alpaca-50812 | 28 | 200 | 3.01 | 1.652 | 605 | 29977 | 0 | 24176 | 55.4 | 0.0 | 24176 | 0 | 0 | 15584 | 323.2 | 323.2 | 323.3 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.40 | 1.283 | 780 | 10613 | 0 | 19279 | 35.5 | 0.0 | 19279 | 0 | 0 | 10804 | 257.7 | 257.7 | 257.8 |
| 5 | alpaca-9907 | 25 | 122 | 2.85 | 1.423 | 703 | 15839 | 0 | 18298 | 46.4 | 0.0 | 18298 | 0 | 0 | 9882 | 244.6 | 244.6 | 244.6 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.67 | 1.448 | 691 | 25895 | 0 | 30975 | 45.5 | 0.0 | 30975 | 0 | 0 | 17662 | 414.1 | 414.1 | 414.2 |
| 7 | alpaca-40699 | 24 | 200 | 2.95 | 1.583 | 632 | 29930 | 0 | 23974 | 55.5 | 0.0 | 23974 | 0 | 0 | 13374 | 320.5 | 320.5 | 320.6 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 5.59 | 1.399 | 715 | 24991 | 0 | 33309 | 42.9 | 0.0 | 33309 | 0 | 0 | 18287 | 445.3 | 445.3 | 445.4 |
| 9 | alpaca-34440 | 30 | 200 | 3.11 | 1.539 | 650 | 28366 | 0 | 25948 | 52.2 | 0.0 | 25948 | 0 | 0 | 14297 | 346.9 | 346.9 | 346.9 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 3.87 | 1.384 | 723 | 25015 | 0 | 30853 | 44.8 | 0.0 | 30853 | 0 | 0 | 15903 | 412.5 | 412.5 | 412.5 |

**E3** (`host_cache_mib=33792`, `paper`)

| # | prompt id | ptok | ctok | prefill tok/s | decode tok/s | TPOT ms | vram_hits | ram_hits | ssd_misses | vram hit% | ram hit% of misses | vram_evict | demote→RAM | ram_evict | transient | pack_read GB | h2d GB | nvme GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 3.76 | 3.485 | 287 | 36078 | 10895 | 6628 | 67.3 | 62.2 | 17523 | 16170 | 5938 | 12389 | 88.6 | 234.3 | 88.6 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 3.71 | 3.315 | 302 | 28909 | 19010 | 7710 | 52.0 | 71.1 | 26720 | 25993 | 7710 | 15820 | 103.1 | 357.2 | 103.1 |
| 3 | alpaca-50812 | 28 | 200 | 3.25 | 3.393 | 295 | 29977 | 17170 | 7006 | 55.4 | 71.0 | 24176 | 23771 | 7006 | 15584 | 93.7 | 323.2 | 93.7 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.57 | 2.595 | 385 | 10613 | 11647 | 7632 | 35.5 | 60.4 | 19279 | 19046 | 7632 | 10804 | 102.0 | 257.7 | 102.0 |
| 5 | alpaca-9907 | 25 | 122 | 3.40 | 2.987 | 335 | 15839 | 13669 | 4629 | 46.4 | 74.7 | 18298 | 18096 | 4629 | 9882 | 61.9 | 244.6 | 61.9 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.85 | 2.981 | 335 | 25895 | 20647 | 10328 | 45.5 | 66.7 | 30975 | 30612 | 10328 | 17662 | 138.1 | 414.1 | 138.1 |
| 7 | alpaca-40699 | 24 | 200 | 3.33 | 3.430 | 292 | 29930 | 18044 | 5930 | 55.5 | 75.3 | 23974 | 23750 | 5930 | 13374 | 79.3 | 320.5 | 79.3 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 5.75 | 2.990 | 334 | 24991 | 22640 | 10669 | 42.9 | 68.0 | 33309 | 32927 | 10669 | 18287 | 142.6 | 445.3 | 142.6 |
| 9 | alpaca-34440 | 30 | 200 | 3.66 | 3.166 | 316 | 28366 | 19436 | 6512 | 52.2 | 74.9 | 25948 | 25747 | 6512 | 14297 | 87.1 | 346.9 | 87.1 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 4.26 | 3.055 | 327 | 25015 | 23635 | 7218 | 44.8 | 76.6 | 30853 | 30618 | 7218 | 15903 | 96.5 | 412.5 | 96.5 |

**E4** (`host_cache_mib=33792`, `paper`, `cache_vram_mib=10752`)

| # | prompt id | ptok | ctok | prefill tok/s | decode tok/s | TPOT ms | vram_hits | ram_hits | ssd_misses | vram hit% | ram hit% of misses | vram_evict | demote→RAM | ram_evict | transient | pack_read GB | h2d GB | nvme GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpaca-37246 | 18 | 200 | 3.48 | 2.912 | 343 | 27830 | 17865 | 7906 | 51.9 | 69.3 | 25771 | 25185 | 7216 | 15623 | 105.7 | 344.5 | 105.7 |
| 2 | mmlu-abstract_algebra-14 | 64 | 200 | 3.76 | 3.084 | 324 | 19621 | 26731 | 9277 | 35.3 | 74.2 | 36008 | 35532 | 9277 | 18443 | 124.0 | 481.4 | 124.0 |
| 3 | alpaca-50812 | 28 | 200 | 3.04 | 3.157 | 317 | 21524 | 23414 | 9215 | 39.7 | 71.8 | 32629 | 32382 | 9215 | 18978 | 123.2 | 436.2 | 123.2 |
| 4 | mmlu-moral_disputes-8315 | 114 | 95 | 4.45 | 2.509 | 399 | 6937 | 13757 | 9198 | 23.2 | 59.9 | 22955 | 22688 | 9198 | 12204 | 123.0 | 306.9 | 123.0 |
| 5 | alpaca-9907 | 25 | 122 | 3.23 | 3.165 | 316 | 11072 | 16732 | 6333 | 32.4 | 72.5 | 23065 | 22923 | 6333 | 11587 | 84.7 | 308.4 | 84.7 |
| 6 | mmlu-high_school_macroeconomics-3940 | 132 | 200 | 4.71 | 3.019 | 331 | 17871 | 26045 | 12954 | 31.4 | 66.8 | 38999 | 38645 | 12954 | 20474 | 173.2 | 521.4 | 173.2 |
| 7 | alpaca-40699 | 24 | 200 | 3.19 | 3.432 | 291 | 21114 | 24924 | 7866 | 39.2 | 76.0 | 32790 | 32621 | 7866 | 16563 | 105.2 | 438.4 | 105.2 |
| 8 | mmlu-professional_law-10971 | 287 | 200 | 5.67 | 3.247 | 308 | 16934 | 28186 | 13180 | 29.0 | 68.1 | 41366 | 40994 | 13180 | 20972 | 176.2 | 553.0 | 176.2 |
| 9 | alpaca-34440 | 30 | 200 | 3.38 | 3.397 | 294 | 20317 | 25500 | 8497 | 37.4 | 75.0 | 33997 | 33813 | 8497 | 17225 | 113.6 | 454.5 | 113.6 |
| 10 | mmlu-jurisprudence-6660 | 70 | 200 | 4.02 | 3.442 | 291 | 18506 | 28334 | 9028 | 33.1 | 75.8 | 37362 | 37103 | 9028 | 18230 | 120.7 | 499.5 | 120.7 |

### 13.5 Observations

- **The RAM tier works as the paper intends.** With 33 GiB of pinned host cache (2650 experts, 24% of the pool) 70% of VRAM misses become
  RAM hits, SSD traffic drops from 3.36 TB to 0.99 TB over the ten requests, TPOT falls from 574 ms to 321 ms (decode 1.76 → 3.14 tok/s), and
  total wall time from 1230 s to 744 s. E4 shows the same TPOT with half the VRAM cache: 71% of the (now more numerous) misses are RAM hits,
  so the RAM tier absorbs most of the lost VRAM capacity (SSD bytes 1.25 TB vs 0.99 TB).
- **RAM demotion is almost always free in the unified design**: E3 246,730 of 251,055 evictions found their bytes still RAM-resident
  (`demotions_resident`), 4,325 were discarded, and the D2H path was never needed (`demotions_d2h = 0`), because the RAM LRU (2650 entries)
  is larger than the VRAM cache and a VRAM entry is at most 1614 loads old.
- **The paper's VRAM admission policy is neutral on this workload** (E2 vs E1: VRAM hit rate 50.45% vs 50.42%). 144k of 251k loads were
  transient; `promotions = 0` because a transient slot is the first victim of the *next layer's* misses in the same forward step, so a
  transient entry never survives to be hit in place. Experts re-requested within 8 steps are retained on their *second* load instead
  (`should_retain` sees `access_count ≥ 2` and a recent previous access). With 1614 slots for a per-token working set of 43×6 = 258 keys
  the retained set simply turns over slowly either way.
- **E2 is 0.25 tok/s slower than E1 (TPOT 664 vs 566 ms).** Two costs of the new path explain it, neither affecting correctness:
  (i) `pick_vram_victim()` for `paper` is an O(slots) Python scan per miss (`expert_pack_tier.py:159-194`): ~6 misses × 1614 slots × 43 layers
  ≈ 0.4 M loop iterations per decode token, i.e. tens of ms; (ii) transient slots are reused lowest-index-first, so the next layer's H2D is
  queued behind a `last_use` event of the kernel that just ran on that slot (`expert_pack.py:976-981`), whereas the LRU victim is long idle.
  Both were subsequently fixed and re-measured as E2v2–E4v2 (§13.6); the numbers in this table are the first, untuned runs. E3/E4 are faster
  than E1 despite this overhead.
- Peak scheduler RSS was 67.1 GiB in E3/E4 (35.4 GB pinned cache + model/GGUF reader + Python), under the 77.3 GiB cgroup limit; page
  cache for the pack is irrelevant with O_DIRECT (`miss_pagecache_*` counters in the stats files are informational, see §7.1).

### 13.6 Re-run with heap victim selection and a FIFO transient pool (E2v2–E4v2)

Commits `f2f5c45` + `1582621` (patch `bench/patches/commits/0003-*`, `0004-*`; `bench/patches/0002-expert-pack-host-tier.patch` is the full
branch diff). Two changes, implemented under the three conditions agreed beforehand:

1. **Heap victim selection, tie-break fixed to slot index.** Retained entries live in a lazy min-heap keyed on
   `(access_count, last_access, slot_index)` (`expert_pack_tier.py:170-199`, used at `:267`); stale heap items are skipped via a per-slot
   version. Because the key is exactly the triple the old full scan minimised (`score = (h.access_count, h.last_access, idx)`), the
   victim is identical to the scan, including ties. `test_heap_victim_equals_full_scan_with_slot_tiebreak` checks this against a
   brute-force scan over 600 random steps, and `check_invariants()` asserts heap ↔ retained-set consistency.
2. **Transient pool, allocated on top of `cache_vram_mib`.** `transient_slots` (default 4×top-k = 24; `expert_pack.py:186`) extra VRAM
   rows are appended to the cache tensor (`expert_pack.py:470-497`); transient installs take a free pool slot, else the oldest pool entry
   FIFO (`expert_pack_tier.py:202-245`), so layer k+1 never overwrites the slot layer k's kernel just read. If the pool is exhausted inside
   one update (large prefill) the install falls back to the retained region exactly as before (`:246-270`). A transient entry is no longer
   promoted in place (`:148-150`); it is retained on its next load if its history then qualifies.
   **VRAM accounting (condition b):** retained slots are **1614 before and 1614 after** (843/843 in E4). The pool is 24 × 13,369,344 B =
   320,864,256 B. Since the retained budget was already capped by `free − reserve` (free 23.73 GB, reserve 2.147 GB, cache 21.58 GB), the
   first attempt to allocate the pool strictly on top failed with `MemoryError` (`bench/results/E*v2_failed_pool_oom/`). I then allowed the
   pool to draw on the reserve by at most half of it (`expert_pack.py:480-495`): in E2v2/E3v2 **313,917,440 B of the pool came out of the
   2048 MiB reserve** (effective reserve 1,748 MiB; the remaining 6.9 MB from headroom), logged as a warning and recorded in the stats as
   `transient_pool_from_reserve_bytes`. In E4v2 (10,752 MiB cache) the pool fitted without touching the reserve. Total VRAM in use rose by
   306 MiB; `nvidia-smi` headroom in E0 was ~1.6 GiB, so this was safe here but it is a knob to watch on tighter configurations.

**Before / after (condition c).** v1 = first series (§13.1), v2 = re-run; E1 repeated for reference.

| metric | E1 (lru) | E2 | **E2v2** | E3 | **E3v2** | E4 | **E4v2** |
|---|---|---|---|---|---|---|---|
| retained VRAM slots / transient pool / RAM slots | 1614 / – / 0 | 1614 / 0 / 0 | 1614 / **24** / 0 | 1614 / 0 / 2650 | 1614 / **24** / 2650 | 843 / 0 / 2650 | 843 / **24** / 2650 |
| vram_hits | 255,442 | 255,613 | 255,501 | 255,613 | 255,501 | 181,726 | 181,076 |
| ram_hits | 0 | 0 | 0 | 176,793 | 177,126 | 231,488 | 232,333 |
| ssd_misses | 251,226 | 251,055 | 251,167 | 74,262 | 74,041 | 93,454 | 93,259 |
| VRAM hit rate | 50.42% | 50.45% | 50.43% | 50.45% | 50.43% | 35.87% | 35.74% |
| RAM hit rate among misses | 0.0% | 0.0% | 0.0% | 70.4% | 70.5% | 71.2% | 71.4% |
| transient_uses | 0 | 144,002 | 143,896 | 144,002 | 143,896 | 170,299 | 170,282 |
| vram_evictions | 251,226 | 251,055 | 251,167 | 251,055 | 251,167 | 324,942 | 325,592 |
| demotions_to_ram | 0 | 0 | 0 | 246,730 | 246,896 | 321,886 | 322,615 |
| ram_evictions | 0 | 0 | 0 | 73,572 | 73,351 | 92,764 | 92,569 |
| mean prefill tok/s | 3.72 | 3.73 | 3.82 | 4.05 | 3.94 | 3.89 | 3.94 |
| mean decode tok/s | 1.784 | 1.526 | 1.674 | 3.140 | 3.116 | 3.137 | 3.111 |
| mean TPOT ms | 566 | 664 | 602 | 321 | 329 | 321 | 324 |
| wall time of 10 requests (s) | 1195 | 1367 | 1257 | 744 | 755 | 750 | 752 |
| pack_read_bytes (TB) | 3.359 | 3.356 | 3.358 | 0.993 | 0.990 | 1.249 | 1.247 |
| NVMe / pack_read | 1.00011 | 1.00014 | 1.00000 | 1.00008 | 1.00000 | 1.00000 | 1.00000 |
| h2d_bytes == (ram_hits+ssd_misses)×payload | yes | yes | yes | yes | yes | yes | yes |
| io_errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| peak scheduler RSS (GiB) | 10.6 | 10.6 | 10.6 | 67.1 | 67.0 | 67.0 | 67.0 |
| outputs token-identical to E0 | yes | yes | **yes** | yes | **yes** | yes | **yes** |

| `acquire()` wall time by worst tier | E2 | E2v2 | E3 | E3v2 | E4 | E4v2 |
|---|---|---|---|---|---|---|
| vram | n=4,103, approx. mean 0.24 ms | n=4,113, approx. mean 0.28 ms | n=4,103, approx. mean 0.27 ms | n=4,113, approx. mean 0.23 ms | n=869, approx. mean 0.25 ms | n=853, approx. mean 0.24 ms |
| ram | n/a | n/a | n=40,705, approx. mean 1.62 ms | n=40,755, approx. mean 0.86 ms | n=37,194, approx. mean 1.27 ms | n=37,255, approx. mean 0.75 ms |
| ssd | n=74,243, approx. mean 12.85 ms | n=74,233, approx. mean 10.64 ms | n=33,538, approx. mean 11.78 ms | n=33,478, approx. mean 11.28 ms | n=40,283, approx. mean 10.29 ms | n=40,238, approx. mean 9.71 ms |

Per-request tables for the v2 runs: `bench/results/E2v2_paper_host0_pool24/`, `E3v2_paper_host33g_pool24/`,
`E4v2_paper_host33g_vram_half_pool24/` (`requests.jsonl`, `stats_*`, `server.log`, `outputs_*.md`).

**What changed and what did not**

- **Counters moved by <0.1%** (E2→E2v2: vram_hits 255,613 → 255,501; E3→E3v2 ram_hits 176,793 → 177,126). The victim *rule* is unchanged,
  but transient entries now sit in the pool instead of occupying retained-region slots, so a handful of experts that used to be hit in a
  transient slot before being evicted are now missed, and vice-versa. Outputs stay byte-identical (cache contents never affect the
  computation, only where the bytes come from), and every sanity check still holds: NVMe bytes = `pack_read_bytes` (ratio 1.00000),
  `h2d_bytes = (ram_hits+ssd_misses)×payload` exactly, `io_errors = 0`.
- **E2 overhead is about 60% recovered:** TPOT 664 → 602 ms (E1: 566 ms); decode 1.53 → 1.67 tok/s. The remaining ~36 ms/token over E1 is
  the rest of the Python bookkeeping of the tiered path (per-key `record_access`/`classify`, a heap push per VRAM hit, event handling).
- **E3/E4 did not get faster** (E3 321 → 329 ms, E4 321 → 324 ms TPOT; within run-to-run noise of a few %). With the RAM tier the per-token
  cost is dominated by RAM-hit H2D plus the remaining SSD misses, and the SSD read of a miss already hides the write-after-read wait the
  pool was meant to remove. The `acquire()` wall time for RAM-only updates did drop (1.62 → 0.86 ms in E3, 1.27 → 0.75 ms in E4), which is
  the heap and pool working, but it is a small share of a 320 ms token. The policy overhead is therefore a real but second-order effect once a
  RAM tier exists; it matters most for the no-RAM `paper` configuration (E2).

## 14. Deliverables (Part 2)

- `bench/patches/0002-expert-pack-host-tier.patch` – full diff of the branch vs `773cb938f` (four commits);
  `bench/patches/commits/` – the commits as `git format-patch` files (0001 probe, 0002 three-tier cache, 0003 heap + transient pool, 0004 pool-from-reserve rule).
- New/changed code: `python/sglang/srt/layers/moe/expert_pack_tier.py` (policy, 284 lines), `python/sglang/srt/layers/moe/expert_pack.py`
  (`_initialize_host_tier`, `_acquire_tiered`, `_evict_vram_slot`, `_ram_slot_for_read`, timing, `snapshot()` additions; legacy `acquire()` body
  untouched), `python/sglang/srt/model_loader/expert_pack_loader.py` (4 config keys).
- Tests: `test/registered/expert_pack/test_expert_pack_tier_policy.py` (13, CPU), `test_expert_pack_tier_gpu.py` (4, GPU); copies in `bench/`. All 28 (incl. 11 upstream) pass at the branch tip `1582621`.
- Scripts: `bench/run_experiment.sh` (one experiment: wait, warm-up, 10 requests, archive, stop), `bench/chain_e3_e4.sh`,
  `bench/run_experiments_e1_e4.sh` (all four from a clean shell), `bench/run_experiments_v2.sh` (E2v2–E4v2), `bench/make_experiment_table.py` (tables + sanity numbers).
- Raw results: `bench/results/E1_lru_host0/`, `E2_paper_host0/`, `E3_paper_host33g/`, `E4_paper_host33g_vram_half/` (+ `.log`, `outputs_*.md`),
  `bench/results/experiments_aggregate.json`.

---

# Part 3 — How many layers ahead must an expert be prefetched to hide SSD→GPU and DRAM→GPU latency? (measured)

## 15. Prefetch-depth experiment

### 15.1 Question and method

Question: on this machine (RTX 5090, PCIe Gen5 x16, Samsung 990 PRO behind overlayfs, EPYC 9354), when a routed expert is **not** in VRAM, how many layers *before* its layer must the fetch be issued so that the layer no longer waits for it — separately for experts that come from **SSD** (NVMe read → pinned RAM → H2D) and from **host DRAM** (RAM-cache hit → H2D only)?

Method (all code is on branch `expert-pack-host-tier`, commits `77d1997` and `8a3bd9e`):

1. **Oracle routing trace.** A recording run (`REC_ram33g_k0`: paper policy, 33 GiB host cache, 1614 VRAM slots, prefetch off) writes every decode/prefill routing decision `(step, layer, expert ids)` to `bench/prefetch/route.jsonl`. Decoding is deterministic at temperature 0, so the same 5 prompts reproduce the same routing; all later runs produced byte-identical outputs (checked against E1).
2. **Replay with prefetch depth k.** `prefetch_depth=k` makes the store, right after resolving layer L of step s, look up the trace entry for layer L+k (wrapping into the next step's first layers at the end of a step) and issue a background job per expert that is not already resident: *SSD-sourced* job = O_DIRECT read into a pinned RAM slot (the read executor) followed by an H2D copy on the transfer stream; *RAM-sourced* job = H2D copy only. The target VRAM slot is chosen by the normal victim policy, and the key is pinned until its layer runs. When layer L+k arrives, `_acquire_tiered` finds the key in VRAM; if its job has finished it counts `prefetch_used_ready`, otherwise it blocks on the job and counts `prefetch_used_waited` plus the blocked time (`prefetch_wait_ns`). Commit `8a3bd9e` splits all three by source (`_ram` / `_ssd`) and writes per-update `pf_wait_ram_ns` / `pf_wait_ssd_ns` into the update trace.
3. **Per-layer compute time** is measured from the update trace as the gap between the exit of layer L's acquire and the entry of layer L+1's acquire (`t_in(L+1) − t_out(L)`), i.e. the attention + MoE compute of one layer with nothing else of ours in it.
4. **Two source regimes** (the policy and prompts are identical, only `host_cache_mib` changes): *SSD regime* = `host_cache_mib=4096` (only ~320 pinned slots, enough for transit; 77 % of the prefetched experts come from SSD); *RAM regime* = `host_cache_mib=33792` (2 passes; 66 % of prefetched experts come from RAM). Because each regime is still a mix, the conclusions below use the **per-source** counters, which make the mix irrelevant.
5. Depth sweep: SSD regime k ∈ {0,1,2,3,4,6,8}, RAM regime k ∈ {0,1,2,3,4,8}. 5 prompts (first 5 of the 10-prompt set), 200 tokens, concurrency 1, temperature 0, `direct_io=true`, `transient_slots=24`, `prefetch_max_inflight=512`. 19 server launches, no failures, `io_errors=0` in all.

Deliverables: `bench/prefetch/{run_pf.sh,chain.sh,chain2.sh,analyze.py,summarize_pf.py,summary.md,route.jsonl}` and `bench/prefetch/results/<run>/{update_trace.jsonl,stats_final.json,server.log,pass*/…}`.

### 15.2 Baseline latencies at k = 0 (nothing prefetched; what a layer pays when it misses)

Decode-only acquire time per layer update (6 experts), by the worst tier touched, from the update traces of the k = 0 runs:

| run (k=0) | VRAM-only p50 | RAM-hit p50 / p90 | SSD-miss p50 / p90 | per-layer compute gap p50 | TPOT |
|---|---|---|---|---|---|
| `REC_ram33g_k0` (RAM regime) | 0.20 ms | 0.74 / 1.22 ms | 3.65 / 7.13 ms | 5.01 ms | 323–326 ms |
| `SSD_host4g_k0` (SSD regime) | 0.23 ms | 0.79 / 1.50 ms | 5.59 / 10.25 ms | 6.29 ms | 516 ms |

So, on the critical path, a RAM-served layer costs ≈ 0.5 ms over a VRAM-only layer, an SSD-served layer ≈ 3.5–5.5 ms (p50) and up to ≈ 10 ms (p90) — compared with ≈ 5 ms of compute per layer. (The acquire times above include Python bookkeeping and the event sync; the pure 12.75 MiB pinned H2D is 0.27 ms, §13.2.)

Analytic expectation before looking at the sweep: a RAM-sourced expert needs ≈ 0.3 ms of H2D, six of them serially on one stream ≈ 1.6 ms < one layer → **k = 1 should suffice**. An SSD-sourced expert needs one ≈ 3.6 ms read plus H2D; a layer whose six experts all come from SSD needs 6 × 12.75 MiB ≈ 76 MiB at ≈ 4.1 GB/s ≈ 19 ms of NVMe time → **k = 1 is not enough; k ≈ 2 covers the typical layer, k ≈ 4 covers a full six-miss burst**.

### 15.3 Result: stalls versus prefetch depth, per source (sweep 2, commit `8a3bd9e`)

"Stalled" = the layer's acquire had to block on an unfinished prefetch job for at least one of its experts. Decode updates only (prefill is excluded: a prefill layer routes hundreds of experts and cannot be hidden by a layer-granular prefetch; those stalls are in the raw tables).

| regime / k | decode layer-updates stalled on a **RAM-sourced** job | stalled on an **SSD-sourced** job | mean stall when stalled (SSD) | SSD stall time per decode token | TPOT (pass 1 / pass 2) |
|---|---|---|---|---|---|
| SSD regime k=0 | – | – (all misses synchronous) | – | 233 ms of synchronous acquire per token | 516 |
| SSD regime k=1 | 0 / 35 088 (0.00 %) | 10 368 (29.6 %) | 1.62 ms | 20.6 ms | 297 |
| SSD regime k=2 | 0 (0.00 %) | 649 (1.85 %) | 3.36 ms | 2.7 ms | 256 |
| SSD regime k=3 | 0 (0.00 %) | 406 (1.16 %) | 3.73 ms | 1.9 ms | 244 |
| SSD regime k=4 | 0 (0.00 %) | 349 (0.99 %) | 4.16 ms | 1.8 ms | 244 |
| SSD regime k=6 | 0 (0.00 %) | 305 (0.87 %) | 3.58 ms | 1.3 ms | 234 |
| RAM regime k=0 | – | – | – | 100 ms of synchronous acquire per token | 326 / 323 |
| RAM regime k=1 | 2 / 70 090 (0.003 %) | 4 801 (6.85 %) | 4.61 ms | 13.6 ms | 255 / 265 |
| RAM regime k=2 | 0 (0.00 %) | 166 (0.24 %) | 3.82 ms | 0.4 ms | 234 / 206 |
| RAM regime k=3 | 0 (0.00 %) | 106 (0.15 %) | 4.05 ms | 0.3 ms | 211 / 206 |
| RAM regime k=4 | 0 (0.00 %) | 77 (0.11 %) | 5.07 ms | 0.2 ms | 204 / 204 |

Per-key view (all updates, prefill included; from the stats counters): RAM-sourced prefetches were ready on arrival in **100.0 %** of cases at every k ≥ 1 (2 waited keys out of 139 968 at k = 1 in the RAM regime, 0 elsewhere). SSD-sourced prefetches were ready in 72 % (k=1), 93 % (k=2), 94 % (k=3), 96 % (k=4), 97.5 % (k=6) of cases in the SSD regime; the keys that still waited did so for ≈ 3.4–4.2 ms, i.e. about one full NVMe read — they are reads that were queued behind other reads, not reads that were issued too late by a fraction.

Sweep 1 (commit `77d1997`, aggregate counters only, same configs plus k = 8) agrees: SSD regime 33.9 % → 1.65 % → 0.73 % → 0.50 % of decode updates stalled for k = 1, 2, 4, 8; RAM regime 4.47 % → 0.04 % → 0.02 % → 0.05 % for k = 1, 2, 4, 8 (the RAM-regime k = 1 stalls are the SSD-sourced third of its prefetches, as sweep 2 shows). Full tables in `bench/prefetch/summary.md`.

### 15.4 Answer

* **DRAM → GPU: 1 layer ahead is enough.** With the fetch issued one layer (≈ 4–5 ms of compute) before use, a RAM-sourced expert was never late: 0 of 25 244 (SSD regime) and 2 of 139 968 (RAM regime) RAM-sourced prefetches had to be waited for at k = 1, and 0 at k ≥ 2. The transfer itself (0.27 ms per expert, ≤ 1.6 ms for six) is far below one layer of compute, so the only requirement is that the job be issued before the layer starts.
* **SSD → GPU: 2 layers ahead is the minimum, 3–4 is the safe setting, more buys little.** At k = 1 (≈ 5 ms lead ≈ one NVMe read) 30 % of decode layers still stalled; at k = 2 (≈ 10 ms lead) 1.9 % stalled and the stall cost fell from 20.6 ms to 2.7 ms per decoded token (≈ 1 % of the 256 ms TPOT); k = 3–4 brings it to ≈ 1 % of layers / 1.8–1.9 ms per token; k = 6 to 0.9 % / 1.3 ms. The residual at k ≥ 3 is not a depth problem: the stalled keys wait a whole read (≈ 3.5–4 ms) because they sit behind other reads in the NVMe queue, and the stalls cluster in a few steps with abnormally long spans (0.6–0.8 s, e.g. steps 699 and 822 of every SSD run) or in the first decode steps after a prefill, when the read queue is still draining prefill-era prefetches. Hiding that tail needs more NVMe parallelism/bandwidth or admission smoothing, not a larger k.
* **Sanity against the analytic estimate:** the measured per-layer compute gap is 4.0–5.5 ms (shrinking slightly with k as the main thread does less synchronous waiting), one NVMe read of 12.75 MiB is 3.6 ms p50 and a six-expert burst is ≈ 19 ms of NVMe time. k = 1 cannot cover even one read with margin, k = 2 covers the typical one-or-two-miss layer, k = 4 covers a six-miss burst — which is exactly the shape of the measured curve.
* **Throughput effect (same 5 prompts, byte-identical outputs in all 19 runs):** SSD regime TPOT 516 ms (k = 0) → 297 (k=1) → 256 (k=2) → 244 (k=3, 4) → 234–237 (k=6, 8). RAM regime 323 → 255–265 (k=1) → 206–234 (k=2) → 204–211 (k=3, 4). Run-to-run noise between identical passes is ≈ ±20–30 ms, so differences beyond k = 2 (SSD) / k = 2 (RAM) are within noise except for the trend. Note that with oracle prefetch the SSD regime (4 GiB host cache) reaches the same TPOT as the RAM regime at k ≥ 3 — with an oracle, the host tier's value shifts from latency hiding to reducing NVMe traffic (1.07 TB vs 0.45 TB read per pass).
* **Caveats.** (i) This is an *oracle* prefetch: the routing is known from a recorded trace, so the depth measured here is the pure transfer-latency requirement; a real predictor adds its own look-ahead error and would have to fetch more candidates per layer. (ii) Concurrency 1, 5 prompts, 200 tokens; with batching the per-layer compute gap grows and the required k shrinks. (iii) The pinned RAM footprint of in-flight prefetches is k × 6 slots (≤ 0.6 GiB at k = 8), which the 4 GiB transit pool absorbed without backpressure (`prefetch_skipped_inflight` = 0 in all SSD runs; 21 and 1 508 in the RAM-regime k = 4 / k = 8 runs from the 512 in-flight cap during prefill).


---

# Part 4 — MTP (NextN) speculative decoding on the Expert Pack

## 16. Making DeepSeek-V4-Flash's MTP head usable under `--load-format expert_pack`

### 16.1 What was missing

The Ollama MXFP4 GGUF that the Expert Pack is built from carries 43 decoder layers and **no MTP tensors** (`block_count=43`, no `blk.43.*`, `eh_proj`, `enorm`, `hnorm`, `shared_head`); the `num_nextn_predict_layers: 1` in the synthesized config is a constant in `expert_pack_runtime.py`, not a fact about the weights. SGLang itself has the draft model (`DeepseekV4ForCausalLMNextN`) and runs DeepSeek-V4 MTP through its EAGLE worker, but the expert-pack loader had no draft path, and the KV-compressor state handling needed for speculative verify is attention-side code the pack never touches (§ answer in chat: ring widening for ratio-4/128 compressors, explicit bank/commit only in the experimental online-c128 mode).

### 16.2 Weights: HF shard 46 → lossless MXFP4 repack + Q8_0 dense

* Downloaded `deepseek-ai/DeepSeek-V4-Flash` shard `model-00046-of-00046.safetensors` (3.59 GB, sha256 `f58f7228…`); it holds all 1,575 `mtp.0.*` tensors: 256 routed experts (`w1/w3/w2` as **int8-packed fp4** `[2048,2048]`/`[4096,1024]` with `F8_E8M0` scales `[rows, cols/32]`), fp8-e4m3 block-128 dense linears with e8m0 scales, bf16/f32 norms, gate, hyper-connection parameters.
* **Experts are not requantized.** HF fp4 is e2m1 with one e8m0 scale per 32 elements, exactly MXFP4's number system; the only difference is nibble order (HF packs elements 2j,2j+1 per byte; GGUF MXFP4 packs j and j+16). `bench/mtp/build_mtp_assets.py` repacks and asserts bit-exact dequantized equality for expert 0. (Side finding: Ollama's GGUF experts are *not* bit-identical to the HF fp4 — 2.8 % of block scales differ by ±1 and 13 % of elements differ by one quantization step — so the MTP layer served here is slightly *more* faithful to the released weights than the 43 target layers.)
* The repacked experts go into a mini GGUF (`blk.0.ffn_{gate,up,down}_exps.weight`, MXFP4, 3.42 GB) from which the **stock pack builder** (`expert_pack/build.py`, `--num-layers 1`) produces `DeepSeek-V4-Flash-MTP.expert-pack` (same object stride 13,369,344 B, same role bytes, own manifest/checkpoint/sha).
* Dense MTP tensors (wq_a/wq_b/wkv/wo_a/wo_b, shared expert w1/w3/w2, e_proj/h_proj) are dequantized from fp8 and quantized to **GGUF Q8_0** — the same type the target's dense GGUF weights use — max relative round-trip error 0.37–0.39 %; norms/gate/hc/sink pass through. Saved as `mtp-dense.safetensors` (181.6 MB) with HF names (`mtp.0.*`), plus a copy of the target's config/tokenizer, in `/opt/ep/models/deepseek-v4-flash-mtp/` (`draft_assets.json` has every path and digest).

### 16.3 Runtime changes (commit `5604f50`, +1 test file; 31 tests pass)

| where | change |
|---|---|
| `ExpertPackStore` | `mtp_pack_path` / `mtp_manifest_path`: a 1-layer pack is validated (1 layer, same expert count/top-k/stride/role bytes/size, identity MXFP4) and indexed as **logical layer 43**; it is read through its own fd (`_fd_by_layer`), closed with the store, excluded from the mincore probe; per-layer stat arrays grow by one. Everything else — VRAM slot cache, RAM tier, paper admission, heap victim selection, prefetch — treats `(43, e)` like any other key, so the draft's experts share the target's caches. |
| `ExpertPackMoEMethod` | the NextN decoder is built with `layer_id=0`; a prefix of `model.decoder.*` maps the FusedMoE to `store.mtp_layer_id` instead of layer 0. |
| `ExpertPackModelLoader` | `is_draft_model` → `_load_draft_model`: instantiates `DeepseekV4ForCausalLMNextN` with an `ExpertPackConfig` on the **target's registered store**, loads the Q8_0 dense tensors (`.qweight_type` before `.qweight`, as the GGUF iterator does), then **shares the target's GGUF embedding and lm_head modules** (the EAGLE worker's `get_/set_embed_and_head` assume a dense `.weight`; they are redirected to the GGUF `qweight`). |
| `decode_token_threshold` | the store classified any update with >1 token as prefill (affects victim order and prefetch); speculative verify runs `num_draft_tokens` rows per step, so updates with ≤ `decode_token_threshold` rows now count as decode (set to the draft-token count by `run_mtp.sh`). |

No change was needed in attention: the draft runs on the `dsv4` multi-step backend (`DeepseekV4MultiStepBackend`) and the target's verify path, both upstream.

Launch (`bench/mtp/launch_spec.sh`): the usual expert-pack command plus `--speculative-algorithm EAGLE --speculative-draft-model-path /opt/ep/models/deepseek-v4-flash-mtp --speculative-num-steps S --speculative-eagle-topk 1 --speculative-num-draft-tokens S+1`, and `mtp_pack_path`/`mtp_manifest_path`/`decode_token_threshold` in `--model-loader-extra-config`. `cache_vram_mib` was lowered 21504 → 19456 (1614 → 1459 slots) to leave room for the draft (181 MB dense + its 1-layer KV pool: the draft worker reported 1.95 GB available after load); the baselines below use the same 19456.

### 16.4 Results (5 prompts, 200 tokens, concurrency 1, temperature 0, paper policy, transient 24, `direct_io=true`)

| run | host cache | spec (steps / draft tokens) | wall s (5 prompts) | e2e tokens/s | TPOT ms (client) | accept rate | accept length | target steps | SSD misses | RAM hits | pack read GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 33 GiB | off | 500 | 1.63 | 402 | – | – | 822 | 34,334 | 74,865 | 459 |
| **M1** | 33 GiB | **1 / 2** | **288** | **2.80** | **284** | 0.833 | 1.84 | 453 | 38,375 | 73,906 | 513 |
| M2 | 33 GiB | 2 / 3 | 337 | 2.48 | 327 | 0.585 | 2.18 | 400 | 45,602 | 91,168 | 610 |
| M3 | 33 GiB | 3 / 4 | 387 | 2.06 | 397 | 0.443 | 2.32 | 366 | 50,814 | 105,704 | 679 |
| S0 | 4 GiB | off | 507 | 1.61 | 544 | – | – | 822 | 82,246 | 26,953 | 1,100 |
| S1 | 4 GiB | 1 / 2 | 431 | 1.87 | 452 | 0.833 | 1.84 | 453 | 93,287 | 18,994 | 1,247 |
| S3 | 4 GiB | 3 / 4 | 613 | 1.30 | 674 | 0.443 | 2.32 | 366 | 152,020 | 4,498 | 2,032 |

(`io_errors` = 0 everywhere; "TPOT ms (client)" is the client-side inter-token time, which with speculation measures bursts of 1–4 tokens; "wall s" and "e2e tokens/s" are the honest end-to-end numbers. Raw per-request data incl. `spec_accept_*` from the server's `meta_info`: `bench/mtp/results/<run>/pass1/requests.jsonl`.)

**Per target step** (RAM regime): a non-speculative step needs 144 VRAM hits + 91 RAM hits + 44 SSD misses; with 1 draft step the verify step (2 tokens) plus one draft forward needs 223 + 163 + 89, i.e. ≈ 1.6–2.0× the expert traffic, for 1.84 tokens per step. With 3 draft steps the traffic is ≈ 2.4–3.3× for 2.32 tokens. That ratio is the whole story of the table.

### 16.5 Reading the numbers

* **MTP works end-to-end on the pack.** The draft's acceptance rate is 83 % per proposed token (1.84 accepted tokens per step with one draft step), the range DeepSeek reports for V3-style MTP; the draft layer's 256 experts stream from the MTP pack through the same tiered cache (450 draft forwards per 5 prompts, visible as `route_calls_by_layer[43]`).
* **With the RAM tier, 1 draft step gives 1.74× end-to-end** (500 → 288 s, 1.63 → 2.80 tokens/s). More draft steps raise the accepted length only a little (1.84 → 2.18 → 2.32) while each extra draft token adds another ~6 experts per layer to the verify step and another draft forward, so 2 and 3 steps are slower than 1 (337 s, 387 s). On this memory-bound system the optimum is the minimum configuration `--speculative-num-steps 1 --speculative-num-draft-tokens 2`.
* **Without the RAM tier the gain shrinks to 1.18×** (507 → 431 s) and 3 steps become a 21 % *loss* (613 s): every extra verify token costs real NVMe reads (pack read 1.10 → 1.25 → 2.03 TB). Speculative decoding on an SSD-resident MoE needs the host tier to pay for its extra expert traffic.
* **Outputs are not byte-identical to the non-speculative greedy outputs** (0/5; the common prefix is 230–240 characters ≈ 60 tokens). This is expected, not a bug: with greedy verification the accepted token is the target's argmax *computed in the verify batch* (2–4 rows through the attention/MoE kernels) rather than in the 1-row decode batch, and near-ties flip under bf16/MXFP4 numerics; the two runs then continue from different tokens. The baselines (M0, S0) reproduce the earlier runs byte-for-byte (5/5), the speculative outputs are coherent and reach the same final answers on the checked prompts (same MMLU choice D, same translation), and an 83 % draft hit rate is impossible with broken draft weights. A byte-identical check across spec/non-spec would need a verify kernel path that is bit-reproducible against decode, which SGLang does not promise.
* **Caveats / not done.** (i) Oracle prefetch (Part 3) was left off in these runs; with speculation the per-step layer sequence is 0…42 (verify) then 43 (draft), so the trace replay would need the draft layer in its schedule. (ii) The target's verify batch is served by the same `mxfp4_matvec` path as prefill; a fused multi-row verify kernel might change the balance. (iii) Concurrency 1 only. (iv) The draft's own 256 experts compete for the same 1459 VRAM slots; a dedicated small resident region for layer 43 (≈ 6 experts/step, 3.3 GB for all 256) was not tried.

### 16.6 Deliverables

`bench/mtp/build_mtp_assets.py` (+ `build_mtp_assets.log`), `launch_spec.sh`, `run_mtp.sh`, `chain_mtp.sh`, `summarize_mtp.py`, `summary.md`, `results/<run>/{pass1/requests.jsonl, stats_*.json, stats_final.json, server.log, update_trace.jsonl}`, draft assets in `/opt/ep/models/deepseek-v4-flash-mtp/` (`draft_assets.json`), code in commit `5604f50` on `expert-pack-host-tier` (`bench/patches/commits/0007-*.patch`).
