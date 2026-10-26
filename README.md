# SGLang SSD Expert Pack on one RTX 5090 — tier audit and SSD-LLaMA host (RAM) expert tier

Everything here was produced on a single RTX 5090 (32 GB) box against upstream SGLang `main`
at commit `773cb938fe9484a8f7e7e46fb1b1fe96412c59ed`. See **[report.md](report.md)** for the full write-up.

## Layout

| path | what |
|---|---|
| `report.md` | Part 1: install, tier-model audit (no DRAM tier in stock SGLang; `direct_io=true` ⇒ every miss is SSD), config A/B benchmarks. Part 2: SSD-LLaMA host tier implementation (design A, spec→code mapping), experiments E0–E4 and the heap/transient-pool re-run E2v2–E4v2, sanity checks. |
| `patches/commits/` | The implementation as four `git format-patch` files on top of `773cb938f`: 0001 flag-gated `miss_tier_probe`, 0002 three-tier VRAM→RAM→SSD cache + paper admission policy, 0003 heap victim selection + FIFO transient pool, 0004 pool-from-reserve rule. `patches/0002-expert-pack-host-tier.patch` is the same as one diff. |
| `src/` | Convenience copies of the changed/added files at the branch tip (`expert_pack.py`, `expert_pack_tier.py`, `expert_pack_loader.py`, two test files). |
| `bench/` | Scripts (`download_gguf.py`, `prepare_pack.py`, `launch_server.sh`, `run_requests.py`, `run_experiment.sh`, `run_experiments_*.sh`, `make_*_table.py`), environment survey (`bench/env/`), and raw results (`bench/results/<run>/requests.jsonl`, server stats snapshots, server logs, all generated outputs). |

## Apply the implementation

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 773cb938fe9484a8f7e7e46fb1b1fe96412c59ed -b expert-pack-host-tier
git am ../sglang-ssd-expert-pack-host-tier/patches/commits/*.patch
python -m unittest discover -s test/registered/expert_pack -p 'test_expert_pack_tier_*.py'   # 17 tests; GPU ones skip without CUDA
```

New `--model-loader-extra-config` keys: `host_cache_mib` (0 = original behaviour), `vram_admission_policy` (`lru` | `paper`),
`admission_min_accesses` (2), `admission_window_steps` (8), `transient_slots` (default 4×top-k, paper policy only), `miss_tier_probe`.
New stats: `vram_hits`, `ram_hits`, `ssd_misses`, `vram_evictions`, `demotions_to_ram`, `ram_evictions`, `transient_uses`,
`ram_cache_bytes_used`, `tier_h2d_latency_hist_us`, `acquire_latency_hist_us` (legacy `cache_hits/misses/evictions` keep their meaning).

## Headline numbers (10 requests, DeepSeek-V4-Flash MXFP4, concurrency 1, temperature 0)

| run | VRAM hits | RAM hits | SSD misses | decode tok/s | SSD bytes |
|---|---|---|---|---|---|
| E0 original | 255,442 | 0 | 251,226 | 1.76 | 3.36 TB |
| E3 paper policy + 33 GiB pinned RAM cache | 255,613 | 176,793 | 74,262 | 3.14 | 0.99 TB |
| E4 same, VRAM cache halved | 181,726 | 231,488 | 93,454 | 3.14 | 1.25 TB |

Outputs are token-identical to the original build in every configuration; NVMe bytes equal `pack_read_bytes`;
`h2d_bytes = (ram_hits + ssd_misses) × 12.75 MiB` exactly.

## Part 3 — prefetch depth needed to hide SSD→GPU and DRAM→GPU expert fetch latency (report §15)

Oracle prefetch (`prefetch_depth=k`, routing replayed from a recorded trace) on the same model/prompts, decode only, per source of the prefetched expert:

| source of the expert | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---|---|---|---|---|
| host DRAM → GPU (H2D only, 0.27 ms/expert) | 0.00 % of decode layers stalled | 0 % | 0 % | 0 % | – |
| SSD → GPU (NVMe read 3.6 ms p50 + H2D) | 29.6 % stalled, 20.6 ms stall/token | 1.85 %, 2.7 ms | 1.16 %, 1.9 ms | 0.99 %, 1.8 ms | 0.87 %, 1.3 ms |

Per-layer decode compute on this machine is 4–5.5 ms. **DRAM-sourced experts: 1 layer ahead is enough. SSD-sourced experts: 2 layers ahead is the minimum (stall cost ≈ 1 % of TPOT), 3–4 is the safe setting; the ≈ 1 % residual is NVMe queueing in a few slow steps, not depth.** TPOT (SSD regime, 4 GiB host cache): 516 ms → 297 → 256 → 244 → 244 → 234 ms for k = 0,1,2,3,4,6. Outputs byte-identical in all 19 runs, `io_errors=0`. Scripts, raw stats and gzipped per-layer traces: `bench/prefetch/`.

## Part 4 — MTP (NextN) speculative decoding on the Expert Pack (report §16)

The HF MTP shard was repacked losslessly (HF fp4 == MXFP4 up to nibble order) into a 1-layer Expert Pack that the store attaches as logical layer 43; the NextN draft loads on the target's store and shares its GGUF embedding/lm_head (`src/.../expert_pack_loader.py`, commit 0007). 5 prompts, 200 tokens, concurrency 1:

| regime | baseline wall s | 1 draft step (2 verify tokens) | 2 steps | 3 steps | accept rate (1 step) |
|---|---|---|---|---|---|
| 33 GiB host cache | 500 | **288 (1.74×)** | 337 | 387 | 0.833 (1.84 tokens/step) |
| 4 GiB host cache (SSD-bound) | 507 | 431 (1.18×) | – | 613 (slower) | 0.833 |

Each extra draft token adds ~6 experts per layer to the verify step, so on this memory-bound machine the minimum config (`--speculative-num-steps 1 --speculative-num-draft-tokens 2`) is the optimum, and speculation only pays off with the RAM tier. Speculative outputs are not byte-identical to greedy (verify-batch numerics), but baselines reproduce byte-for-byte and the draft hit rate is 83 %. Assets builder and drivers: `bench/mtp/`.
