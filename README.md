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
