| run | pass | gen tokens | wall s | tokens/s (e2e) | decode tok/s (client) | TPOT ms | accept rate | accept len | VRAM hits | RAM hits | SSD misses | pack read GB | io_err | draft calls (layer 43) | identical/5 | mean common prefix chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0_base_ram33g | pass1 | 817 | 500 | 1.63 | 2.49 | 402 | - | - | 118213 | 74865 | 34334 | 459 | 0 | 0 | 5 | 614 |
| M1_spec_s1d2_ram33g | pass1 | 806 | 288 | 2.80 | 3.52 | 284 | 0.833 | 1.84 | 100669 | 73906 | 38375 | 513 | 0 | 450 | 0 | 241 |
| M2_spec_s2d3_ram33g | pass1 | 837 | 337 | 2.48 | 3.06 | 327 | 0.585 | 2.18 | 110855 | 91168 | 45602 | 610 | 0 | 789 | 0 | 233 |
| M3_spec_s3d4_ram33g | pass1 | 798 | 387 | 2.06 | 2.52 | 397 | 0.443 | 2.32 | 124648 | 105704 | 50814 | 679 | 0 | 1079 | 0 | 240 |
| S0_base_ssd4g | pass1 | 817 | 507 | 1.61 | 1.84 | 544 | - | - | 118213 | 26953 | 82246 | 1100 | 0 | 0 | 5 | 614 |
| S1_spec_s1d2_ssd4g | pass1 | 806 | 431 | 1.87 | 2.21 | 452 | 0.833 | 1.84 | 100669 | 18994 | 93287 | 1247 | 0 | 450 | 0 | 241 |
| S3_spec_s3d4_ssd4g | pass1 | 798 | 613 | 1.30 | 1.48 | 674 | 0.443 | 2.32 | 124648 | 4498 | 152020 | 2032 | 0 | 1079 | 0 | 240 |

## Re-run with SSD-blocking accounting (A0/A1, commit `361ea4c`)

Same RAM-regime settings as M0/M1 (host 33 GiB, VRAM 19456 MiB, paper policy, transient 24). Outputs byte-identical to M0/M1 (5/5 each). **M0's 500 s baseline was inflated by an NVMe latency spike** (per-request mean read 16.5 ms and 55.1 ms on requests 3–4 with exactly the same read counts; A0 sees 5.6/8.5 ms). Clean baseline 344 s → MTP 1-step 286 s = **1.20×** (not 1.74×).


| run | pass | gen tokens | wall s | tokens/s (e2e) | decode tok/s (client) | TPOT ms | accept rate | accept len | VRAM hits | RAM hits | SSD misses | pack read GB | io_err | draft calls (layer 43) | identical/5 | mean common prefix chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_base_ram33g_acct | pass1 | 817 | 344 | 2.38 | 2.87 | 349 | - | - | 118213 | 74865 | 34334 | 459 | 0 | 0 | 5 | 614 |
| A1_spec_s1d2_ram33g_acct | pass1 | 806 | 286 | 2.82 | 3.57 | 280 | 0.833 | 1.84 | 100669 | 73906 | 38375 | 513 | 0 | 450 | 0 | 241 |

### SSD-miss share of a forward pass (decode steps only, from `update_trace.jsonl` `blk_ssd_ns`)

## A0_base: 815 decode steps (forward passes), verify batch tok=1
experts requested per step: VRAM 140.4  RAM 88.7  SSD 28.9  (SSD share of requests 11.2 %)
layers with >=1 SSD miss per step: 19.0 of 44

| per forward pass (ms) | mean | median | p90 |
|---|---|---|---|
| step interval (token-step wall) | 354.8 | 344.7 | 424.7 |
| MoE layer 0 enter -> last acquire exit | 340.7 | 333.4 | 412.8 |
| time inside acquire() (all layers) | 109.7 | 101.7 | 167.9 |
| blocked on SSD reads | 63.8 | 58.2 | 111.4 |
|   of which target layers 0-42 | 63.8 | 58.2 | 111.4 |
|   of which draft layer 43 | 0.0 | 0.0 | 0.0 |
| sum of individual SSD read durations | 93.7 | 73.6 | 179.3 |

SSD blocking share of step interval: 18.0 % (ratio of sums), 17.4 % (mean of per-step ratios), median per-step 16.7 %
non-SSD acquire overhead (RAM H2D issue + bookkeeping): 12.9 % of interval; compute+rest: 69.1 %
mean SSD blocking per SSD-missed expert: 2.21 ms (parallel reads: sum of read durations / blocking = 1.47x)
steps with zero SSD misses: 0/815

## A1_spec_s1d2: 446 decode steps (forward passes), verify batch tok=2
experts requested per step: VRAM 217.0  RAM 159.6  SSD 61.3  (SSD share of requests 14.0 %)
layers with >=1 SSD miss per step: 29.9 of 44

| per forward pass (ms) | mean | median | p90 |
|---|---|---|---|
| step interval (token-step wall) | 516.2 | 487.9 | 632.8 |
| MoE layer 0 enter -> last acquire exit | 503.5 | 478.6 | 618.4 |
| time inside acquire() (all layers) | 259.9 | 251.6 | 336.2 |
| blocked on SSD reads | 124.9 | 117.1 | 196.5 |
|   of which target layers 0-42 | 121.4 | 114.5 | 193.1 |
|   of which draft layer 43 | 3.5 | 2.6 | 7.9 |
| sum of individual SSD read durations | 260.9 | 206.7 | 479.0 |

SSD blocking share of step interval: 24.2 % (ratio of sums), 23.3 % (mean of per-step ratios), median per-step 23.9 %
non-SSD acquire overhead (RAM H2D issue + bookkeeping): 26.2 % of interval; compute+rest: 49.7 %
mean SSD blocking per SSD-missed expert: 2.04 ms (parallel reads: sum of read durations / blocking = 2.09x)
steps with zero SSD misses: 0/446


