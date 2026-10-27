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

