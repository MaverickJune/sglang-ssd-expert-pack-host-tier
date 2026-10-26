# Sweep 1 (commit 77d1997, aggregate counters only)

| run | k | pass | TPOT ms | decode tok/s | prefetch issued (RAM / SSD src) | keys waited | wait/stalled key ms | waited RAM-src (mean) \| SSD-src (mean) | decode layer-updates stalled | RAM hits / SSD misses (unprefetched) | pack read GB | io_err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REC_ram33g_k0 | 0 | pass1 | 326 | 3.10 | 0 (0 / 0) | - | 0.00 | - | 0.0% (0 us/update) | 72382 / 33524 | 448 | 0 |
| REC_ram33g_k0 | 0 | pass2 | 323 | 3.13 | 0 (0 / 0) | - | 0.00 | - | 0.0% (0 us/update) | 70836 / 31352 | 419 | 0 |
| SSD_host4g_k0 | 0 | pass1 | 516 | 1.96 | 0 (0 / 0) | - | 0.00 | - | 0.0% (0 us/update) | 26093 / 79813 | 1067 | 0 |
| SSD_host4g_k1 | 1 | pass1 | 292 | 3.46 | 106498 (25244 / 81254) | 24.6% | 30.21 | - | 33.9% (570 us/update) | 0 / 0 | 1086 | 0 |
| SSD_host4g_k2 | 2 | pass1 | 262 | 3.82 | 104279 (23564 / 80715) | 5.0% | 92.69 | - | 1.6% (52 us/update) | 0 / 0 | 1079 | 0 |
| SSD_host4g_k4 | 4 | pass1 | 247 | 4.05 | 101912 (21822 / 80090) | 2.8% | 67.11 | - | 0.7% (29 us/update) | 0 / 0 | 1071 | 0 |
| SSD_host4g_k8 | 8 | pass1 | 237 | 4.22 | 97109 (18770 / 78339) | 0.9% | 66.08 | - | 0.5% (21 us/update) | 0 / 0 | 1047 | 0 |
| RAM_host33g_k1 | 1 | pass1 | 259 | 3.87 | 106498 (72661 / 33837) | 5.9% | 72.76 | - | 4.5% (53 us/update) | 0 / 0 | 452 | 0 |
| RAM_host33g_k1 | 1 | pass2 | 250 | 4.06 | 99385 (67225 / 32160) | 6.5% | 71.54 | - | 4.5% (53 us/update) | 0 / 0 | 430 | 0 |
| RAM_host33g_k2 | 2 | pass1 | 302 | 3.34 | 104279 (69790 / 34489) | 2.2% | 98.91 | - | 0.0% (1 us/update) | 0 / 0 | 461 | 0 |
| RAM_host33g_k2 | 2 | pass2 | 288 | 3.52 | 97076 (64037 / 33039) | 2.1% | 101.46 | - | 0.0% (1 us/update) | 0 / 0 | 442 | 0 |
| RAM_host33g_k4 | 4 | pass1 | 238 | 4.25 | 101907 (66781 / 35126) | 2.7% | 117.69 | - | 0.0% (1 us/update) | 5 / 16 | 470 | 0 |
| RAM_host33g_k4 | 4 | pass2 | 279 | 3.63 | 94853 (60910 / 33943) | 2.4% | 99.59 | - | 0.0% (1 us/update) | 0 / 0 | 454 | 0 |
| RAM_host33g_k8 | 8 | pass1 | 253 | 4.01 | 96236 (61163 / 35073) | 0.9% | 80.16 | - | 0.0% (2 us/update) | 76 / 1432 | 489 | 0 |
| RAM_host33g_k8 | 8 | pass2 | 238 | 4.21 | 90662 (56431 / 34231) | 1.2% | 80.58 | - | 0.0% (2 us/update) | 12 / 1211 | 474 | 0 |

REC_ram33g_k0: decode layer gap p50 5.01 ms, VRAM-only acquire p50 0.202 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 0
SSD_host4g_k0: decode layer gap p50 6.29 ms, VRAM-only acquire p50 0.231 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 0
SSD_host4g_k1: decode layer gap p50 5.10 ms, VRAM-only acquire p50 0.730 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 0
SSD_host4g_k2: decode layer gap p50 5.10 ms, VRAM-only acquire p50 0.537 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 0
SSD_host4g_k4: decode layer gap p50 4.81 ms, VRAM-only acquire p50 0.504 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 0
SSD_host4g_k8: decode layer gap p50 4.55 ms, VRAM-only acquire p50 0.499 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 0
RAM_host33g_k1: decode layer gap p50 4.91 ms, VRAM-only acquire p50 0.518 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 0
RAM_host33g_k2: decode layer gap p50 5.77 ms, VRAM-only acquire p50 0.605 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 0
RAM_host33g_k4: decode layer gap p50 4.97 ms, VRAM-only acquire p50 0.522 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 0
RAM_host33g_k8: decode layer gap p50 4.63 ms, VRAM-only acquire p50 0.527 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 0

# Sweep 2 (commit 8a3bd9e, per-source counters)

| run | k | pass | TPOT ms | decode tok/s | prefetch issued (RAM / SSD src) | keys waited | wait/stalled key ms | waited RAM-src (mean) \| SSD-src (mean) | decode layer-updates stalled | RAM hits / SSD misses (unprefetched) | pack read GB | io_err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSDv2_host4g_k1 | 1 | pass1 | 297 | 3.39 | 106498 (25244 / 81254) | 21.4% | 1.56 | 0.0% (0.00 ms) | 28.0% (1.56 ms) | 29.5% (480 us/update) | 0 / 0 | 1086 | 0 |
| SSDv2_host4g_k2 | 2 | pass1 | 256 | 3.92 | 104279 (23564 / 80715) | 5.2% | 3.36 | 0.0% (0.00 ms) | 6.7% (3.36 ms) | 1.8% (62 us/update) | 0 / 0 | 1079 | 0 |
| SSDv2_host4g_k3 | 3 | pass1 | 244 | 4.10 | 102977 (22621 / 80356) | 4.4% | 3.68 | 0.0% (0.00 ms) | 5.6% (3.68 ms) | 1.2% (43 us/update) | 0 / 0 | 1074 | 0 |
| SSDv2_host4g_k4 | 4 | pass1 | 244 | 4.10 | 101912 (21822 / 80090) | 3.1% | 3.46 | 0.0% (0.00 ms) | 3.9% (3.46 ms) | 1.0% (41 us/update) | 0 / 0 | 1071 | 0 |
| SSDv2_host4g_k6 | 6 | pass1 | 234 | 4.29 | 99532 (20354 / 79178) | 2.0% | 3.39 | 0.0% (0.00 ms) | 2.5% (3.39 ms) | 0.9% (31 us/update) | 0 / 0 | 1059 | 0 |
| RAMv2_host33g_k1 | 1 | pass1 | 255 | 3.99 | 106498 (72661 / 33837) | 7.1% | 2.85 | 0.0% (0.65 ms) | 22.3% (2.85 ms) | 6.9% (316 us/update) | 0 / 0 | 452 | 0 |
| RAMv2_host33g_k1 | 1 | pass2 | 265 | 3.82 | 99385 (67225 / 32160) | 7.8% | 4.01 | 0.0% (3.80 ms) | 24.2% (4.01 ms) | 6.9% (316 us/update) | 0 / 0 | 430 | 0 |
| RAMv2_host33g_k2 | 2 | pass1 | 234 | 4.28 | 104279 (69790 / 34489) | 2.7% | 3.81 | 0.0% (0.00 ms) | 8.2% (3.81 ms) | 0.2% (9 us/update) | 0 / 0 | 461 | 0 |
| RAMv2_host33g_k2 | 2 | pass2 | 206 | 4.87 | 97076 (64037 / 33039) | 3.5% | 3.69 | 0.0% (0.00 ms) | 10.2% (3.69 ms) | 0.2% (9 us/update) | 0 / 0 | 442 | 0 |
| RAMv2_host33g_k3 | 3 | pass1 | 211 | 4.74 | 102977 (68126 / 34851) | 2.9% | 3.95 | 0.0% (0.00 ms) | 8.6% (3.95 ms) | 0.2% (6 us/update) | 0 / 0 | 466 | 0 |
| RAMv2_host33g_k3 | 3 | pass2 | 206 | 4.86 | 95815 (62294 / 33521) | 3.1% | 4.16 | 0.0% (0.00 ms) | 8.8% (4.16 ms) | 0.2% (6 us/update) | 0 / 0 | 449 | 0 |
| RAMv2_host33g_k4 | 4 | pass1 | 204 | 4.91 | 101912 (66772 / 35140) | 3.2% | 3.61 | 0.0% (0.00 ms) | 9.4% (3.61 ms) | 0.1% (6 us/update) | 0 / 0 | 470 | 0 |
| RAMv2_host33g_k4 | 4 | pass2 | 204 | 4.92 | 94909 (61043 / 33866) | 3.3% | 3.78 | 0.0% (0.00 ms) | 9.3% (3.78 ms) | 0.1% (6 us/update) | 8 / 21 | 454 | 0 |

SSDv2_host4g_k1: decode layer gap p50 5.49 ms, VRAM-only acquire p50 0.727 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 10368
SSDv2_host4g_k2: decode layer gap p50 5.00 ms, VRAM-only acquire p50 0.543 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 649
SSDv2_host4g_k3: decode layer gap p50 4.72 ms, VRAM-only acquire p50 0.507 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 406
SSDv2_host4g_k4: decode layer gap p50 4.68 ms, VRAM-only acquire p50 0.514 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 349
SSDv2_host4g_k6: decode layer gap p50 4.47 ms, VRAM-only acquire p50 0.487 ms, decode updates 35088, stalled on RAM-src 0, on SSD-src 305
RAMv2_host33g_k1: decode layer gap p50 4.63 ms, VRAM-only acquire p50 0.532 ms, decode updates 70090, stalled on RAM-src 2, on SSD-src 4801
RAMv2_host33g_k2: decode layer gap p50 4.17 ms, VRAM-only acquire p50 0.481 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 166
RAMv2_host33g_k3: decode layer gap p50 4.09 ms, VRAM-only acquire p50 0.450 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 106
RAMv2_host33g_k4: decode layer gap p50 3.93 ms, VRAM-only acquire p50 0.475 ms, decode updates 70090, stalled on RAM-src 0, on SSD-src 77
