#!/usr/bin/env python3
"""Serial 10-request Expert Pack benchmark client with per-request stats deltas.

Sends the ten Alpaca/MMLU prompts from the SGLang SSD Expert Pack blog one at a
time (concurrency 1, temperature 0, max 200 new tokens, default EOS handling)
to a running SGLang server, and for every request records:
  * token counts, TTFT / prefill tok/s, decode tok/s (from the streamed meta_info)
  * the delta of the server's expert-pack stats JSON (hits/misses/evictions/
    pack_read_bytes/h2d_bytes/io_errors and, if the miss_tier_probe patch is on,
    the page-cache residency split of misses)
  * the delta of /proc/diskstats sectors read for the NVMe device
Output: <out_dir>/requests.jsonl, <out_dir>/summary.json, <out_dir>/stats_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

CHAT_PREFIX = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"
CHAT_SUFFIX = "<｜Assistant｜><think>"
MMLU = "Answer the multiple-choice question. Select the correct option and briefly explain your answer.\n\nQuestion: "
PROMPTS = [
    ("alpaca-37246", "alpaca", "Summarize the movie “Toy Story”"),
    ("mmlu-abstract_algebra-14", "mmlu", MMLU + "Find the maximum possible order for an element of S_n for n = 10.\nA. 6\nB. 12\nC. 30\nD. 105"),
    ("alpaca-50812", "alpaca", "Given a list of items, suggest an interesting activity.\n\nInput: pencils, paper, markers"),
    ("mmlu-moral_disputes-8315", "mmlu", MMLU + "According to Hardin, the “ratchet effect” refers to the fact that\nA. overpopulation does not affect the number of people who are poor.\nB. overpopulation leads to creation of food banks that help curb poverty rates.\nC. world hunger and poverty leads to recognition of rights not to be hungry.\nD. the use of a world food bank to feed the hungry leads to an escalating series of emergency situations."),
    ("alpaca-9907", "alpaca", "Translate this phrase from Spanish to English: El sol no brilla hoy."),
    ("mmlu-high_school_macroeconomics-3940", "mmlu", MMLU + "The crowding-out effect from government borrowing is best described as\nA. the rightward shift in AD in response to the decreasing interest rates from contractionary fiscal policy.\nB. the leftward shift in AD in response to the rising interest rates from expansionary fiscal policy.\nC. the effect of the President increasing the money supply which decreases real interest rates and increases AD.\nD. the effect on the economy of hearing the chairperson of the central bank say that he or she believes that the economy is in a recession."),
    ("alpaca-40699", "alpaca", "Give an example of a bias that could exist in an AI algorithm."),
    ("mmlu-professional_law-10971", "mmlu", MMLU + "Homeowner owns a property in its natural condition with a house on it. There was no fill of any kind on the property. Neighbor, who owns the adjacent property to the East, built a driveway whose western boundary is along the border of homeowner’s property. The excavator dug the driveway five feet deep. The land began to subside along the line of excavation and about three feet of homeowner’s land fell off into the driveway, making that part of her property useless. Homeowner demanded that neighbor fill in the property to buttress the erosion created. That was not done and the erosion continued to occur. Homeowner sued and asked for an injunction compelling the neighbor to build and maintain a retaining wall. Will the court rule for the plaintiff/homeowner?\nA. Yes, because excavation is an abnormally dangerous activity and neighbor is absolutely liable for any damages caused by the violation.\nB. Yes, because every landowner has a right to the lateral support of the soil in its natural state.\nC. No, because the neighbor did not go onto the adjacent land and confined all excavation to his own land.\nD. No, the right to lateral support is a common law right that has been abrogated by statute in virtually all states so that the right no longer exists."),
    ("alpaca-34440", "alpaca", "Make a menu item for a restaurant that contains the following ingredients.\n\nInput: Salmon, avocado, spinach"),
    ("mmlu-jurisprudence-6660", "mmlu", MMLU + "Which of the following is the strongest argument against ethical relativism’s hostility to human rights?\nA. Utilitarianism\nB. Communitarianism.\nC. Cognitivism.\nD. Positivism."),
]
COUNTERS = (
    "cache_hits", "cache_misses", "cache_evictions", "pack_reads", "pack_read_bytes",
    "pack_read_ns", "h2d_bytes", "io_errors", "fallback_count",
    "miss_pagecache_resident", "miss_pagecache_partial", "miss_pagecache_absent",
    "miss_pagecache_resident_ns", "miss_pagecache_absent_ns", "miss_pagecache_resident_bytes",
    # three-tier counters (branch expert-pack-host-tier)
    "vram_hits", "ram_hits", "ssd_misses", "vram_evictions", "demotions_to_ram", "demotions_resident",
    "demotions_d2h", "demotions_discarded", "ram_evictions", "transient_uses", "promotions", "demotion_d2h_bytes",
)
GAUGES = ("ram_cache_bytes_used", "ram_resident_experts", "vram_retained_experts", "vram_transient_experts",
          "forward_steps", "tier_h2d_latency_hist_us", "acquire_latency_hist_us")


def diskstats(device: str) -> dict:
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if f[2] == device:
            return {"reads": int(f[3]), "read_bytes": int(f[5]) * 512,
                    "read_ms": int(f[6]), "writes": int(f[7]), "write_bytes": int(f[9]) * 512}
    raise SystemExit(f"device {device} not in /proc/diskstats")


def server_rss() -> dict:
    """Peak/current RSS of the SGLang scheduler process (the one holding the expert caches)."""
    import subprocess
    out = {}
    try:
        pid = subprocess.run(["pgrep", "-f", "^sglang::scheduler"], capture_output=True, text=True).stdout.split()
        if pid:
            for line in Path(f"/proc/{pid[0]}/status").read_text().splitlines():
                if line.startswith(("VmRSS", "VmHWM")):
                    k, v = line.split(":", 1)
                    out[k] = int(v.split()[0]) * 1024
            out["pid"] = int(pid[0])
    except Exception as exc:  # best effort
        out["error"] = str(exc)
    return out


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("Cached", "MemFree", "MemAvailable"):
            out[k] = int(v.split()[0]) * 1024
    return out


ZERO_STATS = {k: 0 for k in COUNTERS} | {"route_tokens_by_layer": [0] * 43, "route_calls_by_layer": [0] * 43}


def read_stats(path: Path, retries: int = 50, allow_missing: bool = False) -> dict:
    """The server only writes the stats file at its first flush, so before the very
    first request on a fresh server it legitimately does not exist (all counters 0)."""
    for _ in range(retries):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if allow_missing and not path.exists():
                return dict(ZERO_STATS)
            time.sleep(0.1)
    raise SystemExit(f"could not read stats file {path}")


def wait_for_route_tokens(path: Path, layer0_target: int, timeout: float = 30.0) -> dict:
    """Stats flush is periodic; wait until layer-0 route tokens reach the target."""
    deadline = time.monotonic() + timeout
    while True:
        stats = read_stats(path)
        if stats["route_tokens_by_layer"][0] >= layer0_target:
            return stats
        if time.monotonic() > deadline:
            return stats
        time.sleep(0.2)


def generate(url: str, prompt: str, max_new_tokens: int, timeout: float) -> dict:
    payload = {
        "text": CHAT_PREFIX + prompt + CHAT_SUFFIX,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        "stream": True,
    }
    req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter_ns()
    first = last = None
    completion_tokens = 0
    prompt_tokens = None
    cached_tokens = 0
    output = ""
    output_ids: list = []
    finish = None
    spec_meta = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            now = time.perf_counter_ns()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[6:]
            ev = json.loads(line)
            meta = ev.get("meta_info") or {}
            ct = int(meta.get("completion_tokens", 0))
            if ct > completion_tokens:
                first = first or now
                last = now
                completion_tokens = ct
            if meta.get("prompt_tokens") is not None:
                prompt_tokens = int(meta["prompt_tokens"])
            if meta.get("cached_tokens") is not None:
                cached_tokens = int(meta["cached_tokens"])
            if ev.get("text") is not None:
                output = ev["text"]
            if ev.get("output_ids"):
                ids = list(ev["output_ids"])
                # streaming chunks carry cumulative ids (same as text); tolerate deltas
                output_ids = ids if ids[: len(output_ids)] == output_ids or len(ids) >= len(output_ids) else output_ids + ids
            finish = meta.get("finish_reason", finish)
            for key in ("spec_accept_rate", "spec_accept_length", "spec_verify_ct",
                        "spec_num_correct_drafts", "spec_num_proposed_drafts", "e2e_latency"):
                if meta.get(key) is not None:
                    spec_meta[key] = meta[key]
    total_s = (time.perf_counter_ns() - started) / 1e9
    ttft_s = (first - started) / 1e9
    decode_span = (last - first) / 1e9
    intervals = max(0, completion_tokens - 1)
    return {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,  # radix prefix-cache hits: these prompt tokens are not re-forwarded
        "finish_reason": finish, "output": output, "output_ids": output_ids, "ttft_s": ttft_s, "spec": spec_meta,
        "prefill_tok_s": prompt_tokens / ttft_s if ttft_s > 0 else None,
        "decode_tok_s": intervals / decode_span if decode_span > 0 else None,
        "total_s": total_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--stats-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="nvme0n1")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--request-timeout", type=float, default=7200)
    ap.add_argument("--label", default="run")
    ap.add_argument("--only", type=int, default=0, help="run only the first N prompts")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[: args.only] if args.only else PROMPTS

    jsonl = (args.out_dir / "requests.jsonl").open("w")
    rows = []
    for idx, (pid, dataset, prompt) in enumerate(prompts, 1):
        before = read_stats(args.stats_path, allow_missing=(idx == 1))
        (args.out_dir / f"stats_{idx:02d}_before.json").write_text(json.dumps(before, indent=1))
        d0, m0, t0 = diskstats(args.device), meminfo(), time.time()
        gen = generate(args.url, prompt, args.max_new_tokens, args.request_timeout)
        # forwards per request = prefill(prompt_tokens) + (completion_tokens - 1) decode steps:
        # the last completion token is sampled from the final decode forward and needs no forward of its own.
        target = before["route_tokens_by_layer"][0] + gen["prompt_tokens"] - gen["cached_tokens"] + max(0, gen["completion_tokens"] - 1)
        after = wait_for_route_tokens(args.stats_path, target)
        d1, m1 = diskstats(args.device), meminfo()
        (args.out_dir / f"stats_{idx:02d}_after.json").write_text(json.dumps(after, indent=1))
        delta = {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in COUNTERS if k in after}
        hits, misses = delta["cache_hits"], delta["cache_misses"]
        row = {
            "idx": idx, "prompt_id": pid, "dataset": dataset, "label": args.label,
            "wall_start": t0, **{k: v for k, v in gen.items() if k != "output"},
            "output": gen["output"],
            "stats_delta": delta,
            "hit_rate": hits / (hits + misses) if hits + misses else None,
            "route_tokens_layer0_delta": after["route_tokens_by_layer"][0] - before["route_tokens_by_layer"][0],
            "stats_complete": after["route_tokens_by_layer"][0] >= target,
            "diskstats_delta": {k: d1[k] - d0[k] for k in d0},
            "meminfo_before": m0, "meminfo_after": m1,
            "server_rss_after": server_rss(),
            "resident_experts_after": after.get("resident_experts"),
            "gauges_after": {k: after.get(k) for k in GAUGES if k in after},
            "cache_capacity_experts": after.get("cache_capacity_experts"),
        }
        rows.append(row)
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n"); jsonl.flush()
        print(f"[{idx:02d}] {pid:40s} ptok={gen['prompt_tokens']:4d} ctok={gen['completion_tokens']:3d} "
              f"prefill={gen['prefill_tok_s']:.2f} decode={gen['decode_tok_s']:.3f} "
              f"hits={hits} misses={misses} hit_rate={row['hit_rate']:.3f} "
              f"pack_read={delta['pack_read_bytes']/1e9:.2f}GB disk_read={row['diskstats_delta']['read_bytes']/1e9:.2f}GB "
              f"io_err={delta['io_errors']} complete={row['stats_complete']}", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    agg = {
        "label": args.label, "n": len(rows),
        "sum": {k: sum(r["stats_delta"].get(k, 0) for r in rows) for k in COUNTERS},
        "mean_prefill_tok_s": mean([r["prefill_tok_s"] for r in rows]),
        "mean_decode_tok_s": mean([r["decode_tok_s"] for r in rows]),
        "diskstats_read_bytes_total": sum(r["diskstats_delta"]["read_bytes"] for r in rows),
    }
    for ds in ("alpaca", "mmlu"):
        sub = [r for r in rows if r["dataset"] == ds]
        agg[f"{ds}_mean_prefill_tok_s"] = mean([r["prefill_tok_s"] for r in sub])
        agg[f"{ds}_mean_decode_tok_s"] = mean([r["decode_tok_s"] for r in sub])
    h, m = agg["sum"]["cache_hits"], agg["sum"]["cache_misses"]
    agg["aggregate_hit_rate"] = h / (h + m) if h + m else None
    agg["final_stats"] = read_stats(args.stats_path)
    (args.out_dir / "summary.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({k: v for k, v in agg.items() if k != "final_stats"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
