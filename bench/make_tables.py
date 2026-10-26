#!/usr/bin/env python3
"""Render markdown tables from run_requests.py output (requests.jsonl)."""
import json, sys
from pathlib import Path

def gb(x): return f"{x/1e9:.2f}"
def f(x, d=2): return "n/a" if x is None else f"{x:.{d}f}"

def table(path: Path, probe: bool):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    hdr = ["#", "prompt id", "prompt tok", "compl tok", "prefill tok/s", "decode tok/s", "hits", "misses"]
    if probe:
        hdr += ["miss: pc-resident", "miss: pc-absent", "miss: partial"]
    hdr += ["hit rate", "evictions", "pack_read GB", "h2d GB", "diskstats read GB", "io_errors", "mean read ms"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        d = r["stats_delta"]; dk = r["diskstats_delta"]
        cells = [str(r["idx"]), r["prompt_id"], str(r["prompt_tokens"]), str(r["completion_tokens"]),
                 f(r["prefill_tok_s"]), f(r["decode_tok_s"], 3), str(d["cache_hits"]), str(d["cache_misses"])]
        if probe:
            cells += [str(d.get("miss_pagecache_resident", 0)), str(d.get("miss_pagecache_absent", 0)), str(d.get("miss_pagecache_partial", 0))]
        mean_ms = d["pack_read_ns"] / d["pack_reads"] / 1e6 if d.get("pack_reads") else None
        cells += [f(r["hit_rate"], 3), str(d["cache_evictions"]), gb(d["pack_read_bytes"]), gb(d["h2d_bytes"]), gb(dk["read_bytes"]), str(d["io_errors"]), f(mean_ms, 1)]
        out.append("| " + " | ".join(cells) + " |")
    # aggregate
    S = lambda k: sum(r["stats_delta"].get(k, 0) for r in rows)
    h, m = S("cache_hits"), S("cache_misses")
    agg = {
        "requests": len(rows), "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "hits": h, "misses": m, "hit_rate": h / (h + m) if h + m else None,
        "evictions": S("cache_evictions"), "pack_read_bytes": S("pack_read_bytes"), "h2d_bytes": S("h2d_bytes"),
        "diskstats_read_bytes": sum(r["diskstats_delta"]["read_bytes"] for r in rows), "io_errors": S("io_errors"),
        "mean_prefill_tok_s": sum(r["prefill_tok_s"] for r in rows) / len(rows),
        "mean_decode_tok_s": sum(r["decode_tok_s"] for r in rows) / len(rows),
    }
    for ds in ("alpaca", "mmlu"):
        sub = [r for r in rows if r["dataset"] == ds]
        if sub:
            agg[f"{ds}_mean_prefill_tok_s"] = sum(r["prefill_tok_s"] for r in sub) / len(sub)
            agg[f"{ds}_mean_decode_tok_s"] = sum(r["decode_tok_s"] for r in sub) / len(sub)
    if probe:
        res, ab = S("miss_pagecache_resident"), S("miss_pagecache_absent")
        agg.update({"miss_pagecache_resident": res, "miss_pagecache_absent": ab, "miss_pagecache_partial": S("miss_pagecache_partial"),
                    "mean_read_ms_resident": S("miss_pagecache_resident_ns") / res / 1e6 if res else None,
                    "mean_read_ms_absent": S("miss_pagecache_absent_ns") / ab / 1e6 if ab else None})
    return "\n".join(out), agg

if __name__ == "__main__":
    t, agg = table(Path(sys.argv[1]), probe=True)
    print(t); print(); print(json.dumps(agg, indent=2))
