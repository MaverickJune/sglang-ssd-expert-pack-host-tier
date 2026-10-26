#!/usr/bin/env python3
"""Aggregate E0..E4 runs (requests.jsonl each) into the report's comparison tables."""
import json, sys
from pathlib import Path

NEW = ("vram_hits", "ram_hits", "ssd_misses", "vram_evictions", "demotions_to_ram", "ram_evictions", "transient_uses")

def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:   # legacy runs (E0) only have cache_* names
        d = r["stats_delta"]
        d.setdefault("vram_hits", d["cache_hits"]); d.setdefault("ram_hits", 0)
        d.setdefault("ssd_misses", d["cache_misses"]); d.setdefault("vram_evictions", d["cache_evictions"])
        for k in ("demotions_to_ram", "ram_evictions", "transient_uses"): d.setdefault(k, 0)
    return rows

def per_request_table(rows):
    hdr = ["#", "prompt id", "ptok", "ctok", "prefill tok/s", "decode tok/s", "TPOT ms", "vram_hits", "ram_hits", "ssd_misses",
           "vram hit%", "ram hit% of misses", "vram_evict", "demote→RAM", "ram_evict", "transient", "pack_read GB", "h2d GB", "nvme GB"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        d, dk = r["stats_delta"], r["diskstats_delta"]
        tot = d["vram_hits"] + d["ram_hits"] + d["ssd_misses"]; miss = d["ram_hits"] + d["ssd_misses"]
        tpot = 1000 / r["decode_tok_s"] if r["decode_tok_s"] else None
        out.append("| " + " | ".join(map(str, [r["idx"], r["prompt_id"], r["prompt_tokens"], r["completion_tokens"],
            f"{r['prefill_tok_s']:.2f}", f"{r['decode_tok_s']:.3f}", f"{tpot:.0f}" if tpot else "n/a",
            d["vram_hits"], d["ram_hits"], d["ssd_misses"], f"{100*d['vram_hits']/tot:.1f}", f"{100*d['ram_hits']/miss:.1f}" if miss else "n/a",
            d["vram_evictions"], d["demotions_to_ram"], d["ram_evictions"], d["transient_uses"],
            f"{d['pack_read_bytes']/1e9:.1f}", f"{d['h2d_bytes']/1e9:.1f}", f"{dk['read_bytes']/1e9:.1f}"])) + " |")
    return "\n".join(out)

def aggregate(rows, payload):
    S = lambda k: sum(r["stats_delta"].get(k, 0) for r in rows)
    v, rh, sm = S("vram_hits"), S("ram_hits"), S("ssd_misses"); tot = v + rh + sm
    a = {"n": len(rows), "vram_hits": v, "ram_hits": rh, "ssd_misses": sm,
         "vram_hit_rate": v / tot, "ram_hit_rate_of_misses": rh / (rh + sm) if rh + sm else 0.0,
         "overall_non_ssd_rate": (v + rh) / tot,
         "vram_evictions": S("vram_evictions"), "demotions_to_ram": S("demotions_to_ram"), "ram_evictions": S("ram_evictions"),
         "transient_uses": S("transient_uses"),
         "mean_prefill_tok_s": sum(r["prefill_tok_s"] for r in rows) / len(rows),
         "mean_decode_tok_s": sum(r["decode_tok_s"] for r in rows) / len(rows),
         "mean_tpot_ms": sum(1000 / r["decode_tok_s"] for r in rows) / len(rows),
         "pack_read_bytes": S("pack_read_bytes"), "h2d_bytes": S("h2d_bytes"),
         "nvme_read_bytes": sum(r["diskstats_delta"]["read_bytes"] for r in rows),
         "io_errors": S("io_errors"),
         "expected_h2d_bytes": (rh + sm) * payload, "expected_pack_read_bytes": sm * payload,
         "peak_rss_bytes": max((r.get("server_rss_after", {}).get("VmHWM", 0) for r in rows), default=0),
         "total_wall_s": sum(r["total_s"] for r in rows)}
    a["nvme_over_pack_read"] = a["nvme_read_bytes"] / a["pack_read_bytes"] if a["pack_read_bytes"] else None
    return a

if __name__ == "__main__":
    payload = 13369344
    for p in sys.argv[1:]:
        rows = load(p); print("==", p); print(per_request_table(rows)); print(json.dumps(aggregate(rows, payload), indent=1))
