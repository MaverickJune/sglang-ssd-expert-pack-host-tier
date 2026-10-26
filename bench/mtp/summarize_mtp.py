#!/usr/bin/env python3
"""Markdown summary of MTP runs: throughput, acceptance, tier counters, output agreement."""
import json, glob, os, sys, statistics as st
REF = "/workspace/bench/results/E1_lru_host0/requests.jsonl"
ref = [json.loads(l)["output"] for l in open(REF)]

def agree(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

rows = []
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip("/"))
    for p in sorted(glob.glob(f"{d}/pass*")):
        reqs = [json.loads(l) for l in open(f"{p}/requests.jsonl")]
        b = json.load(open(f"{p}/stats_01_before.json")); a = json.load(open(f"{p}/stats_05_after.json"))
        D = lambda k: a.get(k, 0) - b.get(k, 0)
        tok = sum(r["completion_tokens"] for r in reqs); wall = sum(r["total_s"] for r in reqs)
        dec = st.mean(r["decode_tok_s"] for r in reqs)
        spec = [r.get("spec", {}) for r in reqs]
        acc = [s.get("spec_accept_rate") for s in spec if s.get("spec_accept_rate") is not None]
        alen = [s.get("spec_accept_length") for s in spec if s.get("spec_accept_length") is not None]
        agr = [agree(r["output"], ref[i]) for i, r in enumerate(reqs)]
        ident = sum(r["output"] == ref[i] for i, r in enumerate(reqs))
        rows.append((name, os.path.basename(p), tok, wall, tok / wall, dec, 1000 / dec,
                     st.mean(acc) if acc else None, st.mean(alen) if alen else None,
                     D("vram_hits"), D("ram_hits"), D("ssd_misses"), D("pack_read_bytes") / 1e9, D("io_errors"),
                     a.get("route_calls_by_layer", [0])[-1] - b.get("route_calls_by_layer", [0])[-1] if len(a.get("route_calls_by_layer", [])) > 43 else 0,
                     ident, st.mean(agr)))
print("| run | pass | gen tokens | wall s | tokens/s (e2e) | decode tok/s (client) | TPOT ms | accept rate | accept len | VRAM hits | RAM hits | SSD misses | pack read GB | io_err | draft calls (layer 43) | identical/5 | mean common prefix chars |")
print("|" + "---|" * 17)
for r in rows:
    f = lambda v, fmt: ("-" if v is None else format(v, fmt))
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.0f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.0f} | {f(r[7], '.3f')} | {f(r[8], '.2f')} | {r[9]} | {r[10]} | {r[11]} | {r[12]:.0f} | {r[13]} | {r[14]} | {r[15]} | {r[16]:.0f} |")
