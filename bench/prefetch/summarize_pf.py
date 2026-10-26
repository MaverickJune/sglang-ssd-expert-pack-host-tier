#!/usr/bin/env python3
"""Markdown summary of prefetch-depth runs: one row per (run, pass)."""
import json, glob, os, sys, statistics as st

def pct(a, b): return f"{100*a/b:.1f}%" if b else "-"

def trace_stats(path):
    n = w = 0; wn = 0; gaps = []; vram_acq = []; wr = ws = 0
    prev = None
    for l in open(path):
        u = json.loads(l)
        if u["p"]:
            prev = None; continue
        n += 1; w += u["pf_wait_ns"] > 0; wn += u["pf_wait_ns"]
        wr += u.get("pf_wait_ram_ns", 0) > 0; ws += u.get("pf_wait_ssd_ns", 0) > 0
        if prev is not None and u["s"] == prev["s"] and u["l"] == prev["l"] + 1:
            gaps.append((u["t_in"] - prev["t_out"]) / 1e6)
        if u["n_ram"] == 0 and u["n_ssd"] == 0:
            vram_acq.append((u["t_out"] - u["t_in"]) / 1e6)
        prev = u
    return dict(n=n, waited=w, waited_ram=wr, waited_ssd=ws, mean_wait_us=wn / max(n, 1) / 1e3,
                gap_p50=st.median(gaps) if gaps else None,
                vram_acq_p50=st.median(vram_acq) if vram_acq else None)

rows = []
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip("/"))
    tr = trace_stats(f"{d}/update_trace.jsonl")
    for p in sorted(glob.glob(f"{d}/pass*")):
        b = json.load(open(f"{p}/stats_01_before.json")); a = json.load(open(f"{p}/stats_05_after.json"))
        D = lambda k: a.get(k, 0) - b.get(k, 0)
        reqs = [json.loads(l) for l in open(f"{p}/requests.jsonl")]
        tpot = st.mean(1000 / r["decode_tok_s"] for r in reqs)
        dec = st.mean(r["decode_tok_s"] for r in reqs)
        src = ""
        if "prefetch_used_ready_ram" in a:
            rr, wr_ = D("prefetch_used_ready_ram"), D("prefetch_used_waited_ram")
            rs, ws_ = D("prefetch_used_ready_ssd"), D("prefetch_used_waited_ssd")
            src = (f"{pct(wr_, rr+wr_)} ({D('prefetch_wait_ns_ram')/1e6/max(wr_,1):.2f} ms) | "
                   f"{pct(ws_, rs+ws_)} ({D('prefetch_wait_ns_ssd')/1e6/max(ws_,1):.2f} ms)")
        rows.append(dict(run=name, k=a.get("prefetch_depth", 0), p=os.path.basename(p), tpot=tpot, dec=dec,
                         issued=D("prefetch_issued"), fr=D("prefetch_from_ram"), fs=D("prefetch_from_ssd"),
                         ready=D("prefetch_used_ready"), waited=D("prefetch_used_waited"),
                         wait_ms=D("prefetch_wait_ns") / 1e6, src=src, ram=D("ram_hits"), ssd=D("ssd_misses"),
                         pack=D("pack_read_bytes") / 1e9, io=D("io_errors"), tr=tr))
print("| run | k | pass | TPOT ms | decode tok/s | prefetch issued (RAM / SSD src) | keys waited | wait/stalled key ms | waited RAM-src (mean) \\| SSD-src (mean) | decode layer-updates stalled | RAM hits / SSD misses (unprefetched) | pack read GB | io_err |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    t = r["tr"]
    print(f"| {r['run']} | {r['k']} | {r['p']} | {r['tpot']:.0f} | {r['dec']:.2f} | {r['issued']} ({r['fr']} / {r['fs']}) | "
          f"{pct(r['waited'], r['ready']+r['waited'])} | {r['wait_ms']/max(r['waited'],1):.2f} | {r['src'] or '-'} | "
          f"{pct(t['waited'], t['n'])} ({t['mean_wait_us']:.0f} us/update) | {r['ram']} / {r['ssd']} | {r['pack']:.0f} | {r['io']} |")
print()
for r in rows:
    if r["p"] == "pass1":
        t = r["tr"]
        print(f"{r['run']}: decode layer gap p50 {t['gap_p50']:.2f} ms, VRAM-only acquire p50 {t['vram_acq_p50']:.3f} ms, decode updates {t['n']}, stalled on RAM-src {t['waited_ram']}, on SSD-src {t['waited_ssd']}")
