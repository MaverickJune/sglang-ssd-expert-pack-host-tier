#!/usr/bin/env python3
"""Per-forward-pass decomposition of decode latency into SSD-miss blocking vs everything else.

Reads an update_trace.jsonl written with blk_ssd_ns/rd_ssd_ns (per MoE-layer acquire call).
A 'step' is one target forward pass (layers 0..42) plus, with MTP, the draft layer 43 call that follows.
"""
import json, sys, collections, statistics as st

def load(path):
    steps = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line); steps[r['s']].append(r)
    return steps

def analyze(path, label):
    steps = load(path); ss = sorted(steps)
    rows = []
    for a, b in zip(ss, ss[1:]):
        v = steps[a]
        if any(r['p'] for r in v):
            continue  # skip prefill steps
        t0 = min(r['t_in'] for r in v)
        interval = min(r['t_in'] for r in steps[b]) - t0
        tgt = [r for r in v if r['l'] < 43]; drf = [r for r in v if r['l'] == 43]
        rows.append(dict(
            interval=interval, span=max(r['t_out'] for r in v) - t0,
            acq=sum(r['t_out'] - r['t_in'] for r in v),
            blk=sum(r.get('blk_ssd_ns', 0) for r in v), blk_t=sum(r.get('blk_ssd_ns', 0) for r in tgt),
            blk_d=sum(r.get('blk_ssd_ns', 0) for r in drf),
            rd=sum(r.get('rd_ssd_ns', 0) for r in v),
            n_ssd=sum(r['n_ssd'] for r in v), n_ram=sum(r['n_ram'] for r in v), n_vram=sum(r['n_vram'] for r in v),
            n_ssd_d=sum(r['n_ssd'] for r in drf), tok=max(r['tok'] for r in v), after_prefill=any(steps[a-1][0]['p'] for _ in [0]) if a-1 in steps else False,
            ssd_layers=sum(1 for r in v if r['n_ssd'] > 0),
        ))
    n = len(rows)
    S = lambda k: sum(r[k] for r in rows)
    ms = lambda x: x / 1e6
    out = []
    out.append(f"## {label}: {n} decode steps (forward passes), verify batch tok={st.mode(r['tok'] for r in rows)}")
    out.append(f"experts requested per step: VRAM {S('n_vram')/n:.1f}  RAM {S('n_ram')/n:.1f}  SSD {S('n_ssd')/n:.1f}  (SSD share of requests {100*S('n_ssd')/(S('n_vram')+S('n_ram')+S('n_ssd')):.1f} %)")
    out.append(f"layers with >=1 SSD miss per step: {S('ssd_layers')/n:.1f} of {44 if any(r['n_ssd_d'] or r['blk_d'] for r in rows) or 'l43' else 43}")
    out.append("")
    out.append("| per forward pass (ms) | mean | median | p90 |")
    out.append("|---|---|---|---|")
    for k, name in (('interval', 'step interval (token-step wall)'), ('span', 'MoE layer 0 enter -> last acquire exit'),
                    ('acq', 'time inside acquire() (all layers)'), ('blk', 'blocked on SSD reads'), ('blk_t', '  of which target layers 0-42'),
                    ('blk_d', '  of which draft layer 43'), ('rd', 'sum of individual SSD read durations')):
        vals = sorted(ms(r[k]) for r in rows)
        out.append(f"| {name} | {st.mean(vals):.1f} | {st.median(vals):.1f} | {vals[int(0.9*(n-1))]:.1f} |")
    out.append("")
    rs = 100 * S('blk') / S('interval'); rm = 100 * st.mean(r['blk'] / r['interval'] for r in rows)
    out.append(f"SSD blocking share of step interval: {rs:.1f} % (ratio of sums), {rm:.1f} % (mean of per-step ratios), median per-step {100*st.median(r['blk']/r['interval'] for r in rows):.1f} %")
    out.append(f"non-SSD acquire overhead (RAM H2D issue + bookkeeping): {100*(S('acq')-S('blk'))/S('interval'):.1f} % of interval; compute+rest: {100*(S('interval')-S('acq'))/S('interval'):.1f} %")
    out.append(f"mean SSD blocking per SSD-missed expert: {ms(S('blk'))/max(1,S('n_ssd')):.2f} ms (parallel reads: sum of read durations / blocking = {S('rd')/max(1,S('blk')):.2f}x)")
    steps_no_ssd = sum(1 for r in rows if r['n_ssd'] == 0)
    out.append(f"steps with zero SSD misses: {steps_no_ssd}/{n}")
    return "\n".join(out), rows

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        label, path = arg.split('=', 1)
        text, _ = analyze(path, label); print(text); print()
