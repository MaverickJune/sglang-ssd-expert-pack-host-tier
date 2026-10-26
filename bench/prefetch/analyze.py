#!/usr/bin/env python3
"""Analyze prefetch experiments: per-update acquire stall by tier, per-layer compute gap,
prefetch readiness, and TPOT per pass. Usage: analyze.py results/<run> [...]"""
import json, statistics, sys
from pathlib import Path

def pct(xs, q):
    xs = sorted(xs); 
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

def analyze(run: Path):
    upd = [json.loads(l) for l in (run / "update_trace.jsonl").read_text().splitlines() if l.strip()]
    stats = json.loads((run / "stats_final.json").read_text())
    passes = sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("pass"))
    out = {"run": run.name, "prefetch_depth": stats.get("prefetch_depth", 0), "updates": len(upd)}
    # per pass: TPOT / decode tok/s from requests.jsonl
    for p in passes:
        rows = [json.loads(l) for l in (p / "requests.jsonl").read_text().splitlines() if l.strip()]
        d = {k: sum(r["stats_delta"].get(k, 0) for r in rows) for k in ("vram_hits", "ram_hits", "ssd_misses", "pack_read_bytes", "h2d_bytes", "io_errors",
             "prefetch_issued", "prefetch_from_ram", "prefetch_from_ssd", "prefetch_used_ready", "prefetch_used_waited", "prefetch_wait_ns", "prefetch_unused", "prefetch_skipped_inflight")}
        out[p.name] = {"decode_tok_s": statistics.mean(r["decode_tok_s"] for r in rows), "tpot_ms": statistics.mean(1000 / r["decode_tok_s"] for r in rows),
                       "prefill_tok_s": statistics.mean(r["prefill_tok_s"] for r in rows), "wall_s": sum(r["total_s"] for r in rows),
                       "first_step": None, "outputs": [r["output"] for r in rows], **d}
        if d["prefetch_issued"]:
            out[p.name]["prefetch_ready_rate"] = d["prefetch_used_ready"] / max(1, d["prefetch_used_ready"] + d["prefetch_used_waited"])
            out[p.name]["prefetch_mean_wait_ms"] = d["prefetch_wait_ns"] / max(1, d["prefetch_used_waited"]) / 1e6
    # decode updates only (tok == 1); classify by worst tier; acquire wall = t_out - t_in
    dec = [u for u in upd if u["tok"] == 1]
    by_tier = {"vram": [], "ram": [], "ssd": []}
    for u in dec:
        t = "ssd" if u["n_ssd"] else ("ram" if u["n_ram"] else "vram")
        by_tier[t].append((u["t_out"] - u["t_in"]) / 1e6)
    out["decode_acquire_ms_by_tier"] = {t: {"n": len(v), "mean": statistics.mean(v) if v else None, "p50": pct(v, .5), "p90": pct(v, .9)} for t, v in by_tier.items()}
    # per-layer compute gap: t_in(next update) - t_out(this update) within the same step, decode only
    gaps = []
    by_step = {}
    for u in dec:
        by_step.setdefault(u["s"], []).append(u)
    step_times = []
    for s_, us in by_step.items():
        us.sort(key=lambda u: u["l"])
        for a, b in zip(us, us[1:]):
            gaps.append((b["t_in"] - a["t_out"]) / 1e6)
        if len(us) >= 2:
            step_times.append((us[-1]["t_out"] - us[0]["t_in"]) / 1e6)
    out["decode_layer_gap_ms"] = {"n": len(gaps), "mean": statistics.mean(gaps) if gaps else None, "p50": pct(gaps, .5), "p90": pct(gaps, .9)}
    out["decode_step_span_ms"] = {"n": len(step_times), "mean": statistics.mean(step_times) if step_times else None}
    out["decode_acquire_total_ms_per_step"] = sum(sum(v) for v in by_tier.values()) / max(1, len(by_step))
    pf_waits = [u["pf_wait_ns"] / 1e6 for u in dec if u["pf_wait_ns"]]
    out["decode_prefetch_wait_ms"] = {"n": len(pf_waits), "mean": statistics.mean(pf_waits) if pf_waits else None, "p90": pct(pf_waits, .9)}
    return out

if __name__ == "__main__":
    res = [analyze(Path(a)) for a in sys.argv[1:]]
    for r in res:
        slim = {k: v for k, v in r.items() if not (isinstance(v, dict) and "outputs" in v)}
        for p in ("pass1", "pass2"):
            if p in r:
                slim[p] = {k: v for k, v in r[p].items() if k != "outputs"}
        print(json.dumps(slim, indent=1))
