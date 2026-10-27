#!/usr/bin/env python3
"""Teacher-forced agreement of recorded outputs with the exact (non-speculative) model.

For each request in the given runs' pass1/requests.jsonl, re-tokenise prompt+output,
ask the exact server for the top-1 prediction at every output position
(return_logprob + top_logprobs_num=1 over the input), and count positions where
the recorded token equals the exact argmax. Special tokens (</think>, EOS) are not
in the recorded text; when the exact model predicts one where the text has none,
it is inserted and the sequence re-scored once (a second prefill), and such
positions are excluded from the count.
Usage: agree.py URL out.json run_dir[=label] ...
"""
import json, sys, urllib.request
sys.path.insert(0, "/workspace/bench")
import run_requests as rr
from transformers import AutoTokenizer

URL, OUT = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("/opt/ep/models/deepseek-v4-flash-mtp")
SPECIAL = {tok.convert_tokens_to_ids(t) for t in ("</think>", "<｜end▁of▁sentence｜>")} - {None}

def score(ids, start):
    payload = {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
               "return_logprob": True, "logprob_start_len": max(0, start - 1), "top_logprobs_num": 1}
    req = urllib.request.Request(URL + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        meta = json.load(resp)["meta_info"]
    # input_token_logprobs[i] = (logprob, token_id, text) of input token start-1+i;
    # input_top_logprobs[i] = top-1 list at the same position.
    toks = meta["input_token_logprobs"]; tops = meta["input_top_logprobs"]
    top1 = {}
    for i, (entry, top) in enumerate(zip(toks, tops)):
        pos = max(0, start - 1) + i
        if entry is None or top is None or not top:
            continue
        assert entry[1] == ids[pos], (pos, entry[1], ids[pos])
        top1[pos] = top[0][1]
    return top1

results = {}
for arg in sys.argv[3:]:
    run, _, label = arg.partition("="); label = label or run.rstrip("/").split("/")[-1]
    rows = [json.loads(l) for l in open(f"{run}/pass1/requests.jsonl")]
    per = []
    for (pid, ds, prompt), r in zip(rr.PROMPTS, rows):
        p_ids = tok(rr.CHAT_PREFIX + prompt + rr.CHAT_SUFFIX)["input_ids"]
        o_ids = list(r["output_ids"]) if r.get("output_ids") else tok(r["output"], add_special_tokens=False)["input_ids"]
        ids = p_ids + o_ids; P = len(p_ids)
        inserted = set()
        for _pass in range(3):
            top1 = score(ids, P)
            fix = None
            for pos in range(P, len(ids)):
                if top1.get(pos) in SPECIAL and ids[pos] not in SPECIAL and pos not in inserted:
                    fix = pos; break
            if fix is None:
                break
            ids.insert(fix, top1[fix]); inserted.add(fix)
            inserted = {q + 1 if q > fix else q for q in inserted} | {fix}
        scored = [pos for pos in range(P, len(ids)) if pos in top1 and pos not in inserted]
        agree = [pos for pos in scored if top1[pos] == ids[pos]]
        first_bad = next((pos - P for pos in scored if top1[pos] != ids[pos]), None)
        per.append({"prompt": pid, "scored": len(scored), "agree": len(agree), "first_disagreement": first_bad,
                    "inserted_special": len(inserted),
                    "disagreements": [(pos - P, tok.decode([ids[pos]]), tok.decode([top1[pos]])) for pos in scored if top1[pos] != ids[pos]][:20]})
        print(f"[{label}] {pid}: {len(agree)}/{len(scored)} agree, first disagreement at {first_bad}, inserted {len(inserted)}", flush=True)
    tot = sum(p["scored"] for p in per); ag = sum(p["agree"] for p in per)
    results[label] = {"agree": ag, "scored": tot, "rate": ag / tot if tot else None, "per_prompt": per}
    print(f"[{label}] TOTAL {ag}/{tot} = {100*ag/tot:.2f} %", flush=True)
json.dump(results, open(OUT, "w"), indent=1)
