#!/usr/bin/env python3
"""Resumable parallel-range download of the DeepSeek-V4-Flash MXFP4 GGUF blob
from the Ollama registry (no ollama install needed), then SHA-256 verification
against the digest recorded in the SGLang blog / Ollama manifest."""
import argparse, hashlib, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLOB = "947ac34c08c0e5c5752ac76398f934b3b6b4075cfe915ba43dd5ac754900a4cd"
URL = f"https://registry.ollama.ai/v2/frob/deepseek-v4-flash-0731/blobs/sha256:{BLOB}"
CHUNK = 256 * 1024 * 1024


def fetch(url, start, end, fd, retries=20):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise OSError(f"expected 206, got {r.status}")
                off = start
                while True:
                    buf = r.read(8 << 20)
                    if not buf:
                        break
                    os.pwrite(fd, buf, off)
                    off += len(buf)
                if off != end + 1:
                    raise OSError(f"short range {start}-{end}: got {off-start}")
            return
        except Exception as exc:
            time.sleep(min(60, 2 ** attempt))
            last = exc
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/opt/ep/models/deepseek-v4-flash-0731/deepseek-v4-flash-0731.gguf"))
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    state = a.out.with_suffix(".gguf.chunks.json")
    if not a.verify_only:
        with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
            size = int(r.headers["Content-Length"]); url = r.url
        print(f"size={size} ({size/1e9:.2f} GB) streams={a.streams}", flush=True)
        done = set(json.loads(state.read_text())) if state.exists() else set()
        fd = os.open(a.out, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, size)
        chunks = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK)) if i not in done]
        lock = threading.Lock(); t0 = time.time(); got = [0]
        def work(c):
            i, s, e = c
            fetch(url, s, e, fd)
            with lock:
                done.add(i); got[0] += e - s + 1
                state.write_text(json.dumps(sorted(done)))
                if len(done) % 20 == 0:
                    print(f"{len(done)}/{(size+CHUNK-1)//CHUNK} chunks, {got[0]/(time.time()-t0)/1e6:.0f} MB/s", flush=True)
        with ThreadPoolExecutor(a.streams) as ex:
            list(ex.map(work, chunks))
        os.fsync(fd); os.close(fd)
        print(f"download complete in {time.time()-t0:.0f}s", flush=True)
    h = hashlib.sha256()
    with a.out.open("rb", buffering=0) as f:
        while blk := f.read(64 << 20):
            h.update(blk)
    print("sha256", h.hexdigest(), "expected", BLOB)
    if h.hexdigest() != BLOB:
        print("SHA-256 MISMATCH", file=sys.stderr); return 1
    state.unlink(missing_ok=True)
    print("SHA-256 OK"); return 0


if __name__ == "__main__":
    sys.exit(main())
