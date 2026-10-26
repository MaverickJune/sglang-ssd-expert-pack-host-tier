#!/usr/bin/env python3
"""Build the draft (MTP / NextN) assets for DeepSeek-V4-Flash Expert Pack serving.

Inputs : HF shard model-00046-of-00046.safetensors (all `mtp.0.*` tensors) and the
         verified model-meta config.json used by the target.
Outputs: <out>/DeepSeek-V4-Flash-MTP.gguf          routed experts of the MTP layer, MXFP4,
                                                     named blk.0.ffn_{gate,up,down}_exps.weight
         <out>/DeepSeek-V4-Flash-MTP.expert-pack    1-layer Expert Pack built by the stock builder
         <out>/mtp-dense.safetensors                 dense MTP tensors, HF names (mtp.0.*):
                                                     fp8 linears -> GGUF Q8_0 (.qweight/.qweight_type),
                                                     everything else passed through
         <out>/config.json, tokenizer*               copies for the draft model directory
         <out>/draft_assets.json                     paths + digests

The HF fp4 experts (e2m1 nibbles, e8m0 block-32 scales) are *repacked*, not
requantized: GGUF MXFP4 keeps the same scale byte per block and the same 32
e2m1 codes, only the nibble order differs (HF: elements (2j, 2j+1) share a byte;
GGUF: elements (j, j+16) share a byte).
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, struct, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as QT
from safetensors.torch import save_file

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def st_open(path: Path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return header, 8 + n, mm


def st_raw(header, base, mm, name) -> np.ndarray:
    s, e = header[name]["data_offsets"]
    return np.asarray(mm[base + s : base + e])


def repack_fp4_to_mxfp4(nibbles: np.ndarray, scale: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """HF fp4 [rows, cols/2] bytes + e8m0 scale [rows, cols/32] -> GGUF MXFP4 bytes [rows, cols/32*17]."""
    nib = nibbles.reshape(rows, cols // 32, 16)
    lo = nib & 0xF
    hi = nib >> 4
    elems = np.empty((rows, cols // 32, 32), dtype=np.uint8)
    elems[..., 0::2] = lo
    elems[..., 1::2] = hi
    qs = (elems[..., :16] | (elems[..., 16:] << 4)).astype(np.uint8)
    out = np.empty((rows, cols // 32, 17), dtype=np.uint8)
    out[..., 0] = scale.reshape(rows, cols // 32)
    out[..., 1:] = qs
    return out.reshape(rows, cols // 32 * 17)


def deq_hf_fp4(nibbles, scale, rows, cols):
    nib = nibbles.reshape(rows, cols // 32, 16)
    e = np.empty((rows, cols // 32, 32), dtype=np.uint8)
    e[..., 0::2] = nib & 0xF
    e[..., 1::2] = nib >> 4
    return (E2M1[e] * np.exp2(scale.reshape(rows, cols // 32).astype(np.float32) - 127)[..., None]).reshape(rows, cols)


def deq_gguf_mxfp4(blocks, rows, cols):
    b = blocks.reshape(rows, cols // 32, 17)
    e = np.concatenate([b[..., 1:] & 0xF, b[..., 1:] >> 4], axis=-1)
    return (E2M1[e] * np.exp2(b[..., 0].astype(np.float32) - 127)[..., None]).reshape(rows, cols)


def deq_fp8_block(weight_bytes: np.ndarray, scale_bytes: np.ndarray, shape, scale_shape) -> np.ndarray:
    w = torch.frombuffer(bytearray(weight_bytes.tobytes()), dtype=torch.float8_e4m3fn).float().reshape(shape)
    s = torch.from_numpy(scale_bytes.astype(np.float32)).reshape(scale_shape)
    s = torch.exp2(s - 127.0)
    bm, bn = shape[0] // scale_shape[0], shape[1] // scale_shape[1]
    s = s.repeat_interleave(bm, 0).repeat_interleave(bn, 1)
    return (w * s).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, default=Path("/opt/ep/models/deepseek-v4-flash-hf-mtp/model-00046-of-00046.safetensors"))
    ap.add_argument("--model-meta", type=Path, default=Path("/root/.cache/sglang-expert-pack/deepseek-v4-flash/87d1fbae3308fd145a1d/model-meta"))
    ap.add_argument("--out", type=Path, default=Path("/opt/ep/models/deepseek-v4-flash-mtp"))
    ap.add_argument("--sglang-root", type=Path, default=Path("/opt/ep/sglang"))
    ap.add_argument("--num-experts", type=int, default=256)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    header, base, mm = st_open(a.shard)
    names = sorted(header)
    assert all(n.startswith("mtp.0.") for n in names), "shard must hold only mtp.0.* tensors"

    # ---- experts -> mini GGUF (MXFP4) ----
    roles = {"gate": "w1", "up": "w3", "down": "w2"}
    gguf_path = a.out / "DeepSeek-V4-Flash-MTP.gguf"
    writer = gguf.GGUFWriter(str(gguf_path), arch="deepseek4")
    writer.add_name("DeepSeek-V4-Flash MTP layer (routed experts, MXFP4 repacked from HF fp4)")
    for role, hf in roles.items():
        per_expert = []
        rows = cols = None
        for e in range(a.num_experts):
            wn, sn = f"mtp.0.ffn.experts.{e}.{hf}.weight", f"mtp.0.ffn.experts.{e}.{hf}.scale"
            wshape, sshape = header[wn]["shape"], header[sn]["shape"]
            assert header[wn]["dtype"] == "I8" and header[sn]["dtype"] == "F8_E8M0"
            r, c = wshape[0], wshape[1] * 2
            assert sshape == [r, c // 32], (wn, wshape, sshape)
            rows, cols = r, c
            blk = repack_fp4_to_mxfp4(st_raw(header, base, mm, wn), st_raw(header, base, mm, sn), r, c)
            if e == 0:  # lossless-repack self check
                ref = deq_hf_fp4(st_raw(header, base, mm, wn), st_raw(header, base, mm, sn), r, c)
                got = deq_gguf_mxfp4(blk, r, c)
                assert np.array_equal(ref, got), f"repack mismatch for {wn}"
            per_expert.append(blk)
        stacked = np.stack(per_expert, axis=0)  # [E, rows, cols/32*17]
        writer.add_tensor(f"blk.0.ffn_{role}_exps.weight", stacked, raw_dtype=QT.MXFP4)  # byte shape -> logical [E, rows, cols]
        print(f"experts {role}: HF {hf} rows={rows} cols={cols} bytes/expert={stacked.shape[-1]}", flush=True)
        del per_expert, stacked
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    gguf_sha = hashlib.sha256(gguf_path.read_bytes()).hexdigest()
    print(f"gguf {gguf_path} sha256={gguf_sha} size={gguf_path.stat().st_size}", flush=True)

    # ---- dense tensors -> safetensors (HF names) ----
    dense: dict[str, torch.Tensor] = {}
    consumed = set()
    scales = {n for n in names if header[n]["dtype"] == "F8_E8M0" and ".ffn.experts." not in n}
    for n in names:
        if ".ffn.experts." in n or n in scales:
            continue
        meta = header[n]
        raw = st_raw(header, base, mm, n)
        if meta["dtype"] == "F8_E4M3":
            sn = n.removesuffix(".weight") + ".scale"
            assert sn in header and header[sn]["dtype"] == "F8_E8M0", n
            consumed.add(sn)
            f32 = deq_fp8_block(raw, st_raw(header, base, mm, sn), meta["shape"], header[sn]["shape"])
            q = gguf.quants.quantize(f32, QT.Q8_0)
            # Q8_0 round-trip check
            back = gguf.quants.dequantize(q, QT.Q8_0)
            rel = float(np.abs(back - f32).max() / (np.abs(f32).max() + 1e-12))
            dense[n.removesuffix(".weight") + ".qweight"] = torch.from_numpy(np.ascontiguousarray(q))
            dense[n.removesuffix(".weight") + ".qweight_type"] = torch.tensor(int(QT.Q8_0), dtype=torch.uint8)
            print(f"dense Q8_0 {n} shape={meta['shape']} max_rel_err={rel:.2e}", flush=True)
        elif meta["dtype"] == "F8_E8M0":
            raise AssertionError(f"orphan scale {n}")
        elif meta["dtype"] == "BF16":
            dense[n] = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(meta["shape"])
        elif meta["dtype"] == "F32":
            dense[n] = torch.from_numpy(raw.copy()).view(torch.float32).reshape(meta["shape"])
        else:
            raise AssertionError(f"unexpected dtype {meta['dtype']} for {n}")
    orphan = scales - consumed
    assert not orphan, f"orphan scales: {sorted(orphan)}"
    dense_path = a.out / "mtp-dense.safetensors"
    save_file(dense, str(dense_path), metadata={"format": "pt", "source_shard": a.shard.name})
    print(f"dense tensors: {len(dense)} -> {dense_path}", flush=True)

    # ---- draft model dir: config + tokenizer ----
    for fn in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = a.model_meta / fn
        if src.exists():
            shutil.copy(src, a.out / fn)
    config_path = a.out / "config.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model_identity = hashlib.sha256(f"sglang-deepseek-expert-pack-v1:{gguf_sha}:{config_sha}".encode()).hexdigest()
    cfg = json.loads(config_path.read_text())

    # ---- 1-layer expert pack via the stock builder ----
    pack = a.out / "DeepSeek-V4-Flash-MTP.expert-pack"
    manifest = Path(str(pack) + ".manifest.json")
    checkpoint = Path(str(pack) + ".checkpoint.json")
    for p in (pack, manifest, checkpoint, Path(str(pack) + ".partial")):
        if p.exists():
            p.unlink()
    build = a.sglang_root / "python/sglang/srt/model_loader/expert_pack/build.py"
    cmd = [sys.executable, str(build), "--source", str(gguf_path), "--source-sha256", gguf_sha,
           "--output", str(pack), "--manifest", str(manifest), "--checkpoint", str(checkpoint),
           "--model-ref", "deepseek-v4-flash-mtp", "--model-identity-sha256", model_identity,
           "--config-blob", str(config_path), "--config-sha256", config_sha,
           "--num-layers", "1", "--num-experts", str(a.num_experts), "--top-k", str(cfg["num_experts_per_tok"]),
           "--alignment", "4096", "--safety-margin-gib", "1"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(build.parent))
    out = {"draft_dir": str(a.out), "mtp_gguf": str(gguf_path), "mtp_gguf_sha256": gguf_sha,
           "mtp_pack_path": str(pack), "mtp_manifest_path": str(manifest), "mtp_dense_path": str(dense_path),
           "config_sha256": config_sha, "model_identity_sha256": model_identity,
           "source_shard": str(a.shard), "source_shard_sha256": hashlib.sha256(a.shard.read_bytes()).hexdigest()}
    (a.out / "draft_assets.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
